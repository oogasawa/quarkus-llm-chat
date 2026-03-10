/*
 * Copyright 2025 oogasawa
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.github.oogasawa.llmchat.service;

import com.github.oogasawa.llmchat.rest.ChatEvent;
import com.github.oogasawa.llmchat.vllm.ChatMessage;
import com.github.oogasawa.llmchat.vllm.ContextLengthExceededException;
import com.github.oogasawa.llmchat.vllm.VllmClient;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * Multi-tenant chat service that routes prompts to local LLM servers.
 *
 * <p>Each user has an isolated conversation history.
 * Per-user busy state prevents concurrent prompts for the same user.</p>
 *
 * @author oogasawa
 */
@ApplicationScoped
public class ChatService {

    private static final Logger logger = Logger.getLogger(ChatService.class.getName());

    private final ModelSet modelSet;
    private final List<VllmClient> llmClients;
    private final int maxHistory;

    /** Per-user conversation history. */
    private final ConcurrentHashMap<String, List<ChatMessage>> userHistories = new ConcurrentHashMap<>();

    /** Per-user busy flag. */
    private final ConcurrentHashMap<String, Thread> activeThreads = new ConcurrentHashMap<>();

    @Inject
    public ChatService(
            @ConfigProperty(name = "llm-chat.servers", defaultValue = "")
            String llmServers,
            @ConfigProperty(name = "llm-chat.max-history", defaultValue = "50")
            int maxHistory) {

        this.maxHistory = maxHistory;

        // Initialize LLM clients from comma-separated URLs
        List<VllmClient> clients = new ArrayList<>();
        if (!llmServers.isBlank()) {
            String[] urls = llmServers.split(",");
            for (String url : urls) {
                String trimmed = url.trim();
                if (!trimmed.isEmpty()) {
                    clients.add(new VllmClient(trimmed));
                    logger.info("LLM server configured: " + trimmed);
                }
            }
        }
        this.llmClients = List.copyOf(clients);
        if (llmClients.isEmpty()) {
            logger.warning("No LLM servers configured (llm-chat.servers is empty)");
        }

        this.modelSet = ModelSetBuilder.build(llmClients);
    }

    /**
     * Returns whether the given user has an active prompt.
     */
    public boolean isBusy(String userId) {
        return activeThreads.containsKey(userId);
    }

    /**
     * Returns the list of available models.
     */
    public List<ModelSet.ModelEntry> getAvailableModels() {
        return modelSet.getAvailableModels();
    }

    /**
     * Sends a prompt to the appropriate LLM server and streams response via callback.
     *
     * <p>This method blocks until the LLM completes.
     * Call from a virtual thread to avoid blocking the event loop.</p>
     *
     * @param userId the user identifier (from BasicAuth)
     * @param prompt the user prompt
     * @param model  the model name to use
     * @param sender callback for sending ChatEvent responses
     */
    /** Maximum number of retries when context length is exceeded. */
    private static final int MAX_CONTEXT_RETRIES = 3;

    /** Default max_tokens when max_model_len is unknown (conservative for small models). */
    private static final int DEFAULT_MAX_TOKENS = 1024;

    /** Upper bound for max_tokens to avoid excessive generation. */
    private static final int MAX_TOKENS_CAP = 8192;

    /** Estimated average tokens per message for send limit calculation. */
    private static final int ESTIMATED_TOKENS_PER_MESSAGE = 200;

    /** Estimated characters per token for budget calculation (conservative for mixed ja/en). */
    private static final int CHARS_PER_TOKEN = 3;

    /**
     * Calculates max_tokens from the model's max_model_len.
     * Uses half the context for output, capped at MAX_TOKENS_CAP.
     *
     * @param maxModelLen the model's max_model_len, or -1 if unknown
     * @return max_tokens value to use
     */
    static int calculateMaxTokens(int maxModelLen) {
        if (maxModelLen <= 0) {
            return DEFAULT_MAX_TOKENS;
        }
        return Math.min(maxModelLen / 2, MAX_TOKENS_CAP);
    }

    /**
     * Calculates the maximum number of messages to send to the model.
     * Reserves space for max_tokens output within the context window.
     *
     * @param maxModelLen the model's max_model_len, or -1 if unknown
     * @param maxTokens   the max_tokens value for output
     * @return maximum number of messages to include in the request
     */
    static int calculateSendLimit(int maxModelLen, int maxTokens) {
        if (maxModelLen <= 0) {
            return 6; // conservative default for unknown context size
        }
        int inputBudget = maxModelLen - maxTokens;
        int limit = inputBudget / ESTIMATED_TOKENS_PER_MESSAGE;
        return Math.max(2, Math.min(limit, 50)); // at least 2, at most 50
    }

    public void sendPrompt(String userId, String prompt, String model, boolean noThink,
                           List<String> imageDataUrls, Consumer<ChatEvent> sender) {
        if (isBusy(userId)) {
            sender.accept(ChatEvent.error("Already processing a prompt. Please wait or cancel."));
            return;
        }

        activeThreads.put(userId, Thread.currentThread());
        try {
            VllmClient client = findClientForModel(model);
            if (client == null) {
                sender.accept(ChatEvent.error("No LLM server configured for model: " + model));
                return;
            }

            // Send status: busy
            sender.accept(ChatEvent.status(model, null, true));

            // Calculate max_tokens and send limit from model's max_model_len
            int maxModelLen = client.getMaxModelLen(model);
            int maxTokens = calculateMaxTokens(maxModelLen);
            int sendLimit = calculateSendLimit(maxModelLen, maxTokens);

            // Build history with new user message
            List<ChatMessage> history = getHistory(userId);
            boolean applyNoThink = noThink && model.toLowerCase().startsWith("qwen3");
            List<String> images = imageDataUrls != null ? imageDataUrls : List.of();
            history.add(new ChatMessage.User(prompt, images));
            trimHistory(history);

            // Build a window of recent messages to send (not the full history)
            List<ChatMessage> sendHistory = recentWindow(history, sendLimit);

            // Pre-trim to fit within the model's context budget
            int charBudget = (maxModelLen > 0 ? maxModelLen - maxTokens : 2048) * CHARS_PER_TOKEN;
            preTrimToFit(sendHistory, charBudget);

            logger.info("User=" + userId + " model=" + model
                    + " noThink=" + noThink + " applyNoThink=" + applyNoThink
                    + " images=" + images.size()
                    + " history=" + history.size()
                    + " sendHistory=" + sendHistory.size()
                    + " maxModelLen=" + maxModelLen
                    + " maxTokens=" + maxTokens
                    + " charBudget=" + charBudget
                    + " prompt="
                    + prompt.substring(0, Math.min(prompt.length(), 80)));

            // Send (with safety-net retry in case estimation was off)
            String response = null;
            for (int attempt = 0; attempt < MAX_CONTEXT_RETRIES; attempt++) {
                try {
                    response = client.sendPrompt(model, sendHistory, applyNoThink, maxTokens, new VllmClient.StreamCallback() {
                        @Override
                        public void onDelta(String content) {
                            sender.accept(ChatEvent.delta(content));
                        }

                        @Override
                        public void onComplete(long durationMs) {
                            sender.accept(ChatEvent.result(null, 0.0, durationMs, model, false));
                        }

                        @Override
                        public void onError(String message) {
                            sender.accept(ChatEvent.error(message));
                        }
                    });
                    break; // success
                } catch (ContextLengthExceededException e) {
                    // Safety net: reduce budget by half and re-trim
                    charBudget = charBudget / 2;
                    preTrimToFit(sendHistory, charBudget);
                    logger.info("Context length exceeded for user=" + userId
                            + ", re-trimming with charBudget=" + charBudget
                            + " sendHistory=" + sendHistory.size()
                            + " (attempt " + (attempt + 1) + ")");
                    if (sendHistory.isEmpty()) {
                        sender.accept(ChatEvent.error("Context too long even after trimming"));
                        break;
                    }
                }
            }

            if (response != null) {
                // Success: add assistant response to history
                history.add(new ChatMessage.Assistant(response));
                trimHistory(history);
            } else {
                // Failure: rollback user message
                if (!history.isEmpty()
                        && history.get(history.size() - 1) instanceof ChatMessage.User) {
                    history.remove(history.size() - 1);
                }
            }
        } finally {
            activeThreads.remove(userId);
            sender.accept(ChatEvent.status(model, null, false));
        }
    }

    /**
     * Clears the conversation history for the given user.
     */
    public void clearHistory(String userId) {
        userHistories.remove(userId);
        logger.info("Cleared history for user: " + userId);
    }

    /**
     * Cancels the currently running request for the given user.
     */
    public void cancel(String userId) {
        Thread t = activeThreads.get(userId);
        if (t != null) {
            t.interrupt();
        }
    }

    /**
     * Returns the conversation history for the given user (creates if absent).
     */
    List<ChatMessage> getHistory(String userId) {
        return userHistories.computeIfAbsent(userId,
                k -> Collections.synchronizedList(new ArrayList<>()));
    }

    /**
     * Finds the LLM client that serves the given model.
     */
    VllmClient findClientForModel(String model) {
        // First try cached model list
        for (VllmClient client : llmClients) {
            if (client.servesModel(model)) {
                return client;
            }
        }
        // Cache miss: refresh model lists and retry
        for (VllmClient client : llmClients) {
            client.fetchModels();
            if (client.servesModel(model)) {
                return client;
            }
        }
        return null;
    }

    /**
     * Returns a mutable copy of the most recent messages from the history,
     * limited to {@code limit} messages and ensuring the window starts with a user message.
     *
     * @param history the full conversation history
     * @param limit   maximum number of messages to include
     * @return a new mutable list containing the recent window
     */
    static List<ChatMessage> recentWindow(List<ChatMessage> history, int limit) {
        int size = history.size();
        int start = Math.max(0, size - limit);
        // Ensure window starts with a user message
        while (start < size && !(history.get(start) instanceof ChatMessage.User)) {
            start++;
        }
        return new ArrayList<>(history.subList(start, size));
    }

    /**
     * Trims conversation history to the max limit.
     * Ensures history starts with a user message.
     */
    void trimHistory(List<ChatMessage> history) {
        while (history.size() > maxHistory) {
            history.remove(0);
        }
        while (!history.isEmpty() && !(history.get(0) instanceof ChatMessage.User)) {
            history.remove(0);
        }
    }

    /**
     * Pre-trims sendHistory to fit within the character budget.
     * Strategy:
     * 1. Remove old messages from the front (oldest first)
     * 2. If still over budget, truncate the last user message from the END
     *    (preserves the user's prompt and the beginning of fetched content)
     * 3. Repeat until within budget
     */
    void preTrimToFit(List<ChatMessage> history, int charBudget) {
        int totalChars = estimateTotalChars(history);
        if (totalChars <= charBudget) return;

        int originalSize = history.size();

        // Phase 1: remove old messages from the front (keep at least the last one)
        while (totalChars > charBudget && history.size() > 1) {
            ChatMessage removed = history.remove(0);
            totalChars = estimateTotalChars(history);
        }
        // Ensure starts with a user message
        while (history.size() > 1 && !(history.get(0) instanceof ChatMessage.User)) {
            history.remove(0);
            totalChars = estimateTotalChars(history);
        }

        // Phase 2: truncate the last user message content from the END
        if (totalChars > charBudget && !history.isEmpty()) {
            int lastUserIdx = -1;
            for (int i = history.size() - 1; i >= 0; i--) {
                if (history.get(i) instanceof ChatMessage.User) {
                    lastUserIdx = i;
                    break;
                }
            }
            if (lastUserIdx >= 0) {
                ChatMessage.User msg = (ChatMessage.User) history.get(lastUserIdx);
                String content = msg.content();
                // Calculate how many chars to keep
                int otherChars = totalChars - content.length();
                int allowedForContent = Math.max(100, charBudget - otherChars);
                if (allowedForContent < content.length()) {
                    String trimmed = content.substring(0, allowedForContent)
                            + "\n[... content truncated to fit context limit]";
                    history.set(lastUserIdx, new ChatMessage.User(trimmed, msg.imageDataUrls()));
                    logger.info("Pre-trim: truncated last user message from "
                            + content.length() + " to " + trimmed.length() + " chars");
                }
            }
        }

        logger.info("Pre-trim: " + originalSize + " -> " + history.size()
                + " messages, budget=" + charBudget);
    }

    /**
     * Estimates total character count across all messages in the history.
     */
    static int estimateTotalChars(List<ChatMessage> history) {
        int total = 0;
        for (ChatMessage msg : history) {
            if (msg instanceof ChatMessage.User u) {
                total += u.content().length();
            } else if (msg instanceof ChatMessage.Assistant a) {
                total += a.content().length();
            }
        }
        return total;
    }
}
