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

            // Build history with new user message
            List<ChatMessage> history = getHistory(userId);
            boolean applyNoThink = noThink && model.toLowerCase().startsWith("qwen3");
            List<String> images = imageDataUrls != null ? imageDataUrls : List.of();
            history.add(new ChatMessage.User(prompt, images));
            trimHistory(history);

            logger.info("User=" + userId + " model=" + model
                    + " noThink=" + noThink + " applyNoThink=" + applyNoThink
                    + " images=" + images.size()
                    + " history=" + history.size() + " prompt="
                    + prompt.substring(0, Math.min(prompt.length(), 80)));

            // Retry loop: trim history aggressively on context length overflow
            String response = null;
            for (int attempt = 0; attempt < MAX_CONTEXT_RETRIES; attempt++) {
                try {
                    response = client.sendPrompt(model, history, applyNoThink, new VllmClient.StreamCallback() {
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
                    break; // success or non-context error
                } catch (ContextLengthExceededException e) {
                    int removed = trimHistoryAggressive(history);
                    if (removed == 0) {
                        sender.accept(ChatEvent.error("Context too long even after trimming all history"));
                        break;
                    }
                    sender.accept(ChatEvent.delta("[Context too long, trimming history... (attempt "
                            + (attempt + 1) + ")]\n"));
                    logger.info("Context length exceeded for user=" + userId
                            + ", trimmed " + removed + " messages, " + history.size()
                            + " remaining (attempt " + (attempt + 1) + ")");
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
     * Aggressively trims history by removing half the messages.
     * Used when context length is exceeded despite normal trimming.
     * Ensures history starts with a user message after trimming.
     *
     * @return the number of messages removed
     */
    int trimHistoryAggressive(List<ChatMessage> history) {
        if (history.size() <= 2) {
            return 0;
        }
        int toRemove = history.size() / 2;
        int removed = 0;
        for (int i = 0; i < toRemove; i++) {
            history.remove(0);
            removed++;
        }
        // Ensure history starts with a user message
        while (!history.isEmpty() && !(history.get(0) instanceof ChatMessage.User)) {
            history.remove(0);
            removed++;
        }
        logger.info("Aggressive trim: removed " + removed + " messages, "
                + history.size() + " remaining");
        return removed;
    }
}
