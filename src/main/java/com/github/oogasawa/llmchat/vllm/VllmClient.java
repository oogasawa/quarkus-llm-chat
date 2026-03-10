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

package com.github.oogasawa.llmchat.vllm;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

/**
 * Stateless HTTP client for vLLM OpenAI-compatible API with streaming support.
 *
 * <p>POJO (no CDI annotations). Conversation history is managed externally
 * by the caller (ChatService), not by this client.</p>
 *
 * @author oogasawa
 */
public class VllmClient {

    private static final Logger logger = Logger.getLogger(VllmClient.class.getName());

    private final String baseUrl;
    private final HttpClient httpClient;
    private volatile List<String> cachedModels = List.of();
    private volatile Map<String, Integer> cachedMaxModelLen = Map.of();

    /**
     * Callback interface for streaming responses.
     */
    public interface StreamCallback {
        void onDelta(String content);
        void onComplete(long durationMs);
        void onError(String message);
    }

    /**
     * Creates a new VllmClient.
     *
     * @param baseUrl vLLM server base URL (e.g., "http://192.168.5.15:8000")
     */
    public VllmClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    /**
     * Returns the base URL of the vLLM server.
     */
    public String getBaseUrl() {
        return baseUrl;
    }

    /**
     * Returns true if this server serves the given model (based on cached model list).
     *
     * @param model the model name to check
     * @return true if the model is in the cached model list
     */
    public boolean servesModel(String model) {
        return cachedModels.contains(model);
    }

    /**
     * Returns the cached model list (last result from {@link #fetchModels()}).
     *
     * @return unmodifiable view of cached model IDs
     */
    public List<String> getCachedModels() {
        return cachedModels;
    }

    /**
     * Fetches the list of available model IDs from the vLLM server.
     *
     * <p>Calls {@code GET /v1/models} and extracts model IDs from the response.
     * Returns an empty list if the server is unreachable or returns an error.</p>
     *
     * @return list of model ID strings
     */
    public List<String> fetchModels() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/models"))
                    .header("Accept", "application/json")
                    .GET()
                    .timeout(java.time.Duration.ofSeconds(5))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                logger.warning("vLLM /v1/models returned HTTP " + response.statusCode());
                return List.of();
            }

            String body = response.body();
            List<String> ids = parseModelIds(body);
            this.cachedModels = ids;
            this.cachedMaxModelLen = parseMaxModelLens(body);
            return ids;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return List.of();
        } catch (Exception e) {
            logger.fine(() -> "vLLM /v1/models unavailable: " + e.getMessage());
            return List.of();
        }
    }

    /**
     * Parses model IDs from the /v1/models JSON response.
     *
     * <p>Expected format: {@code {"data":[{"id":"model-name"}, ...]}}</p>
     *
     * @param json the response body
     * @return list of model IDs
     */
    static List<String> parseModelIds(String json) {
        List<String> ids = new ArrayList<>();
        int dataIdx = json.indexOf("\"data\"");
        if (dataIdx < 0) return ids;
        int arrStart = json.indexOf('[', dataIdx);
        if (arrStart < 0) return ids;

        String marker = "\"id\":\"";
        int depth = 0;
        int i = arrStart;

        while (i < json.length()) {
            char c = json.charAt(i);
            if (c == '[' || c == '{') {
                depth++;
                i++;
            } else if (c == ']' || c == '}') {
                depth--;
                if (depth <= 0) break;
                i++;
            } else if (c == '"') {
                if (depth == 2 && json.startsWith(marker, i)) {
                    int start = i + marker.length();
                    String value = unescapeJsonString(json, start);
                    if (value != null && !value.isEmpty()) {
                        ids.add(value);
                    }
                    i = start;
                    while (i < json.length() && json.charAt(i) != '"') {
                        if (json.charAt(i) == '\\') i++;
                        i++;
                    }
                    i++;
                } else {
                    i++;
                    while (i < json.length() && json.charAt(i) != '"') {
                        if (json.charAt(i) == '\\') i++;
                        i++;
                    }
                    i++;
                }
            } else {
                i++;
            }
        }
        return ids;
    }

    /**
     * Returns the cached max_model_len for the given model, or -1 if unknown.
     *
     * @param model the model name
     * @return max_model_len, or -1 if not cached
     */
    public int getMaxModelLen(String model) {
        return cachedMaxModelLen.getOrDefault(model, -1);
    }

    /**
     * Parses model IDs and their max_model_len from the /v1/models JSON response.
     *
     * <p>Scans for pairs of {@code "id":"..."} and {@code "max_model_len":NNN}
     * within each object in the data array.</p>
     *
     * @param json the response body
     * @return map of model ID to max_model_len
     */
    static Map<String, Integer> parseMaxModelLens(String json) {
        Map<String, Integer> result = new HashMap<>();
        int dataIdx = json.indexOf("\"data\"");
        if (dataIdx < 0) return result;
        int arrStart = json.indexOf('[', dataIdx);
        if (arrStart < 0) return result;

        // Find each object in the data array
        String idMarker = "\"id\":\"";
        String lenMarker = "\"max_model_len\":";
        int i = arrStart;
        while (i < json.length()) {
            int objStart = json.indexOf('{', i);
            if (objStart < 0) break;

            // Find matching closing brace (handle nested objects)
            int depth = 0;
            int objEnd = objStart;
            for (int j = objStart; j < json.length(); j++) {
                char c = json.charAt(j);
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) { objEnd = j; break; }
                }
            }
            if (objEnd <= objStart) break;

            String obj = json.substring(objStart, objEnd + 1);

            // Extract id
            int idIdx = obj.indexOf(idMarker);
            String id = null;
            if (idIdx >= 0) {
                int start = idIdx + idMarker.length();
                id = unescapeJsonString(obj, start);
            }

            // Extract max_model_len
            int lenIdx = obj.indexOf(lenMarker);
            if (id != null && lenIdx >= 0) {
                int numStart = lenIdx + lenMarker.length();
                int numEnd = numStart;
                while (numEnd < obj.length() && (Character.isDigit(obj.charAt(numEnd)))) {
                    numEnd++;
                }
                if (numEnd > numStart) {
                    try {
                        int maxLen = Integer.parseInt(obj.substring(numStart, numEnd));
                        result.put(id, maxLen);
                    } catch (NumberFormatException e) {
                        // ignore
                    }
                }
            }

            i = objEnd + 1;
        }
        return result;
    }

    /**
     * Sends a prompt to the vLLM server and streams the response.
     *
     * <p>The caller provides the full conversation history (including the new user message).
     * This method sends the request and streams the response.
     * History management (trimming, rollback) is the caller's responsibility.</p>
     *
     * @param model     the model name
     * @param history   the conversation history (will NOT be modified)
     * @param noThink   whether to disable thinking mode
     * @param maxTokens maximum tokens to generate (0 or negative = omit from request)
     * @param callback  callback for streaming events
     * @return the full assistant response text, or null on failure
     */
    public String sendPrompt(String model, List<ChatMessage> history, boolean noThink,
                             int maxTokens, StreamCallback callback) {
        long startTime = System.currentTimeMillis();

        try {
            String requestBody = buildRequestBody(model, history, noThink, maxTokens);
            HttpResponse<Stream<String>> response = null;

            response = sendWithRetry(requestBody);

            if (response.statusCode() == 400) {
                StringBuilder errorBody = new StringBuilder();
                response.body().forEach(errorBody::append);
                String errorStr = errorBody.toString();
                if (isContextLengthError(errorStr)) {
                    throw new ContextLengthExceededException(errorStr);
                }
                callback.onError("vLLM returned HTTP 400: " + errorStr);
                return null;
            }
            if (response.statusCode() != 200) {
                StringBuilder errorBody = new StringBuilder();
                response.body().forEach(errorBody::append);
                String error = "vLLM returned HTTP " + response.statusCode() + ": " + errorBody;
                logger.warning(error);
                callback.onError(error);
                return null;
            }

            StringBuilder fullResponse = new StringBuilder();
            boolean interrupted = false;

            var iterator = response.body().iterator();
            while (iterator.hasNext()) {
                if (Thread.currentThread().isInterrupted()) {
                    interrupted = true;
                    break;
                }
                String line = iterator.next();
                String content = parseSseLine(line);
                if (content != null) {
                    fullResponse.append(content);
                    callback.onDelta(content);
                }
            }

            if (interrupted) {
                callback.onError("Request cancelled");
                return null;
            }

            long durationMs = System.currentTimeMillis() - startTime;
            callback.onComplete(durationMs);

            return fullResponse.length() > 0 ? fullResponse.toString() : null;

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            callback.onError("Request cancelled");
            return null;
        } catch (java.io.UncheckedIOException e) {
            // Stream read interrupted (cancel)
            if (e.getCause() != null && e.getCause().getCause() instanceof InterruptedException) {
                Thread.currentThread().interrupt();
                callback.onError("Request cancelled");
                return null;
            }
            logger.log(Level.WARNING, "vLLM request failed", e);
            String msg = e.getMessage();
            if (isContextLengthError(msg)) {
                throw new ContextLengthExceededException(msg);
            }
            callback.onError("vLLM error (" + baseUrl + "): " + msg);
            return null;
        } catch (ContextLengthExceededException e) {
            throw e; // propagate to ChatService retry loop
        } catch (Exception e) {
            logger.log(Level.WARNING, "vLLM request failed", e);
            String detail = e.getMessage();
            if (detail == null) {
                detail = e.getClass().getSimpleName();
                if (e.getCause() != null) {
                    detail += ": " + e.getCause().getMessage();
                }
            }
            // Check if the error message contains context length info
            if (isContextLengthError(detail)) {
                throw new ContextLengthExceededException(detail);
            }
            callback.onError("vLLM error (" + baseUrl + "): " + detail);
            return null;
        }
    }

    /**
     * Sends an HTTP request with retry on connection reset.
     */
    private HttpResponse<Stream<String>> sendWithRetry(String requestBody)
            throws IOException, InterruptedException {

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        try {
            return httpClient.send(request, HttpResponse.BodyHandlers.ofLines());
        } catch (IOException e) {
            String msg = e.getMessage();
            if (msg != null && (msg.contains("Connection reset")
                    || msg.contains("received no bytes"))) {
                logger.info("Connection reset, retrying with fresh HttpClient");
                HttpClient freshClient = HttpClient.newBuilder()
                        .version(HttpClient.Version.HTTP_1_1)
                        .build();
                return freshClient.send(request, HttpResponse.BodyHandlers.ofLines());
            }
            throw e;
        }
    }

    /**
     * Builds the JSON request body for the OpenAI chat completions API.
     *
     * @param model     the model name
     * @param messages  the conversation history
     * @param noThink   whether to disable thinking mode
     * @param maxTokens maximum tokens to generate (0 or negative = omit)
     * @return JSON string
     */
    static String buildRequestBody(String model, List<ChatMessage> messages, boolean noThink, int maxTokens) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"model\":\"").append(escapeJson(model)).append("\",");
        sb.append("\"stream\":true,");
        if (maxTokens > 0) {
            sb.append("\"max_tokens\":").append(maxTokens).append(",");
        }
        if (noThink) {
            sb.append("\"chat_template_kwargs\":{\"enable_thinking\":false},");
        }
        sb.append("\"messages\":[");

        for (int i = 0; i < messages.size(); i++) {
            if (i > 0) sb.append(",");
            ChatMessage msg = messages.get(i);
            sb.append("{\"role\":\"").append(escapeJson(msg.role())).append("\",");

            if (msg instanceof ChatMessage.User u && u.hasImages()) {
                // Multimodal content: array of image_url + text parts
                sb.append("\"content\":[");
                for (int j = 0; j < u.imageDataUrls().size(); j++) {
                    if (j > 0) sb.append(",");
                    sb.append("{\"type\":\"image_url\",\"image_url\":{\"url\":\"");
                    sb.append(escapeJson(u.imageDataUrls().get(j)));
                    sb.append("\"}}");
                }
                sb.append(",{\"type\":\"text\",\"text\":\"");
                sb.append(escapeJson(u.content()));
                sb.append("\"}]");
            } else {
                String content = switch (msg) {
                    case ChatMessage.User u -> u.content();
                    case ChatMessage.Assistant a -> a.content();
                };
                sb.append("\"content\":\"").append(escapeJson(content)).append("\"");
            }
            sb.append("}");
        }

        sb.append("]}");
        return sb.toString();
    }

    /**
     * Parses a single SSE line from the vLLM streaming response.
     *
     * @param line the SSE line
     * @return extracted content text, or null if not a content line
     */
    static String parseSseLine(String line) {
        if (line == null || !line.startsWith("data: ")) {
            return null;
        }

        String data = line.substring(6).trim();

        if (data.equals("[DONE]")) {
            return null;
        }

        return extractDeltaContent(data);
    }

    /**
     * Extracts the delta content from a streaming chunk JSON.
     *
     * @param json the JSON string from the SSE data field
     * @return the content string, or null if not found
     */
    static String extractDeltaContent(String json) {
        String marker = "\"content\":\"";
        int deltaIdx = json.indexOf("\"delta\"");
        if (deltaIdx < 0) {
            return null;
        }

        int contentIdx = json.indexOf(marker, deltaIdx);
        if (contentIdx < 0) {
            return null;
        }

        int startIdx = contentIdx + marker.length();
        return unescapeJsonString(json, startIdx);
    }

    /**
     * Checks if an HTTP 400 error body indicates a context length overflow.
     */
    static boolean isContextLengthError(String errorBody) {
        return errorBody != null
                && (errorBody.contains("input_tokens")
                    || errorBody.contains("context length")
                    || errorBody.contains("maximum input length"));
    }

    /**
     * Reads a JSON string value starting at the given index, handling escape sequences.
     *
     * @param json     the JSON string
     * @param startIdx index of the first character after the opening quote
     * @return the unescaped string value, or null on parse error
     */
    static String unescapeJsonString(String json, int startIdx) {
        StringBuilder result = new StringBuilder();
        int i = startIdx;

        while (i < json.length()) {
            char c = json.charAt(i);
            if (c == '"') {
                return result.toString();
            } else if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case '/' -> result.append('/');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    case 'b' -> result.append('\b');
                    case 'f' -> result.append('\f');
                    case 'u' -> {
                        if (i + 5 < json.length()) {
                            String hex = json.substring(i + 2, i + 6);
                            try {
                                result.append((char) Integer.parseInt(hex, 16));
                            } catch (NumberFormatException e) {
                                result.append("\\u").append(hex);
                            }
                            i += 4;
                        }
                    }
                    default -> {
                        result.append('\\');
                        result.append(next);
                    }
                }
                i += 2;
            } else {
                result.append(c);
                i++;
            }
        }
        return result.toString();
    }

    /**
     * Escapes a string for inclusion in a JSON string literal.
     *
     * @param str the string to escape
     * @return the escaped string
     */
    static String escapeJson(String str) {
        if (str == null) return "";
        StringBuilder sb = new StringBuilder(str.length());
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }
}
