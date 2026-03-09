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

package com.github.oogasawa.llmchat.rest;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * SSE event DTO for chat communication.
 *
 * <p>Server to Client event types:</p>
 * <ul>
 *   <li>{@code delta} - Partial text content from Claude</li>
 *   <li>{@code result} - Final result with session ID and cost info</li>
 *   <li>{@code error} - Error message</li>
 *   <li>{@code info} - Informational message (e.g., model changed)</li>
 *   <li>{@code status} - Status update (model, session, busy state)</li>
 *   <li>{@code prompt} - Interactive prompt from Claude (tool permission, yes/no, etc.)</li>
 *   <li>{@code heartbeat} - Keep-alive for SSE connection</li>
 *   <li>{@code log} - Server log entry (level, logger, message, timestamp)</li>
 * </ul>
 *
 * @author oogasawa
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ChatEvent(
    String type,
    String content,
    String sessionId,
    Double costUsd,
    Long durationMs,
    String model,
    Boolean busy,
    String promptId,
    String promptType,
    List<String> options,
    String logLevel,
    String loggerName,
    Long timestamp
) {

    public static ChatEvent delta(String content) {
        return new ChatEvent("delta", content, null, null, null, null, null, null, null, null, null, null, null);
    }

    public static ChatEvent result(String sessionId, double costUsd, long durationMs) {
        return new ChatEvent("result", null, sessionId, costUsd, durationMs, null, null, null, null, null, null, null, null);
    }

    public static ChatEvent result(String sessionId, double costUsd, long durationMs,
                                    String model, boolean busy) {
        return new ChatEvent("result", null, sessionId, costUsd, durationMs, model, busy, null, null, null, null, null, null);
    }

    public static ChatEvent error(String content) {
        return new ChatEvent("error", content, null, null, null, null, null, null, null, null, null, null, null);
    }

    public static ChatEvent info(String content) {
        return new ChatEvent("info", content, null, null, null, null, null, null, null, null, null, null, null);
    }

    public static ChatEvent status(String model, String sessionId, boolean busy) {
        return new ChatEvent("status", null, sessionId, null, null, model, busy, null, null, null, null, null, null);
    }

    /**
     * Creates a thinking/activity indicator event.
     *
     * @param content optional activity description (e.g., "Using WebSearch...")
     */
    public static ChatEvent thinking(String content) {
        return new ChatEvent("thinking", content, null, null, null, null, null, null, null, null, null, null, null);
    }

    /**
     * Creates a heartbeat event to keep the SSE connection alive.
     */
    public static ChatEvent heartbeat() {
        return new ChatEvent("heartbeat", null, null, null, null, null, null, null, null, null, null, null, null);
    }

    /**
     * Creates a prompt event for interactive Claude prompts (tool permission, yes/no, etc.).
     *
     * @param promptId unique identifier for this prompt
     * @param content the prompt text to display
     * @param promptType the type of prompt (e.g., "permission", "yesno", "text")
     * @param options list of option labels for buttons (null for free-text input)
     */
    public static ChatEvent prompt(String promptId, String content,
                                    String promptType, List<String> options) {
        return new ChatEvent("prompt", content, null, null, null, null, null,
                             promptId, promptType, options, null, null, null);
    }

    /**
     * Creates a server log event for real-time log streaming to the web UI.
     *
     * @param level   log level (e.g., "INFO", "WARNING", "FINE")
     * @param logger  logger name (typically the class name)
     * @param message log message
     * @param ts      epoch millis
     */
    public static ChatEvent log(String level, String logger, String message, long ts) {
        return new ChatEvent("log", message, null, null, null, null, null,
                             null, null, null, level, logger, ts);
    }
}
