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

import java.util.List;

/**
 * Sealed interface representing a single message in the conversation history.
 *
 * <p>Supports user and assistant text messages for simple chat interaction.</p>
 *
 * @author oogasawa
 */
public sealed interface ChatMessage {

    /**
     * Returns the role string for this message ("user" or "assistant").
     */
    String role();

    /**
     * A user message with text content and optional images.
     *
     * @param content the text content
     * @param imageDataUrls base64 data URLs (e.g. "data:image/png;base64,..."), may be empty
     */
    record User(String content, List<String> imageDataUrls) implements ChatMessage {
        public User(String content) {
            this(content, List.of());
        }
        @Override
        public String role() { return "user"; }
        public boolean hasImages() { return imageDataUrls != null && !imageDataUrls.isEmpty(); }
    }

    /**
     * An assistant message with text content.
     */
    record Assistant(String content) implements ChatMessage {
        @Override
        public String role() { return "assistant"; }
    }
}
