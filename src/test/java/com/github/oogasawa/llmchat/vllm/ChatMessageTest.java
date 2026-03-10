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

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ChatMessage types and buildRequestBody integration.
 *
 * @author oogasawa
 */
class ChatMessageTest {

    @Test
    void userMessage_role() {
        ChatMessage msg = new ChatMessage.User("hello");
        assertEquals("user", msg.role());
    }

    @Test
    void assistantMessage_role() {
        ChatMessage msg = new ChatMessage.Assistant("hi");
        assertEquals("assistant", msg.role());
    }

    @Test
    void sealedInterface_allTypesExhaustive() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("q"),
                new ChatMessage.Assistant("a")
        );

        for (ChatMessage msg : messages) {
            String role = switch (msg) {
                case ChatMessage.User u -> u.role();
                case ChatMessage.Assistant a -> a.role();
            };
            assertNotNull(role);
        }
    }

    // --- buildRequestBody integration tests ---

    @Test
    void buildRequestBody_userAndAssistant() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("hello"),
                new ChatMessage.Assistant("hi there")
        );
        String body = VllmClient.buildRequestBody("test-model", messages, false, 2048);

        assertTrue(body.contains("\"model\":\"test-model\""));
        assertTrue(body.contains("\"stream\":true"));
        assertTrue(body.contains("\"role\":\"user\""));
        assertTrue(body.contains("\"content\":\"hello\""));
        assertTrue(body.contains("\"role\":\"assistant\""));
        assertTrue(body.contains("\"content\":\"hi there\""));
    }

    @Test
    void buildRequestBody_emptyMessages() {
        List<ChatMessage> messages = List.of();
        String body = VllmClient.buildRequestBody("model", messages, false, 2048);
        assertTrue(body.contains("\"messages\":[]"));
    }

    @Test
    void buildRequestBody_specialCharactersInContent() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("line1\nline2"),
                new ChatMessage.Assistant("say \"hello\"")
        );
        String body = VllmClient.buildRequestBody("model", messages, false, 2048);

        assertTrue(body.contains("\\n"));
        assertTrue(body.contains("\\\"hello\\\""));
    }
}
