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

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ChatEventTest {

    @Test
    void delta_createsCorrectEvent() {
        ChatEvent event = ChatEvent.delta("Hello");

        assertEquals("delta", event.type());
        assertEquals("Hello", event.content());
        assertNull(event.sessionId());
        assertNull(event.costUsd());
        assertNull(event.durationMs());
        assertNull(event.model());
        assertNull(event.busy());
        assertNull(event.promptId());
        assertNull(event.promptType());
        assertNull(event.options());
    }

    @Test
    void result_createsCorrectEvent() {
        ChatEvent event = ChatEvent.result("sess-abc", 0.005, 1234);

        assertEquals("result", event.type());
        assertNull(event.content());
        assertEquals("sess-abc", event.sessionId());
        assertEquals(0.005, event.costUsd(), 0.0001);
        assertEquals(1234L, event.durationMs());
        assertNull(event.model());
        assertNull(event.busy());
    }

    @Test
    void error_createsCorrectEvent() {
        ChatEvent event = ChatEvent.error("Something failed");

        assertEquals("error", event.type());
        assertEquals("Something failed", event.content());
        assertNull(event.sessionId());
    }

    @Test
    void info_createsCorrectEvent() {
        ChatEvent event = ChatEvent.info("Model changed to: opus");

        assertEquals("info", event.type());
        assertEquals("Model changed to: opus", event.content());
        assertNull(event.sessionId());
    }

    @Test
    void status_createsCorrectEvent() {
        ChatEvent event = ChatEvent.status("sonnet", "sess-abc", false);

        assertEquals("status", event.type());
        assertNull(event.content());
        assertEquals("sonnet", event.model());
        assertEquals("sess-abc", event.sessionId());
        assertFalse(event.busy());
    }

    @Test
    void status_busyTrue() {
        ChatEvent event = ChatEvent.status("opus", null, true);

        assertEquals("status", event.type());
        assertEquals("opus", event.model());
        assertNull(event.sessionId());
        assertTrue(event.busy());
    }

    @Test
    void heartbeat_createsCorrectEvent() {
        ChatEvent event = ChatEvent.heartbeat();

        assertEquals("heartbeat", event.type());
        assertNull(event.content());
        assertNull(event.promptId());
    }

    @Test
    void prompt_createsCorrectEvent() {
        List<String> options = List.of("Allow", "Deny");
        ChatEvent event = ChatEvent.prompt("prompt-123", "Allow tool execution?", "permission", options);

        assertEquals("prompt", event.type());
        assertEquals("Allow tool execution?", event.content());
        assertEquals("prompt-123", event.promptId());
        assertEquals("permission", event.promptType());
        assertEquals(2, event.options().size());
        assertEquals("Allow", event.options().get(0));
        assertEquals("Deny", event.options().get(1));
        assertNull(event.sessionId());
        assertNull(event.model());
        assertNull(event.busy());
    }

    @Test
    void prompt_withNullOptions() {
        ChatEvent event = ChatEvent.prompt("prompt-456", "Enter your name:", "text", null);

        assertEquals("prompt", event.type());
        assertEquals("Enter your name:", event.content());
        assertEquals("prompt-456", event.promptId());
        assertEquals("text", event.promptType());
        assertNull(event.options());
    }
}
