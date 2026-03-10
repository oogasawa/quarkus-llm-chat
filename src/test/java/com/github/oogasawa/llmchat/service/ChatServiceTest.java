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
import com.github.oogasawa.llmchat.rest.ChatResource;
import com.github.oogasawa.llmchat.vllm.ChatMessage;
import com.github.oogasawa.llmchat.vllm.VllmClient;
import org.junit.jupiter.api.Test;

import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

class ChatServiceTest {

    // --- isBusy ---

    @Test
    void isBusy_initiallyFalse() {
        ChatService service = createService("");
        assertFalse(service.isBusy("user1"));
    }

    // --- getHistory ---

    @Test
    void getHistory_createsNewListPerUser() {
        ChatService service = createService("");
        var h1 = service.getHistory("user1");
        var h2 = service.getHistory("user2");
        assertNotSame(h1, h2);
    }

    @Test
    void getHistory_returnsSameListForSameUser() {
        ChatService service = createService("");
        var h1 = service.getHistory("user1");
        var h2 = service.getHistory("user1");
        assertSame(h1, h2);
    }

    // --- clearHistory ---

    @Test
    void clearHistory_removesUserHistory() {
        ChatService service = createService("");
        service.getHistory("user1").add(new ChatMessage.User("hello"));
        assertEquals(1, service.getHistory("user1").size());

        service.clearHistory("user1");
        // After clear, getHistory creates a new empty list
        assertEquals(0, service.getHistory("user1").size());
    }

    // --- trimHistory ---

    @Test
    void trimHistory_removesOldestAndOrphanedAssistant() {
        ChatService service = new ChatService("", 4);
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("msg1"));
        history.add(new ChatMessage.Assistant("resp1"));
        history.add(new ChatMessage.User("msg2"));
        history.add(new ChatMessage.Assistant("resp2"));
        history.add(new ChatMessage.User("msg3"));
        service.trimHistory(history);

        // maxHistory=4: "msg1" removed (over limit), then "resp1" removed (orphaned)
        assertEquals(3, history.size());
        assertInstanceOf(ChatMessage.User.class, history.get(0));
        assertEquals("msg2", ((ChatMessage.User) history.get(0)).content());
    }

    @Test
    void trimHistory_alwaysStartsWithUser() {
        ChatService service = new ChatService("", 4);
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("q1"));
        history.add(new ChatMessage.Assistant("a1"));
        history.add(new ChatMessage.User("q2"));
        history.add(new ChatMessage.Assistant("a2"));
        history.add(new ChatMessage.User("q3"));
        history.add(new ChatMessage.Assistant("a3"));
        service.trimHistory(history);

        assertInstanceOf(ChatMessage.User.class, history.get(0));
    }

    // --- cancel ---

    @Test
    void cancel_whenNotBusy_doesNotThrow() {
        ChatService service = createService("");
        assertDoesNotThrow(() -> service.cancel("user1"));
    }

    @Test
    void cancel_interruptsActiveThread() throws Exception {
        Logger vllmLogger = Logger.getLogger(VllmClient.class.getName());
        Level originalLevel = vllmLogger.getLevel();
        vllmLogger.setLevel(Level.OFF);
        try {
            try (ServerSocket ss = new ServerSocket(0)) {
                int port = ss.getLocalPort();
                ChatService service = new ChatService(
                        "http://localhost:" + port, 10);

                var events = new CopyOnWriteArrayList<ChatEvent>();
                var started = new CountDownLatch(1);

                Thread worker = Thread.startVirtualThread(() -> {
                    started.countDown();
                    service.sendPrompt("user1", "hello", "fake-model", false, null, events::add);
                });

                assertTrue(started.await(5, TimeUnit.SECONDS));
                Thread.sleep(200);

                service.cancel("user1");
                worker.join(5000);

                assertFalse(worker.isAlive());
            }
        } finally {
            vllmLogger.setLevel(originalLevel);
        }
    }

    @Test
    void sendPrompt_setsAndClearsBusyFlag() throws Exception {
        Logger vllmLogger = Logger.getLogger(VllmClient.class.getName());
        Level originalLevel = vllmLogger.getLevel();
        vllmLogger.setLevel(Level.OFF);
        try {
            ChatService service = new ChatService("http://localhost:1", 10);

            var events = new CopyOnWriteArrayList<ChatEvent>();
            var done = new CountDownLatch(1);

            Thread.startVirtualThread(() -> {
                service.sendPrompt("user1", "hello", "some-model", false, null, events::add);
                done.countDown();
            });

            assertTrue(done.await(10, TimeUnit.SECONDS));
            assertFalse(service.isBusy("user1"));
        } finally {
            vllmLogger.setLevel(originalLevel);
        }
    }

    // --- extractHost ---

    @Test
    void extractHost_withPort() {
        assertEquals("192.168.5.15:8000", LocalLlmModelSet.extractHost("http://192.168.5.15:8000"));
    }

    @Test
    void extractHost_withoutPort() {
        assertEquals("example.com", LocalLlmModelSet.extractHost("http://example.com"));
    }

    @Test
    void extractHost_httpsWithPort() {
        assertEquals("localhost:11434", LocalLlmModelSet.extractHost("https://localhost:11434"));
    }

    @Test
    void extractHost_invalidUrl_returnsOriginal() {
        assertEquals("not-a-url", LocalLlmModelSet.extractHost("not-a-url"));
    }

    // --- BasicAuth parsing ---

    @Test
    void parseBasicAuth_valid() {
        // "user:pass" -> base64 "dXNlcjpwYXNz"
        assertEquals("user", ChatResource.parseBasicAuth("Basic dXNlcjpwYXNz"));
    }

    @Test
    void parseBasicAuth_null() {
        assertNull(ChatResource.parseBasicAuth(null));
    }

    @Test
    void parseBasicAuth_notBasic() {
        assertNull(ChatResource.parseBasicAuth("Bearer token123"));
    }

    @Test
    void parseBasicAuth_invalidBase64() {
        assertNull(ChatResource.parseBasicAuth("Basic !!!invalid!!!"));
    }

    @Test
    void parseBasicAuth_noColon() {
        // "useronly" -> base64 "dXNlcm9ubHk="
        assertNull(ChatResource.parseBasicAuth("Basic dXNlcm9ubHk="));
    }

    @Test
    void parseBasicAuth_emptyUsername() {
        // ":pass" -> base64 "OnBhc3M="
        assertNull(ChatResource.parseBasicAuth("Basic OnBhc3M="));
    }

    // --- helpers ---

    private static ChatService createService(String servers) {
        return new ChatService(servers, 50);
    }
}
