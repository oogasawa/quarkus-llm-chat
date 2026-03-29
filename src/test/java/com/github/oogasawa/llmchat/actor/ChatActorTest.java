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

package com.github.oogasawa.llmchat.actor;

import com.github.oogasawa.llmchat.rest.ChatEvent;
import com.github.oogasawa.llmchat.rest.ChatResource;
import com.github.oogasawa.llmchat.vllm.ChatMessage;
import com.github.oogasawa.llmchat.vllm.VllmClient;
import com.scivicslab.pojoactor.core.ActorRef;
import com.scivicslab.pojoactor.core.ActorSystem;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

class ChatActorTest {

    // --- isBusy ---

    @Test
    void isBusy_initiallyFalse() {
        ChatActor actor = createActor("");
        assertFalse(actor.isBusy("user1"));
    }

    // --- getHistory ---

    @Test
    void getHistory_createsNewListPerUser() {
        ChatActor actor = createActor("");
        var h1 = actor.getHistory("user1");
        var h2 = actor.getHistory("user2");
        assertNotSame(h1, h2);
    }

    @Test
    void getHistory_returnsSameListForSameUser() {
        ChatActor actor = createActor("");
        var h1 = actor.getHistory("user1");
        var h2 = actor.getHistory("user1");
        assertSame(h1, h2);
    }

    // --- clearHistory ---

    @Test
    void clearHistory_removesUserHistory() {
        ChatActor actor = createActor("");
        actor.getHistory("user1").add(new ChatMessage.User("hello"));
        assertEquals(1, actor.getHistory("user1").size());

        actor.clearHistory("user1");
        // After clear, getHistory creates a new empty list
        assertEquals(0, actor.getHistory("user1").size());
    }

    // --- trimHistory ---

    @Test
    void trimHistory_removesOldestAndOrphanedAssistant() {
        ChatActor actor = new ChatActor("", 4);
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("msg1"));
        history.add(new ChatMessage.Assistant("resp1"));
        history.add(new ChatMessage.User("msg2"));
        history.add(new ChatMessage.Assistant("resp2"));
        history.add(new ChatMessage.User("msg3"));
        actor.trimHistory(history);

        // maxHistory=4: "msg1" removed (over limit), then "resp1" removed (orphaned)
        assertEquals(3, history.size());
        assertInstanceOf(ChatMessage.User.class, history.get(0));
        assertEquals("msg2", ((ChatMessage.User) history.get(0)).content());
    }

    @Test
    void trimHistory_alwaysStartsWithUser() {
        ChatActor actor = new ChatActor("", 4);
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("q1"));
        history.add(new ChatMessage.Assistant("a1"));
        history.add(new ChatMessage.User("q2"));
        history.add(new ChatMessage.Assistant("a2"));
        history.add(new ChatMessage.User("q3"));
        history.add(new ChatMessage.Assistant("a3"));
        actor.trimHistory(history);

        assertInstanceOf(ChatMessage.User.class, history.get(0));
    }

    // --- cancel ---

    @Test
    void cancel_whenNotBusy_doesNotThrow() {
        ChatActor actor = createActor("");
        assertDoesNotThrow(() -> actor.cancel("user1"));
    }

    @Test
    void startPrompt_withActorSystem_setsAndClearsBusy() throws Exception {
        Logger vllmLogger = Logger.getLogger(VllmClient.class.getName());
        Level originalLevel = vllmLogger.getLevel();
        vllmLogger.setLevel(Level.OFF);
        try {
            ActorSystem system = new ActorSystem("test");
            ChatActor chatActor = new ChatActor("http://localhost:1", 10);
            ActorRef<ChatActor> ref = system.actorOf("chat", chatActor);

            var events = new CopyOnWriteArrayList<ChatEvent>();
            CompletableFuture<Void> done = new CompletableFuture<>();

            ref.tell(a -> a.startPrompt("user1", "hello", "some-model", false, null,
                    events::add, ref, done));

            done.get(10, TimeUnit.SECONDS);
            // After completion, user should not be busy
            boolean busy = ref.ask(a -> a.isBusy("user1")).get(5, TimeUnit.SECONDS);
            assertFalse(busy);

            system.terminate();
        } finally {
            vllmLogger.setLevel(originalLevel);
        }
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

    // --- calculateMaxTokens ---

    @Test
    void calculateMaxTokens_smallContext() {
        // llm-jp: 4096 / 2 = 2048
        assertEquals(2048, ChatActor.calculateMaxTokens(4096));
    }

    @Test
    void calculateMaxTokens_largeContext() {
        // Qwen3.5: 262144 / 2 = 131072, capped at 8192
        assertEquals(8192, ChatActor.calculateMaxTokens(262144));
    }

    @Test
    void calculateMaxTokens_unknown() {
        assertEquals(1024, ChatActor.calculateMaxTokens(-1));
    }

    // --- calculateSendLimit ---

    @Test
    void calculateSendLimit_smallContext() {
        // llm-jp: (4096 - 2048) / 200 = 10
        assertEquals(10, ChatActor.calculateSendLimit(4096, 2048));
    }

    @Test
    void calculateSendLimit_largeContext() {
        // Qwen3.5: (262144 - 8192) / 200 = 1269, capped at 50
        assertEquals(50, ChatActor.calculateSendLimit(262144, 8192));
    }

    @Test
    void calculateSendLimit_unknown() {
        assertEquals(6, ChatActor.calculateSendLimit(-1, 1024));
    }

    @Test
    void calculateSendLimit_verySmallContext() {
        // (1024 - 512) / 200 = 2, minimum is 2
        assertEquals(2, ChatActor.calculateSendLimit(1024, 512));
    }

    // --- recentWindow ---

    @Test
    void recentWindow_limitsSize() {
        List<ChatMessage> history = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            history.add(new ChatMessage.User("q" + i));
            history.add(new ChatMessage.Assistant("a" + i));
        }
        List<ChatMessage> window = ChatActor.recentWindow(history, 6);
        assertEquals(6, window.size());
        // Should start with a user message
        assertInstanceOf(ChatMessage.User.class, window.get(0));
    }

    @Test
    void recentWindow_skipsLeadingAssistant() {
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("q1"));
        history.add(new ChatMessage.Assistant("a1"));
        history.add(new ChatMessage.User("q2"));
        history.add(new ChatMessage.Assistant("a2"));
        history.add(new ChatMessage.User("q3"));
        // limit=4 => start at index 1 (Assistant), should skip to index 2 (User)
        List<ChatMessage> window = ChatActor.recentWindow(history, 4);
        assertInstanceOf(ChatMessage.User.class, window.get(0));
        assertEquals("q2", ((ChatMessage.User) window.get(0)).content());
    }

    @Test
    void recentWindow_smallerThanLimit() {
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("q1"));
        history.add(new ChatMessage.Assistant("a1"));
        List<ChatMessage> window = ChatActor.recentWindow(history, 10);
        assertEquals(2, window.size());
    }

    @Test
    void recentWindow_isMutableCopy() {
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("q1"));
        List<ChatMessage> window = ChatActor.recentWindow(history, 10);
        window.add(new ChatMessage.User("q2"));
        // Original should not be modified
        assertEquals(1, history.size());
    }

    // --- preTrimToFit ---

    @Test
    void preTrimToFit_withinBudget_noChange() {
        ChatActor actor = createActor("");
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("hello"));       // 5 chars
        history.add(new ChatMessage.Assistant("world"));   // 5 chars
        actor.preTrimToFit(history, 100);
        assertEquals(2, history.size());
    }

    @Test
    void preTrimToFit_removesOldMessagesFirst() {
        ChatActor actor = createActor("");
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("old1"));         // 4
        history.add(new ChatMessage.Assistant("old2"));    // 4
        history.add(new ChatMessage.User("old3"));         // 4
        history.add(new ChatMessage.Assistant("old4"));    // 4
        history.add(new ChatMessage.User("current"));      // 7
        // Total: 23 chars, budget: 10
        actor.preTrimToFit(history, 10);
        // Should keep only last message
        assertEquals(1, history.size());
        assertEquals("current", ((ChatMessage.User) history.get(0)).content());
    }

    @Test
    void preTrimToFit_truncatesLastUserFromEnd() {
        ChatActor actor = createActor("");
        List<ChatMessage> history = new ArrayList<>();
        // 200 chars of content
        String longContent = "prompt\n\n---\nContent of http://example.com:\n" + "x".repeat(160);
        history.add(new ChatMessage.User(longContent));
        // Budget: 50 chars
        actor.preTrimToFit(history, 50);
        assertEquals(1, history.size());
        String result = ((ChatMessage.User) history.get(0)).content();
        // Should be truncated, starts with "prompt" (beginning preserved)
        assertTrue(result.startsWith("prompt"));
        assertTrue(result.length() < longContent.length());
        assertTrue(result.contains("[... content truncated"));
    }

    @Test
    void preTrimToFit_removesOldThenTruncates() {
        ChatActor actor = createActor("");
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("old question"));
        history.add(new ChatMessage.Assistant("old answer"));
        String longContent = "my prompt\n\n---\nContent:\n" + "y".repeat(300);
        history.add(new ChatMessage.User(longContent));
        // Budget: 80 chars - should remove old messages, then truncate last
        actor.preTrimToFit(history, 80);
        assertEquals(1, history.size());
        String result = ((ChatMessage.User) history.get(0)).content();
        assertTrue(result.startsWith("my prompt"));
        assertTrue(result.length() < longContent.length());
    }

    // --- estimateTotalChars ---

    @Test
    void estimateTotalChars_empty() {
        assertEquals(0, ChatActor.estimateTotalChars(new ArrayList<>()));
    }

    @Test
    void estimateTotalChars_mixed() {
        List<ChatMessage> history = new ArrayList<>();
        history.add(new ChatMessage.User("hello"));        // 5
        history.add(new ChatMessage.Assistant("world!"));   // 6
        assertEquals(11, ChatActor.estimateTotalChars(history));
    }

    // --- Log ring buffer ---

    @Test
    void publishLog_andGetRecentLogs() {
        ChatActor actor = createActor("");
        actor.publishLog("INFO", "test.Logger", "test message", System.currentTimeMillis());
        List<ChatEvent> logs = actor.getRecentLogs();
        assertEquals(1, logs.size());
        assertEquals("log", logs.get(0).type());
        assertEquals("test message", logs.get(0).content());
    }

    // --- helpers ---

    private static ChatActor createActor(String servers) {
        return new ChatActor(servers, 50);
    }
}
