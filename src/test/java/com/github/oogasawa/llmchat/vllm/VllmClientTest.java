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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for VllmClient static parsing and utility methods.
 * No HTTP calls -- tests only the pure logic.
 *
 * @author oogasawa
 */
class VllmClientTest {

    // --- parseSseLine tests ---

    @Test
    void parseSseLine_normalContentChunk() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}";
        assertEquals("Hello", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_contentWithNewline() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"line1\\nline2\"}}]}";
        assertEquals("line1\nline2", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_contentWithQuote() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"say \\\"hello\\\"\"}}]}";
        assertEquals("say \"hello\"", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_doneSignal() {
        assertNull(VllmClient.parseSseLine("data: [DONE]"));
    }

    @Test
    void parseSseLine_nullInput() {
        assertNull(VllmClient.parseSseLine(null));
    }

    @Test
    void parseSseLine_emptyLine() {
        assertNull(VllmClient.parseSseLine(""));
    }

    @Test
    void parseSseLine_nonDataLine() {
        assertNull(VllmClient.parseSseLine("event: message"));
    }

    @Test
    void parseSseLine_emptyDelta() {
        String line = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}";
        assertNull(VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_emptyContent() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}";
        assertEquals("", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_contentWithUnicode() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"\\u3053\\u3093\\u306b\\u3061\\u306f\"}}]}";
        assertEquals("\u3053\u3093\u306b\u3061\u306f", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_contentWithBackslash() {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"path\\\\to\\\\file\"}}]}";
        assertEquals("path\\to\\file", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_realisticChunk() {
        String line = "data: {\"id\":\"cmpl-123\",\"object\":\"chat.completion.chunk\","
                + "\"created\":1709000000,\"model\":\"Qwen/Qwen2.5-Coder-32B-Instruct\","
                + "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"def \"},"
                + "\"finish_reason\":null}]}";
        assertEquals("def ", VllmClient.parseSseLine(line));
    }

    @Test
    void parseSseLine_finishReasonStop() {
        String line = "data: {\"choices\":[{\"index\":0,\"delta\":{},"
                + "\"finish_reason\":\"stop\"}]}";
        assertNull(VllmClient.parseSseLine(line));
    }

    // --- buildRequestBody tests ---

    @Test
    void buildRequestBody_singleMessage() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("hello")
        );
        String body = VllmClient.buildRequestBody("test-model", messages, false, 2048);

        assertTrue(body.contains("\"model\":\"test-model\""));
        assertTrue(body.contains("\"stream\":true"));
        assertTrue(body.contains("\"max_tokens\":2048"));
        assertTrue(body.contains("\"role\":\"user\""));
        assertTrue(body.contains("\"content\":\"hello\""));
    }

    @Test
    void buildRequestBody_multipleMessages() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("hello"),
                new ChatMessage.Assistant("hi"),
                new ChatMessage.User("how are you")
        );
        String body = VllmClient.buildRequestBody("Qwen/Qwen2.5-Coder-32B-Instruct", messages, false, 4096);

        assertTrue(body.contains("\"model\":\"Qwen/Qwen2.5-Coder-32B-Instruct\""));
        assertTrue(body.contains("\"content\":\"hello\""));
        assertTrue(body.contains("\"content\":\"hi\""));
        assertTrue(body.contains("\"content\":\"how are you\""));
    }

    @Test
    void buildRequestBody_specialCharacters() {
        List<ChatMessage> messages = List.of(
                new ChatMessage.User("line1\nline2\t\"quoted\"")
        );
        String body = VllmClient.buildRequestBody("model", messages, false, 2048);

        assertTrue(body.contains("\\n"));
        assertTrue(body.contains("\\t"));
        assertTrue(body.contains("\\\"quoted\\\""));
    }

    @Test
    void buildRequestBody_emptyMessages() {
        List<ChatMessage> messages = List.of();
        String body = VllmClient.buildRequestBody("model", messages, false, 2048);
        assertTrue(body.contains("\"messages\":[]"));
    }

    // --- sendPrompt with unreachable host ---

    @Test
    void sendPrompt_unreachableHost_callsOnError() {
        Logger vllmLogger = Logger.getLogger(VllmClient.class.getName());
        Level originalLevel = vllmLogger.getLevel();
        vllmLogger.setLevel(Level.OFF);
        try {
            VllmClient client = new VllmClient("http://localhost:1");
            List<ChatMessage> history = new ArrayList<>();
            history.add(new ChatMessage.User("hello"));

            var errors = new ArrayList<String>();
            String result = client.sendPrompt("model", history, false, 2048, new VllmClient.StreamCallback() {
                @Override public void onDelta(String content) {}
                @Override public void onComplete(long durationMs) {}
                @Override public void onError(String message) { errors.add(message); }
            });

            assertNull(result);
            assertFalse(errors.isEmpty());
        } finally {
            vllmLogger.setLevel(originalLevel);
        }
    }

    // --- escapeJson tests ---

    @Test
    void escapeJson_plainText() {
        assertEquals("hello", VllmClient.escapeJson("hello"));
    }

    @Test
    void escapeJson_quotes() {
        assertEquals("say \\\"hello\\\"", VllmClient.escapeJson("say \"hello\""));
    }

    @Test
    void escapeJson_backslash() {
        assertEquals("path\\\\to\\\\file", VllmClient.escapeJson("path\\to\\file"));
    }

    @Test
    void escapeJson_newlineAndTab() {
        assertEquals("line1\\nline2\\ttab", VllmClient.escapeJson("line1\nline2\ttab"));
    }

    @Test
    void escapeJson_controlCharacters() {
        String result = VllmClient.escapeJson("\u0000\u001f");
        assertEquals("\\u0000\\u001f", result);
    }

    @Test
    void escapeJson_nullInput() {
        assertEquals("", VllmClient.escapeJson(null));
    }

    @Test
    void escapeJson_japanese() {
        assertEquals("\u3053\u3093\u306b\u3061\u306f", VllmClient.escapeJson("\u3053\u3093\u306b\u3061\u306f"));
    }

    // --- unescapeJsonString tests ---

    @Test
    void unescapeJsonString_simple() {
        String json = "\"hello\"rest";
        assertEquals("hello", VllmClient.unescapeJsonString(json, 1));
    }

    @Test
    void unescapeJsonString_withEscapes() {
        String json = "line1\\nline2\"";
        assertEquals("line1\nline2", VllmClient.unescapeJsonString(json, 0));
    }

    @Test
    void unescapeJsonString_unicodeEscape() {
        String json = "\\u0041\\u0042\"";
        assertEquals("AB", VllmClient.unescapeJsonString(json, 0));
    }

    // --- extractDeltaContent tests ---

    @Test
    void extractDeltaContent_normal() {
        String json = "{\"choices\":[{\"delta\":{\"content\":\"test\"}}]}";
        assertEquals("test", VllmClient.extractDeltaContent(json));
    }

    @Test
    void extractDeltaContent_noDelta() {
        String json = "{\"choices\":[{\"message\":{\"content\":\"test\"}}]}";
        assertNull(VllmClient.extractDeltaContent(json));
    }

    @Test
    void extractDeltaContent_emptyDelta() {
        String json = "{\"choices\":[{\"delta\":{}}]}";
        assertNull(VllmClient.extractDeltaContent(json));
    }

    @Test
    void extractDeltaContent_roleOnly() {
        String json = "{\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}";
        assertNull(VllmClient.extractDeltaContent(json));
    }

    // --- parseModelIds tests ---

    @Test
    void parseModelIds_singleModel() {
        String json = "{\"object\":\"list\",\"data\":[{\"id\":\"/models/Qwen2.5-Coder-32B-Instruct\",\"object\":\"model\"}]}";
        List<String> ids = VllmClient.parseModelIds(json);
        assertEquals(1, ids.size());
        assertEquals("/models/Qwen2.5-Coder-32B-Instruct", ids.get(0));
    }

    @Test
    void parseModelIds_multipleModels() {
        String json = "{\"data\":[{\"id\":\"model-a\"},{\"id\":\"model-b\"},{\"id\":\"model-c\"}]}";
        List<String> ids = VllmClient.parseModelIds(json);
        assertEquals(3, ids.size());
        assertEquals("model-a", ids.get(0));
        assertEquals("model-b", ids.get(1));
        assertEquals("model-c", ids.get(2));
    }

    @Test
    void parseModelIds_emptyData() {
        String json = "{\"data\":[]}";
        List<String> ids = VllmClient.parseModelIds(json);
        assertTrue(ids.isEmpty());
    }

    @Test
    void parseModelIds_realisticVllmResponse() {
        String json = "{\"object\":\"list\",\"data\":[{\"id\":\"/models/Qwen3-Coder-30B-A3B-Instruct\","
                + "\"object\":\"model\",\"created\":1771827094,\"owned_by\":\"vllm\","
                + "\"root\":\"/models/Qwen3-Coder-30B-A3B-Instruct\",\"parent\":null,"
                + "\"max_model_len\":262144,\"permission\":[{\"id\":\"modelperm-123\"}]}]}";
        List<String> ids = VllmClient.parseModelIds(json);
        assertEquals(1, ids.size());
        assertEquals("/models/Qwen3-Coder-30B-A3B-Instruct", ids.get(0));
    }

    // --- servesModel / cachedModels tests ---

    @Test
    void servesModel_emptyByDefault() {
        VllmClient client = new VllmClient("http://localhost:8000");
        assertFalse(client.servesModel("any-model"));
    }

    @Test
    void getCachedModels_emptyByDefault() {
        VllmClient client = new VllmClient("http://localhost:8000");
        assertTrue(client.getCachedModels().isEmpty());
    }

    // --- isContextLengthError tests ---

    @Test
    void isContextLengthError_true() {
        assertTrue(VllmClient.isContextLengthError("exceeds context length limit"));
        assertTrue(VllmClient.isContextLengthError("input_tokens exceeded"));
        assertTrue(VllmClient.isContextLengthError("maximum input length"));
    }

    @Test
    void isContextLengthError_false() {
        assertFalse(VllmClient.isContextLengthError("some other error"));
        assertFalse(VllmClient.isContextLengthError(null));
    }

    // --- parseMaxModelLens tests ---

    @Test
    void parseMaxModelLens_singleModel() {
        String json = "{\"object\":\"list\",\"data\":[{\"id\":\"llm-jp-3-3.7b-instruct\","
                + "\"object\":\"model\",\"max_model_len\":4096}]}";
        Map<String, Integer> result = VllmClient.parseMaxModelLens(json);
        assertEquals(1, result.size());
        assertEquals(4096, result.get("llm-jp-3-3.7b-instruct"));
    }

    @Test
    void parseMaxModelLens_multipleModels() {
        String json = "{\"data\":[{\"id\":\"model-a\",\"max_model_len\":4096},"
                + "{\"id\":\"model-b\",\"max_model_len\":262144}]}";
        Map<String, Integer> result = VllmClient.parseMaxModelLens(json);
        assertEquals(2, result.size());
        assertEquals(4096, result.get("model-a"));
        assertEquals(262144, result.get("model-b"));
    }

    @Test
    void parseMaxModelLens_realisticVllmResponse() {
        String json = "{\"object\":\"list\",\"data\":[{\"id\":\"/models/Qwen3-Coder-30B-A3B-Instruct\","
                + "\"object\":\"model\",\"created\":1771827094,\"owned_by\":\"vllm\","
                + "\"root\":\"/models/Qwen3-Coder-30B-A3B-Instruct\",\"parent\":null,"
                + "\"max_model_len\":262144,\"permission\":[{\"id\":\"modelperm-123\"}]}]}";
        Map<String, Integer> result = VllmClient.parseMaxModelLens(json);
        assertEquals(262144, result.get("/models/Qwen3-Coder-30B-A3B-Instruct"));
    }

    @Test
    void parseMaxModelLens_noMaxModelLen() {
        String json = "{\"data\":[{\"id\":\"model-a\",\"object\":\"model\"}]}";
        Map<String, Integer> result = VllmClient.parseMaxModelLens(json);
        assertTrue(result.isEmpty());
    }

    @Test
    void parseMaxModelLens_emptyData() {
        String json = "{\"data\":[]}";
        Map<String, Integer> result = VllmClient.parseMaxModelLens(json);
        assertTrue(result.isEmpty());
    }

    // --- getMaxModelLen tests ---

    @Test
    void getMaxModelLen_unknownModel() {
        VllmClient client = new VllmClient("http://localhost:8000");
        assertEquals(-1, client.getMaxModelLen("unknown-model"));
    }

    // --- buildRequestBody max_tokens tests ---

    @Test
    void buildRequestBody_withMaxTokens() {
        List<ChatMessage> messages = List.of(new ChatMessage.User("hello"));
        String body = VllmClient.buildRequestBody("model", messages, false, 4096);
        assertTrue(body.contains("\"max_tokens\":4096"));
    }

    @Test
    void buildRequestBody_zeroMaxTokens_omitted() {
        List<ChatMessage> messages = List.of(new ChatMessage.User("hello"));
        String body = VllmClient.buildRequestBody("model", messages, false, 0);
        assertFalse(body.contains("max_tokens"));
    }

    @Test
    void buildRequestBody_negativeMaxTokens_omitted() {
        List<ChatMessage> messages = List.of(new ChatMessage.User("hello"));
        String body = VllmClient.buildRequestBody("model", messages, false, -1);
        assertFalse(body.contains("max_tokens"));
    }
}
