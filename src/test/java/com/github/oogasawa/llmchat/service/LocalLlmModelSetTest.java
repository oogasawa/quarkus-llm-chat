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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LocalLlmModelSetTest {

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
}
