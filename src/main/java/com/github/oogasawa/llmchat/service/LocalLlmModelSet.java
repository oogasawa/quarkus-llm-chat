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

import com.github.oogasawa.llmchat.vllm.VllmClient;

import java.net.URI;
import java.util.ArrayList;
import java.util.List;

/**
 * Model set for local LLM mode.
 *
 * <p>Discovers available models dynamically by querying each configured
 * OpenAI-compatible server's /v1/models endpoint. All models in this set
 * are routed to VllmClient.</p>
 *
 * @author oogasawa
 */
public class LocalLlmModelSet extends ModelSet {

    private final List<VllmClient> clients;

    public LocalLlmModelSet(List<VllmClient> clients) {
        this.clients = clients;
    }

    @Override
    public List<ModelEntry> getAvailableModels() {
        List<ModelEntry> models = new ArrayList<>();
        for (VllmClient client : clients) {
            String server = extractHost(client.getBaseUrl());
            for (String name : client.fetchModels()) {
                models.add(new ModelEntry(name, "local", server));
            }
        }
        return models;
    }

    @Override
    public boolean isLocalModel(String model) {
        return true;
    }

    static String extractHost(String url) {
        try {
            URI uri = URI.create(url);
            String host = uri.getHost();
            if (host == null) {
                return url;
            }
            return uri.getPort() > 0
                    ? host + ":" + uri.getPort()
                    : host;
        } catch (Exception e) {
            return url;
        }
    }
}
