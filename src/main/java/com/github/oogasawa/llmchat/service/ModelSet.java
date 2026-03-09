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

import java.util.List;

/**
 * Abstract base for model sets.
 *
 * <p>Each concrete subclass represents a deployment mode (local-llm, claude, etc.)
 * and provides the list of available models for the UI dropdown, plus routing logic
 * to determine whether a model should be sent to a local LLM server or Claude CLI.</p>
 *
 * @author oogasawa
 */
public abstract class ModelSet {

    /** A model entry with name, type (local/claude), and optional server host. */
    public record ModelEntry(String name, String type, String server) {}

    /**
     * Returns the list of available models for this deployment mode.
     * Called by the /api/models endpoint to populate the frontend dropdown.
     *
     * @return list of model entries
     */
    public abstract List<ModelEntry> getAvailableModels();

    /**
     * Determines whether the given model name should be routed to a local LLM server.
     *
     * @param model the model name selected by the user
     * @return true if the model is served by a local LLM server, false if by Claude CLI
     */
    public abstract boolean isLocalModel(String model);
}
