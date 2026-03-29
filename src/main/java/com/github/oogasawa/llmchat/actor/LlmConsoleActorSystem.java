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

import com.github.oogasawa.llmchat.service.LogStreamHandler;
import com.scivicslab.pojoactor.core.ActorRef;
import com.scivicslab.pojoactor.core.ActorSystem;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.logging.Logger;

/**
 * CDI bean that owns the POJO-actor system for this application.
 * Creates and holds the single {@link ChatActor} instance.
 */
@ApplicationScoped
public class LlmConsoleActorSystem {

    private static final Logger LOG = Logger.getLogger(LlmConsoleActorSystem.class.getName());

    @ConfigProperty(name = "llm-chat.servers", defaultValue = "")
    String llmServers;

    @ConfigProperty(name = "llm-chat.max-history", defaultValue = "50")
    int maxHistory;

    @Inject
    LogStreamHandler logStreamHandler;

    private ActorSystem actorSystem;
    private ActorRef<ChatActor> chatActorRef;

    @PostConstruct
    void init() {
        actorSystem = new ActorSystem("llm-chat");
        chatActorRef = actorSystem.actorOf("chat",
                new ChatActor(llmServers, maxHistory));
        // Wire the actor ref into LogStreamHandler now that construction is complete.
        logStreamHandler.wireActorRef(chatActorRef);
        LOG.info("LlmConsoleActorSystem initialized");
    }

    @PreDestroy
    void shutdown() {
        if (actorSystem != null) actorSystem.terminate();
    }

    public ActorRef<ChatActor> getChatActor() {
        return chatActorRef;
    }
}
