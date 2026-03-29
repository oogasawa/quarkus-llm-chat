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

import com.github.oogasawa.llmchat.actor.ChatActor;
import com.github.oogasawa.llmchat.rest.ChatEvent;
import com.scivicslab.pojoactor.core.ActorRef;
import io.quarkus.runtime.Startup;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * JUL Handler that forwards log records to {@link ChatActor}.
 * The ring buffer and SSE emitter state now live entirely in the actor;
 * this class is a thin bridge that must remain a JUL Handler so it can be
 * registered with the root logger.
 * <p>
 * The actor reference is wired in by {@link com.github.oogasawa.llmchat.actor.LlmConsoleActorSystem}
 * after the actor is constructed, avoiding the circular-init problem that occurs
 * when ChatActor logs during construction and publish() tries to access CDI.
 * </p>
 */
@ApplicationScoped
@Startup
public class LogStreamHandler extends Handler {

    private static final String OWN_LOGGER = LogStreamHandler.class.getName();

    // Set by LlmConsoleActorSystem after the actor is created; volatile for cross-thread visibility.
    private volatile ActorRef<ChatActor> chatActorRef;

    @PostConstruct
    void init() {
        Logger.getLogger("").addHandler(this);
    }

    /** Called by LlmConsoleActorSystem once the ChatActor is ready. */
    public void wireActorRef(ActorRef<ChatActor> ref) {
        this.chatActorRef = ref;
    }

    public void setSseEmitter(Consumer<ChatEvent> emitter) {
        var actor = chatActorRef;
        if (actor != null) actor.tell(a -> a.setSseEmitter(emitter));
    }

    public void clearSseEmitter() {
        var actor = chatActorRef;
        if (actor != null) actor.tell(a -> a.clearSseEmitter());
    }

    @Override
    public void publish(LogRecord record) {
        if (record == null) return;
        if (OWN_LOGGER.equals(record.getLoggerName())) return;
        var actor = chatActorRef;  // volatile read -- null before wireActorRef() is called
        if (actor == null) return;
        String level = record.getLevel().getName();
        String loggerName = record.getLoggerName();
        String message = formatMessage(record);
        long timestamp = record.getMillis();
        actor.tell(a -> a.publishLog(level, loggerName, message, timestamp));
    }

    private String formatMessage(LogRecord record) {
        String msg = record.getMessage();
        if (msg == null) return "";
        Object[] params = record.getParameters();
        if (params != null && params.length > 0) {
            try {
                return java.text.MessageFormat.format(msg, params);
            } catch (Exception e) {
                return msg;
            }
        }
        return msg;
    }

    public List<ChatEvent> getRecentLogs() {
        var actor = chatActorRef;
        if (actor == null) return List.of();
        return actor.ask(a -> a.getRecentLogs()).join();
    }

    @Override
    public void flush() {}

    @Override
    public void close() throws SecurityException {}
}
