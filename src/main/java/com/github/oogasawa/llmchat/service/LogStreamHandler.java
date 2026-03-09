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
import io.quarkus.runtime.Startup;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Captures JUL log records into a ring buffer and optionally forwards them
 * as SSE events for real-time display in the web UI.
 */
@ApplicationScoped
@Startup
public class LogStreamHandler extends Handler {

    private static final int BUFFER_SIZE = 500;
    private static final String OWN_LOGGER = LogStreamHandler.class.getName();

    private final ChatEvent[] buffer = new ChatEvent[BUFFER_SIZE];
    private int head = 0;
    private int count = 0;

    private volatile Consumer<ChatEvent> sseEmitter;

    @PostConstruct
    void init() {
        Logger.getLogger("").addHandler(this);
    }

    public void setSseEmitter(Consumer<ChatEvent> emitter) {
        this.sseEmitter = emitter;
    }

    public void clearSseEmitter() {
        this.sseEmitter = null;
    }

    @Override
    public synchronized void publish(LogRecord record) {
        if (record == null) return;
        String loggerName = record.getLoggerName();
        if (OWN_LOGGER.equals(loggerName)) return;

        String level = record.getLevel().getName();
        String message = formatMessage(record);
        long timestamp = record.getMillis();

        ChatEvent event = ChatEvent.log(level, loggerName, message, timestamp);

        buffer[head] = event;
        head = (head + 1) % BUFFER_SIZE;
        if (count < BUFFER_SIZE) count++;

        Consumer<ChatEvent> emitter = sseEmitter;
        if (emitter != null) {
            try {
                emitter.accept(event);
            } catch (Exception ignored) {
                // SSE write failure — don't recurse
            }
        }
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

    public synchronized List<ChatEvent> getRecentLogs() {
        List<ChatEvent> result = new ArrayList<>(count);
        int start = (head - count + BUFFER_SIZE) % BUFFER_SIZE;
        for (int i = 0; i < count; i++) {
            result.add(buffer[(start + i) % BUFFER_SIZE]);
        }
        return result;
    }

    @Override
    public void flush() {}

    @Override
    public void close() throws SecurityException {}
}
