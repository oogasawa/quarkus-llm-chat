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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.oogasawa.llmchat.service.ChatService;
import com.github.oogasawa.llmchat.service.LogStreamHandler;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.http.HttpServerResponse;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import io.smallrye.common.annotation.Blocking;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.Context;
import jakarta.ws.rs.core.HttpHeaders;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Multi-tenant REST + SSE endpoint for chat interaction.
 *
 * <p>Each user (identified by BasicAuth) gets their own SSE connection
 * and isolated conversation history.</p>
 *
 * <ul>
 *   <li>GET  /api/chat/stream - Persistent SSE stream (raw Vert.x route)</li>
 *   <li>POST /api/chat - Submit a prompt (events arrive via SSE stream)</li>
 *   <li>POST /api/cancel - Cancel the current prompt</li>
 *   <li>POST /api/clear - Clear conversation history</li>
 *   <li>GET  /api/status - Get current status</li>
 *   <li>GET  /api/models - Get available models</li>
 *   <li>GET  /api/config - Get app configuration</li>
 * </ul>
 *
 * @author oogasawa
 */
@Path("/api")
@Blocking
public class ChatResource {

    private static final Logger logger = Logger.getLogger(ChatResource.class.getName());

    @Inject
    ChatService chatService;

    @Inject
    LogStreamHandler logStreamHandler;

    @Inject
    Vertx vertx;

    @Inject
    ObjectMapper objectMapper;

    @ConfigProperty(name = "llm-chat.title", defaultValue = "Local LLM")
    String appTitle;

    @ConfigProperty(name = "llm-chat.single-user-mode", defaultValue = "false")
    boolean singleUserMode;

    private static final String DEFAULT_USER = "default";

    /** Per-user SSE connections: userId -> HttpServerResponse. */
    private final ConcurrentHashMap<String, HttpServerResponse> sseConnections = new ConcurrentHashMap<>();

    /** Per-user heartbeat timer IDs. */
    private final ConcurrentHashMap<String, Long> heartbeatTimers = new ConcurrentHashMap<>();

    /**
     * Registers a raw Vert.x route for SSE streaming.
     */
    void registerSseRoute(@Observes Router router) {
        router.get("/api/chat/stream").handler(this::handleSseConnect);
    }

    /**
     * Handles a new SSE client connection.
     * User is identified from the BasicAuth header on the SSE request.
     */
    private void handleSseConnect(RoutingContext rc) {
        // Try query parameter first (EventSource cannot send custom headers),
        // then fall back to BasicAuth header.
        var userParams = rc.queryParam("user");
        String queryUser = (userParams != null && !userParams.isEmpty()) ? userParams.get(0) : null;
        String resolved = (queryUser != null && !queryUser.isBlank())
                ? queryUser : extractUserId(rc.request());
        final String userId = (resolved != null) ? resolved
                : (singleUserMode ? DEFAULT_USER : null);
        if (userId == null) {
            rc.response().setStatusCode(401).end("Unauthorized");
            return;
        }

        // Close previous SSE connection for this user
        var prev = sseConnections.get(userId);
        if (prev != null && !prev.ended()) {
            try { prev.end(); } catch (Exception ignored) {}
        }
        Long prevTimer = heartbeatTimers.remove(userId);
        if (prevTimer != null) {
            vertx.cancelTimer(prevTimer);
        }

        var response = rc.response();
        response.setChunked(true);
        response.putHeader("Content-Type", "text/event-stream");
        response.putHeader("Cache-Control", "no-cache");
        response.putHeader("X-Accel-Buffering", "no");

        sseConnections.put(userId, response);

        // Tell browser to wait 10 seconds before reconnecting
        response.write("retry: 10000\n\n");

        // Send initial status
        writeSse(response, ChatEvent.status(null, null, chatService.isBusy(userId)));

        // Heartbeat every 15 seconds
        long timerId = vertx.setPeriodic(15_000, id -> {
            var r = sseConnections.get(userId);
            if (r != null && !r.ended()) {
                writeSse(r, ChatEvent.heartbeat());
            } else {
                vertx.cancelTimer(id);
            }
        });
        heartbeatTimers.put(userId, timerId);

        // Forward server logs to all connected users
        logStreamHandler.setSseEmitter(event -> broadcastToAll(event));

        response.closeHandler(v -> {
            sseConnections.remove(userId, response);
            Long tid = heartbeatTimers.remove(userId);
            if (tid != null) {
                vertx.cancelTimer(tid);
            }
            if (sseConnections.isEmpty()) {
                logStreamHandler.clearSseEmitter();
            }
        });

        logger.info("SSE connected: user=" + userId);
    }

    /**
     * Writes a single SSE event to a response.
     */
    private void writeSse(HttpServerResponse response, ChatEvent event) {
        try {
            String json = objectMapper.writeValueAsString(event);
            response.write("data: " + json + "\n\n");
        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to write SSE event: type=" + event.type(), e);
        }
    }

    /**
     * Emits a ChatEvent to a specific user's SSE connection.
     */
    private void emitSse(String userId, ChatEvent event) {
        var resp = sseConnections.get(userId);
        if (resp != null && !resp.ended()) {
            vertx.runOnContext(v -> writeSse(resp, event));
        } else {
            logger.warning("SSE event DROPPED (no connection for " + userId
                    + ", resp=" + (resp == null ? "null" : "ended") + "): type=" + event.type());
        }
    }

    /**
     * Broadcasts a ChatEvent to all connected users (used for log events).
     */
    private void broadcastToAll(ChatEvent event) {
        for (var entry : sseConnections.entrySet()) {
            var resp = entry.getValue();
            if (resp != null && !resp.ended()) {
                vertx.runOnContext(v -> writeSse(resp, event));
            }
        }
    }

    /**
     * Submits a prompt.
     */
    @POST
    @Path("/chat")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public ChatEvent chat(PromptRequest request, @QueryParam("user") String queryUser, @Context HttpHeaders headers) {
        String userId = resolveUserId(queryUser, headers);
        if (userId == null) {
            return ChatEvent.error("Unauthorized");
        }

        if (request == null || request.text == null || request.text.isBlank()) {
            return ChatEvent.error("Empty prompt");
        }

        if (!sseConnections.containsKey(userId)) {
            return ChatEvent.error("No SSE connection. Please refresh the page.");
        }

        String model = request.model;

        Thread.startVirtualThread(() -> {
            try {
                chatService.sendPrompt(userId, request.text, model, request.noThink,
                        request.images, event -> emitSse(userId, event));
            } catch (Exception e) {
                logger.log(Level.WARNING, "Chat prompt failed for user " + userId, e);
                emitSse(userId, ChatEvent.error("Internal error: " + e.getMessage()));
            }
        });

        return ChatEvent.info("Processing");
    }

    /**
     * Cancels the currently running prompt.
     */
    @POST
    @Path("/cancel")
    @Produces(MediaType.APPLICATION_JSON)
    public ChatEvent cancel(@QueryParam("user") String queryUser, @Context HttpHeaders headers) {
        String userId = resolveUserId(queryUser, headers);
        if (userId == null) {
            return ChatEvent.error("Unauthorized");
        }
        chatService.cancel(userId);
        return ChatEvent.info("Cancelled");
    }

    /**
     * Clears the conversation history.
     */
    @POST
    @Path("/clear")
    @Produces(MediaType.APPLICATION_JSON)
    public ChatEvent clear(@QueryParam("user") String queryUser, @Context HttpHeaders headers) {
        String userId = resolveUserId(queryUser, headers);
        if (userId == null) {
            return ChatEvent.error("Unauthorized");
        }
        chatService.clearHistory(userId);
        return ChatEvent.info("History cleared");
    }

    /**
     * Returns current status for the user.
     */
    @GET
    @Path("/status")
    @Produces(MediaType.APPLICATION_JSON)
    public ChatEvent status(@QueryParam("user") String queryUser, @Context HttpHeaders headers) {
        String userId = resolveUserId(queryUser, headers);
        boolean busy = userId != null && chatService.isBusy(userId);
        return ChatEvent.status(null, null, busy);
    }

    /**
     * Returns the list of available models.
     */
    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    public List<ModelInfo> models() {
        return chatService.getAvailableModels().stream()
                .map(e -> new ModelInfo(e.name(), e.type(), e.server()))
                .toList();
    }

    /**
     * Returns recent server log entries from the ring buffer.
     */
    @GET
    @Path("/logs")
    @Produces(MediaType.APPLICATION_JSON)
    public List<ChatEvent> logs() {
        return logStreamHandler.getRecentLogs();
    }

    /**
     * Returns application configuration for the frontend.
     */
    @GET
    @Path("/config")
    @Produces(MediaType.APPLICATION_JSON)
    public AppConfig config() {
        return new AppConfig(appTitle, singleUserMode);
    }

    /**
     * Fetches a URL and returns its text content.
     */
    @POST
    @Path("/fetch-url")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public FetchResult fetchUrl(FetchRequest request) {
        if (request == null || request.url == null || request.url.isBlank()) {
            return new FetchResult(false, "", "Empty URL");
        }
        String url = request.url.trim();
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            return new FetchResult(false, "", "Invalid URL (must start with http:// or https://)");
        }
        String content = com.github.oogasawa.llmchat.service.UrlFetcher.fetchAndExtract(url);
        if (content.startsWith("[Error]")) {
            return new FetchResult(false, "", content);
        }
        return new FetchResult(true, content, null);
    }

    // --- User extraction ---

    /**
     * Extracts user ID: query parameter takes precedence, then BasicAuth header.
     * Falls back to DEFAULT_USER in single-user mode.
     */
    String resolveUserId(String queryUser, HttpHeaders headers) {
        if (queryUser != null && !queryUser.isBlank()) {
            return queryUser;
        }
        String fromAuth = extractUserId(headers);
        if (fromAuth != null) {
            return fromAuth;
        }
        return singleUserMode ? DEFAULT_USER : null;
    }

    /**
     * Extracts user ID from BasicAuth header (JAX-RS HttpHeaders).
     *
     * @return the username, or null if not authenticated
     */
    static String extractUserId(HttpHeaders headers) {
        String auth = headers.getHeaderString("Authorization");
        return parseBasicAuth(auth);
    }

    /**
     * Extracts user ID from BasicAuth header (Vert.x request).
     *
     * @return the username, or null if not authenticated
     */
    static String extractUserId(HttpServerRequest request) {
        String auth = request.getHeader("Authorization");
        return parseBasicAuth(auth);
    }

    /**
     * Parses BasicAuth header value to extract username.
     *
     * @param authHeader the Authorization header value (e.g., "Basic dXNlcjpwYXNz")
     * @return the username, or null if invalid
     */
    public static String parseBasicAuth(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Basic ")) {
            return null;
        }
        try {
            String decoded = new String(
                    Base64.getDecoder().decode(authHeader.substring(6).trim()),
                    StandardCharsets.UTF_8);
            int colon = decoded.indexOf(':');
            if (colon <= 0) {
                return null;
            }
            return decoded.substring(0, colon);
        } catch (Exception e) {
            return null;
        }
    }

    // --- DTOs ---

    public record AppConfig(String title, boolean singleUserMode) {}

    public record ModelInfo(String name, String type, String server) {}

    public static class FetchRequest {
        public String url;
    }

    public record FetchResult(boolean ok, String content, String error) {}

    public static class PromptRequest {
        public String text;
        public String model;
        public boolean noThink;
        public List<String> images;
    }
}
