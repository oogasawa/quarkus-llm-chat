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

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Fetches a web page and extracts its text content.
 */
public class UrlFetcher {

    private static final Logger logger = Logger.getLogger(UrlFetcher.class.getName());

    private static final int TIMEOUT_MS = 10_000;
    private static final int MAX_BODY_SIZE = 2 * 1024 * 1024; // 2MB
    private static final int MAX_TEXT_LENGTH = 30_000; // ~30K chars to avoid blowing up context

    /**
     * Fetches the URL and returns extracted text content.
     *
     * @param url the URL to fetch
     * @return the extracted text, or an error message prefixed with "[Error]"
     */
    public static String fetchAndExtract(String url) {
        try {
            Document doc = Jsoup.connect(url)
                    .userAgent("Mozilla/5.0 (compatible; LocalLLM/1.0)")
                    .timeout(TIMEOUT_MS)
                    .maxBodySize(MAX_BODY_SIZE)
                    .followRedirects(true)
                    .get();

            // Remove script, style, nav, footer elements
            doc.select("script, style, nav, footer, header, aside, .sidebar, .menu, .nav").remove();

            String title = doc.title();
            String text = doc.body() != null ? doc.body().text() : "";

            if (text.length() > MAX_TEXT_LENGTH) {
                text = text.substring(0, MAX_TEXT_LENGTH) + "\n... (truncated)";
            }

            StringBuilder result = new StringBuilder();
            if (!title.isEmpty()) {
                result.append("Title: ").append(title).append("\n\n");
            }
            result.append(text);

            logger.info("Fetched URL: " + url + " -> " + text.length() + " chars");
            return result.toString();

        } catch (Exception e) {
            logger.log(Level.WARNING, "Failed to fetch URL: " + url, e);
            return "[Error] Failed to fetch " + url + ": " + e.getMessage();
        }
    }
}
