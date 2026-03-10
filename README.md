# quarkus-llm-chat

A lightweight multi-tenant Web UI for local LLM servers. Connect to vLLM, Ollama, or any OpenAI-compatible API server and chat from your browser with real-time SSE streaming.

Each user (identified by BasicAuth) gets an isolated conversation history and SSE connection.

## Prerequisites

- Java 21+
- Maven 3.9+
- Local LLM server (vLLM, Ollama, etc.) exposing the OpenAI-compatible API (`/v1/chat/completions`)

## Build

```bash
git clone https://github.com/oogasawa/quarkus-llm-chat.git
cd quarkus-llm-chat
rm -rf target
mvn install
```

The runnable jar is generated at `target/quarkus-app/quarkus-run.jar`.

## Configuration

Specify your LLM server URLs in `src/main/resources/application.properties`:

```properties
# Local LLM servers (OpenAI-compatible API). Comma-separated URLs.
llm-chat.servers=http://192.168.5.15:8000,http://192.168.5.13:8000

# Max conversation history per user (default: 50)
llm-chat.max-history=50

# Application title
llm-chat.title=LLM Chat
```

Override via system properties:

```bash
java -Dllm-chat.servers=http://192.168.5.15:8000 \
     -jar target/quarkus-app/quarkus-run.jar
```

## Run

```bash
java -jar target/quarkus-app/quarkus-run.jar
```

Open `http://localhost:8090` in your browser.

## Features

- **Multi-tenant** — per-user conversation history and SSE streams (BasicAuth)
- **Prompt queue** — queue up prompts while the AI is responding
- **Multiple LLM server support** — connect to vLLM, Ollama, and other OpenAI-compatible servers simultaneously
- **URL fetch** — paste URLs in your prompt, click the Fetch button, and the page content is automatically prepended as reference context (RAG-style: context first, question last)
- **Image OCR** — paste or drag-and-drop images into the prompt area; they are sent to vision-capable models for OCR and analysis
- **Smart context management** — pre-trims conversation history and fetched content to fit within the model's context window, preserving the user's question and the beginning of reference material
- Real-time streaming responses (SSE) with Markdown rendering
- Dynamic model list with refresh button
- Save conversation history as Markdown file
- Copy button on both user prompts and assistant responses
- 10 color themes

## REST API

```bash
# List available models
curl http://localhost:8090/api/models

# Send a prompt
curl -X POST http://localhost:8090/api/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Basic dXNlcjpwYXNz' \
  -d '{"text":"Hello","model":"/models/Qwen3-Coder"}'
```

## Test

```bash
rm -rf target
mvn test
```

## License

Apache License 2.0
