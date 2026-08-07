# shared-inference

A small async client for OpenAI-compatible inference services. It uses
`niquests` for HTTP and `opentelemetry-api` for optional, privacy-safe LLM
spans; it does not depend on provider SDKs or an LLM framework.

## API

```python
from shared_inference import InferenceClient

client = InferenceClient(
    base_url="https://openrouter.ai/api/v1",
    api_key="...",
    provider="openrouter",
    domain="browser_copilot",
)

completion = await client.complete(
    model="model-id",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0,
)
embedding = await client.embed(model="embedding-model", input=["text"])
rerank = await client.rerank(
    model="rerank-model", query="question", documents=["document"]
)
await client.close()
```

The client preserves raw JSON, usage, provider request IDs, tool calls,
reasoning fields, multimodal messages, response formats, sparse embedding
extensions, rerank indices, and relevance scores. HTTP failures, transport
failures, and timeouts are exposed as typed exceptions.

The `base_url` should be the API root. For example, use
`https://openrouter.ai/api/v1` for OpenRouter or `http://host:8102/v1` for a
local embeddings server. The client appends `/chat/completions`,
`/embeddings`, or `/rerank` for the selected operation.

## Tracing

The client creates `llm.complete`, `llm.embed`, and `llm.rerank` spans through
OpenTelemetry when an application has configured a tracer provider. Without a
provider, tracing is a no-op. Spans contain operation, domain, provider,
model, latency, usage, request ID, and error/status metadata; prompts,
documents, images, and API keys are not recorded.

## Development

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
uv run pytest
```

The live tests are opt-in and expect local inference services:

```bash
SHARED_INFERENCE_LIVE=1 uv run pytest -m live
```

GitHub Actions runs formatting, linting, and the non-live test suite on Python
3.14.
