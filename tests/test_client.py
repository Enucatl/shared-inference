from unittest.mock import AsyncMock, Mock

import niquests
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from shared_inference import (
    InferenceClient,
    InferenceHTTPError,
    InferenceTimeoutError,
)


@pytest.mark.asyncio
async def test_chat_preserves_tools_and_raw_response() -> None:
    response = Mock(ok=True, status_code=200, headers={"x-request-id": "r1"})
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c"}],
                }
            }
        ],
        "usage": {"total_tokens": 4},
    }
    session = Mock(post=AsyncMock(return_value=response))
    client = InferenceClient(base_url="https://example/v1", session=session)
    result = await client.complete(model="m", messages=[], tools=[{"type": "function"}])
    assert result.tool_calls == [{"id": "c"}]
    assert result.request_id == "r1"
    assert session.post.call_args.kwargs["json"]["tools"] == [{"type": "function"}]


@pytest.mark.asyncio
async def test_http_errors_are_normalized() -> None:
    response = Mock(ok=False, status_code=429, headers={})
    response.json.return_value = {"error": {"message": "busy"}}
    session = Mock(post=AsyncMock(return_value=response))
    with pytest.raises(InferenceHTTPError) as raised:
        await InferenceClient(base_url="https://example/v1", session=session).embed(
            model="m", input=["x"]
        )
    assert raised.value.status_code == 429


@pytest.mark.asyncio
async def test_close_awaits_session_close() -> None:
    session = Mock(close=AsyncMock())
    await InferenceClient(base_url="https://example/v1", session=session).close()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_embeddings_preserve_order_sparse_extensions_and_usage() -> None:
    response = Mock(ok=True, status_code=200, headers={"request-id": "e1"})
    response.json.return_value = {
        "object": "list",
        "data": [
            {
                "index": 0,
                "embedding": [1, 2],
                "sparse_embedding": {"indices": [3], "values": [0.5]},
            },
            {"index": 1, "embedding": [4, 5]},
        ],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    session = Mock(post=AsyncMock(return_value=response))
    result = await InferenceClient(base_url="http://local/v1", session=session).embed(
        model="bge", input=["a", "b"]
    )
    assert result.embeddings == [[1.0, 2.0], [4.0, 5.0]]
    assert result.sparse_embeddings == [{"indices": [3], "values": [0.5]}, None]
    assert result.usage.prompt_tokens == 2
    assert session.post.call_args.args[0] == "http://local/v1/embeddings"


@pytest.mark.asyncio
async def test_chat_preserves_multimodal_structured_and_reasoning_fields() -> None:
    response = Mock(ok=True, status_code=200, headers={})
    response.json.return_value = {
        "choices": [
            {"message": {"content": '{"title":"x"}', "reasoning_content": "because"}}
        ]
    }
    session = Mock(post=AsyncMock(return_value=response))
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Read this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }
    client = InferenceClient(base_url="http://local/v1", session=session)
    result = await client.complete(
        model="vision",
        messages=[message],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=100,
    )
    assert result.content == '{"title":"x"}'
    assert result.reasoning == "because"
    assert session.post.call_args.kwargs["json"]["messages"] == [message]
    assert session.post.call_args.kwargs["json"]["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.asyncio
async def test_rerank_maps_indices_and_scores() -> None:
    response = Mock(ok=True, status_code=200, headers={})
    response.json.return_value = {
        "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "score": 0.2}]
    }
    session = Mock(post=AsyncMock(return_value=response))
    result = await InferenceClient(
        base_url="https://openrouter/api/v1", session=session
    ).rerank(model="cohere/rerank", query="q", documents=["a", "b"])
    assert result.indices == [1, 0]
    assert result.scores == [0.9, 0.2]
    assert session.post.call_args.args[0] == "https://openrouter/api/v1/rerank"


@pytest.mark.asyncio
async def test_timeout_is_normalized() -> None:
    session = Mock(post=AsyncMock(side_effect=niquests.Timeout("slow")))
    with pytest.raises(InferenceTimeoutError):
        await InferenceClient(base_url="http://local/v1", session=session).complete(
            model="m", messages=[]
        )


@pytest.mark.asyncio
async def test_tracing_records_each_operation_and_full_payloads() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    responses = []
    for body in (
        {"choices": [{"message": {"content": "ok"}}]},
        {"data": [{"embedding": [1.0]}]},
        {"results": [{"index": 0, "relevance_score": 0.9}]},
    ):
        response = Mock(ok=True, status_code=200, headers={})
        response.json.return_value = body
        responses.append(response)
    session = Mock(post=AsyncMock(side_effect=responses))
    client = InferenceClient(
        base_url="http://local/v1", provider="local", domain="default", session=session
    )

    await client.complete(
        model="chat-model",
        messages=[{"role": "user", "content": "hello"}],
        domain="browser_copilot",
        temperature=0,
    )
    await client.embed(
        model="embedding-model", input=["safe text"], domain="ingest_embedding"
    )
    await client.rerank(
        model="rerank-model",
        query="safe query",
        documents=["safe document"],
        domain="recall_rerank",
    )

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "llm.complete",
        "llm.embed",
        "llm.rerank",
    ]
    assert [span.attributes["llm.domain"] for span in spans] == [
        "browser_copilot",
        "ingest_embedding",
        "recall_rerank",
    ]
    assert (
        '"messages":[{"role":"user","content":"hello"}]'
        in spans[0].attributes["llm.request"]
    )
    assert '"input":["safe text"]' in spans[1].attributes["llm.request"]
    assert '"documents":["safe document"]' in spans[2].attributes["llm.request"]
    assert '"content":"ok"' in spans[0].attributes["llm.response"]
    assert spans[0].attributes["openinference.span.kind"] == "LLM"
    assert spans[0].attributes["input.mime_type"] == "application/json"
    assert spans[0].attributes["output.mime_type"] == "application/json"
    assert spans[0].attributes["llm.input_messages.0.message.role"] == "user"
    assert spans[0].attributes["llm.input_messages.0.message.content"] == "hello"
    assert spans[0].attributes["llm.output_messages.0.message.content"] == "ok"
    assert spans[1].attributes["openinference.span.kind"] == "EMBEDDING"
    assert spans[1].attributes["embedding.text.0"] == "safe text"
    assert spans[1].attributes["embedding.embeddings.0.embedding.text"] == "safe text"
    assert spans[1].attributes["embedding.embeddings.0.embedding.vector"] == (1.0,)
    assert spans[2].attributes["openinference.span.kind"] == "RERANKER"
    assert spans[2].attributes["reranker.query"] == "safe query"
    assert (
        spans[2].attributes["reranker.input_documents.0.document.content"]
        == "safe document"
    )
    assert (
        spans[2].attributes["reranker.output_documents.0.document.content"]
        == "safe document"
    )
    assert spans[2].attributes["reranker.output_documents.0.document.score"] == 0.9
