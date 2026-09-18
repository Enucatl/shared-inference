import json
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _message_content(value: Any) -> str:
    return value if isinstance(value, str) else _serialize(value)


def _set_message_attributes(span: Any, prefix: str, messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for field in ("role", "name"):
            value = message.get(field)
            if value is not None:
                span.set_attribute(
                    f"llm.{prefix}_messages.{index}.message.{field}", value
                )
        if message.get("content") is not None:
            span.set_attribute(
                f"llm.{prefix}_messages.{index}.message.content",
                _message_content(message["content"]),
            )
        if message.get("tool_calls"):
            span.set_attribute(
                f"llm.{prefix}_messages.{index}.message.tool_calls",
                _serialize(message["tool_calls"]),
            )


def _document_attributes(documents: Any) -> list[dict[str, Any]]:
    if not isinstance(documents, list):
        return []
    normalized = []
    for index, document in enumerate(documents):
        if isinstance(document, dict):
            value = dict(document)
            if "content" in value:
                value["document.content"] = value.pop("content")
            if "id" in value:
                value["document.id"] = value.pop("id")
            if "score" in value:
                value["document.score"] = value.pop("score")
            normalized.append(value)
        else:
            normalized.append({"document.id": index, "document.content": document})
    return normalized


def _set_document_attributes(span: Any, prefix: str, documents: Any) -> None:
    for index, document in enumerate(_document_attributes(documents)):
        for field, value in document.items():
            if value is not None:
                span.set_attribute(f"{prefix}.{index}.{field}", value)


@contextmanager
def llm_span(
    operation: str,
    *,
    domain: str,
    provider: str,
    model: str,
    request: Any = None,
) -> Iterator[Any]:
    """Create a full-fidelity LLM span when an SDK/provider is configured."""
    try:
        from opentelemetry import trace

        span_context = trace.get_tracer("shared_inference").start_as_current_span(
            operation
        )
    except ImportError:
        yield None
        return
    with span_context as span:
        started = perf_counter()
        for key, value in {
            "llm.domain": domain,
            "llm.provider": provider,
            "llm.model": model,
            "llm.operation": operation.removeprefix("llm."),
            "llm.request": _serialize(request),
            "openinference.span.kind": {
                "llm.complete": "LLM",
                "llm.embed": "EMBEDDING",
                "llm.rerank": "RERANKER",
            }.get(operation, "LLM"),
            "input.value": _serialize(request),
            "input.mime_type": "application/json",
            "llm.model_name": model,
        }.items():
            span.set_attribute(key, value)
        if operation == "llm.complete" and isinstance(request, dict):
            _set_message_attributes(span, "input", request.get("messages"))
            invocation_parameters = {
                key: value
                for key, value in request.items()
                if key not in {"model", "messages"}
            }
            if invocation_parameters:
                span.set_attribute(
                    "llm.invocation_parameters", _serialize(invocation_parameters)
                )
        elif operation == "llm.embed" and isinstance(request, dict):
            for index, value in enumerate(request.get("input", [])):
                span.set_attribute(f"embedding.text.{index}", _message_content(value))
            span.set_attribute("embedding.model_name", model)
        elif operation == "llm.rerank" and isinstance(request, dict):
            span.set_attribute("reranker.model_name", model)
            if request.get("query") is not None:
                span.set_attribute("reranker.query", request["query"])
            _set_document_attributes(
                span, "reranker.input_documents", request.get("documents", [])
            )
        try:
            yield span
        except Exception as exc:
            span.set_attribute("llm.error", str(exc))
            span.record_exception(exc)
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.set_attribute("llm.latency_ms", (perf_counter() - started) * 1000)


def record_result(
    span: Any,
    *,
    usage: Any,
    request_id: str | None,
    response: Any = None,
    request: Any = None,
) -> None:
    if span is None:
        return
    usage_raw = getattr(usage, "raw", None)
    cost = usage_raw.get("cost") if isinstance(usage_raw, dict) else None
    for key, value in {
        "llm.request_id": request_id,
        "llm.usage.prompt_tokens": usage.prompt_tokens,
        "llm.usage.completion_tokens": usage.completion_tokens,
        "llm.usage.total_tokens": usage.total_tokens,
        "llm.usage.cost": cost,
        "llm.token_count.prompt": usage.prompt_tokens,
        "llm.token_count.completion": usage.completion_tokens,
        "llm.token_count.total": usage.total_tokens,
        "llm.response": _serialize(response),
        "output.value": _serialize(response),
        "output.mime_type": "application/json",
    }.items():
        if value is not None:
            span.set_attribute(key, value)
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = (
                (choices[0].get("message") or {})
                if isinstance(choices[0], dict)
                else {}
            )
            _set_message_attributes(span, "output", [message])
        data = response.get("data") or []
        if data:
            inputs = request.get("input", []) if isinstance(request, dict) else []
            if not isinstance(inputs, list):
                inputs = [inputs]
            for index, item in enumerate(data):
                vector = item.get("embedding") if isinstance(item, dict) else item
                text = inputs[index] if index < len(inputs) else None
                if text is not None:
                    span.set_attribute(
                        f"embedding.embeddings.{index}.embedding.text", text
                    )
                if vector is not None:
                    span.set_attribute(
                        f"embedding.embeddings.{index}.embedding.vector", vector
                    )
        if response.get("results") is not None:
            results = response["results"]
            documents = (
                request.get("documents", []) if isinstance(request, dict) else []
            )
            output_documents = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                index = result.get("index")
                document = (
                    documents[index]
                    if isinstance(index, int) and index < len(documents)
                    else None
                )
                output = {
                    "document.id": index,
                    "document.score": result.get(
                        "relevance_score", result.get("score")
                    ),
                }
                if document is not None:
                    output["document.content"] = document
                output_documents.append(output)
            _set_document_attributes(
                span, "reranker.output_documents", output_documents
            )
