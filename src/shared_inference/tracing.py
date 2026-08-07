import json
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


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
        }.items():
            span.set_attribute(key, value)
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
    span: Any, *, usage: Any, request_id: str | None, response: Any = None
) -> None:
    if span is None:
        return
    for key, value in {
        "llm.request_id": request_id,
        "llm.usage.prompt_tokens": usage.prompt_tokens,
        "llm.usage.completion_tokens": usage.completion_tokens,
        "llm.usage.total_tokens": usage.total_tokens,
        "llm.response": _serialize(response),
    }.items():
        if value is not None:
            span.set_attribute(key, value)
