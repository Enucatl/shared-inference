from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


@contextmanager
def llm_span(
    operation: str, *, domain: str, provider: str, model: str
) -> Iterator[Any]:
    """Create a privacy-safe LLM span; the API is a no-op without an SDK/provider."""
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
        }.items():
            span.set_attribute(key, value)
        try:
            yield span
        finally:
            span.set_attribute("llm.latency_ms", (perf_counter() - started) * 1000)


def record_result(span: Any, *, usage: Any, request_id: str | None) -> None:
    if span is None:
        return
    for key, value in {
        "llm.request_id": request_id,
        "llm.usage.prompt_tokens": usage.prompt_tokens,
        "llm.usage.completion_tokens": usage.completion_tokens,
        "llm.usage.total_tokens": usage.total_tokens,
    }.items():
        if value is not None:
            span.set_attribute(key, value)
