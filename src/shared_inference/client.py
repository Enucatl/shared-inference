from typing import Any

import niquests

from .errors import InferenceError, InferenceHTTPError, InferenceTimeoutError
from .models import CompletionResult, EmbeddingResult, RerankResult, Usage
from .tracing import llm_span, record_result


def _normalize_model(model: str, provider: str) -> str:
    """Remove the legacy local ``openai/`` transport prefix only."""
    if provider != "openrouter" and model.startswith("openai/"):
        return model.removeprefix("openai/")
    return model


class InferenceClient:
    """One long-lived async client for OpenAI-compatible inference endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        provider: str = "openai-compatible",
        timeout: float = 120,
        domain: str = "inference",
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout
        self.domain = domain
        self.session = session or niquests.AsyncSession(timeout=timeout)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        domain: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        model = _normalize_model(model, self.provider)
        return await self._complete(
            "/chat/completions",
            model=model,
            messages=messages,
            domain=domain,
            **kwargs,
        )

    async def embed(
        self, *, model: str, input: list[str], domain: str | None = None, **kwargs: Any
    ) -> EmbeddingResult:
        model = _normalize_model(model, self.provider)
        payload = {"model": model, "input": input, **kwargs}
        with llm_span(
            "llm.embed",
            domain=domain or self.domain,
            provider=self.provider,
            model=model,
            request=payload,
        ) as span:
            raw, request_id = await self._post("/embeddings", payload)
            usage = _usage(raw.get("usage"))
            result = EmbeddingResult(
                embeddings=[
                    list(map(float, item.get("embedding", [])))
                    for item in raw.get("data", [])
                ],
                sparse_embeddings=[
                    item.get("sparse_embedding") for item in raw.get("data", [])
                ],
                usage=usage,
                request_id=request_id,
                raw=raw,
            )
            record_result(
                span,
                usage=usage,
                request_id=request_id,
                response=raw,
                request=payload,
            )
            return result

    async def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        domain: str | None = None,
        **kwargs: Any,
    ) -> RerankResult:
        model = _normalize_model(model, self.provider)
        payload = {"model": model, "query": query, "documents": documents, **kwargs}
        with llm_span(
            "llm.rerank",
            domain=domain or self.domain,
            provider=self.provider,
            model=model,
            request=payload,
        ) as span:
            raw, request_id = await self._post("/rerank", payload)
            results = raw.get("results", [])
            usage = _usage(raw.get("usage"))
            result = RerankResult(
                indices=[int(item.get("index", -1)) for item in results],
                scores=[
                    float(item.get("relevance_score", item.get("score", 0)))
                    for item in results
                ],
                usage=usage,
                request_id=request_id,
                raw=raw,
            )
            record_result(
                span,
                usage=usage,
                request_id=request_id,
                response=raw,
                request=payload,
            )
            return result

    async def _complete(
        self,
        path: str,
        *,
        model: str,
        messages: list[dict[str, Any]],
        domain: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        payload = {"model": model, "messages": messages, **kwargs}
        with llm_span(
            "llm.complete",
            domain=domain or self.domain,
            provider=self.provider,
            model=model,
            request=payload,
        ) as span:
            raw, request_id = await self._post(path, payload)
            choice = (raw.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = _usage(raw.get("usage"))
            result = CompletionResult(
                content=message.get("content"),
                message=message,
                tool_calls=message.get("tool_calls") or [],
                reasoning=message.get("reasoning_content") or message.get("reasoning"),
                usage=usage,
                request_id=request_id,
                raw=raw,
            )
            record_result(span, usage=usage, request_id=request_id, response=raw)
            return result

    async def _post(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = await self.session.post(
                f"{self.base_url}{path}", json=payload, headers=headers
            )
        except (niquests.Timeout, TimeoutError) as exc:
            raise InferenceTimeoutError("Inference request timed out") from exc
        except niquests.RequestException as exc:
            raise InferenceError(f"Inference request failed: {exc}") from exc
        try:
            raw = response.json()
        except ValueError as exc:
            raw = {"error": response.text}
            if not response.ok:
                raise InferenceHTTPError(
                    "Provider returned invalid JSON",
                    status_code=response.status_code,
                    raw=raw,
                ) from exc
            raise InferenceError("Provider returned invalid JSON", raw=raw) from exc
        if not response.ok:
            raise InferenceHTTPError(
                str(raw.get("error", "Inference provider returned an error")),
                status_code=response.status_code,
                raw=raw,
            )
        return raw, response.headers.get("x-request-id") or response.headers.get(
            "request-id"
        )

    async def close(self) -> None:
        await self.session.close()

    async def get(self, url: str, **kwargs: Any) -> Any:
        """Perform a health-check GET through the shared session."""
        return await self.session.get(url, **kwargs)


def _usage(value: dict[str, Any] | None) -> Usage:
    value = value or {}
    return Usage(
        value.get("prompt_tokens"),
        value.get("completion_tokens"),
        value.get("total_tokens"),
        value,
    )
