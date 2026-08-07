from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompletionResult:
    content: str | None
    message: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reasoning: str | None
    usage: Usage
    request_id: str | None
    raw: dict[str, Any]


@dataclass(slots=True)
class EmbeddingResult:
    embeddings: list[list[float]]
    usage: Usage
    request_id: str | None
    raw: dict[str, Any]
    sparse_embeddings: list[dict[str, Any] | None] = field(default_factory=list)


@dataclass(slots=True)
class RerankResult:
    indices: list[int]
    scores: list[float]
    usage: Usage
    request_id: str | None
    raw: dict[str, Any]
