import os

import pytest

from shared_inference import InferenceClient


def _enabled() -> bool:
    return os.environ.get("SHARED_INFERENCE_LIVE") == "1"


@pytest.mark.live
@pytest.mark.asyncio
async def test_local_embeddings_live() -> None:
    if not _enabled():
        pytest.skip("set SHARED_INFERENCE_LIVE=1 to run live tests")
    client = InferenceClient(
        base_url=os.environ.get(
            "SHARED_INFERENCE_EMBEDDING_URL", "http://complex.home.arpa:8102/v1"
        ),
        provider="local",
        domain="live_embedding",
        timeout=30,
    )
    try:
        result = await client.embed(
            model=os.environ.get("SHARED_INFERENCE_EMBEDDING_MODEL", "BAAI/bge-m3"),
            input=["live test one", "live test two"],
        )
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) > 0
    finally:
        await client.close()


@pytest.mark.live
@pytest.mark.asyncio
async def test_local_ocr_live() -> None:
    if not _enabled():
        pytest.skip("set SHARED_INFERENCE_LIVE=1 to run live tests")
    client = InferenceClient(
        base_url=os.environ.get(
            "SHARED_INFERENCE_CHAT_URL", "http://complex.home.arpa:8100/v1"
        ),
        provider="local",
        domain="live_ocr",
        timeout=60,
    )
    try:
        result = await client.complete(
            model=os.environ.get(
                "SHARED_INFERENCE_CHAT_MODEL", "nanonets/Nanonets-OCR2-3B"
            ),
            messages=[{"role": "user", "content": "Reply with exactly OCR_OK"}],
            temperature=0,
            max_tokens=8,
        )
        assert result.content and "OCR_OK" in result.content
    finally:
        await client.close()
