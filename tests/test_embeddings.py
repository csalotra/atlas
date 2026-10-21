from unittest.mock import AsyncMock

import pytest
from openai.types.create_embedding_response import CreateEmbeddingResponse

from app.rag.embeddings import EmbeddingClient


@pytest.mark.asyncio
async def test_embed_returns_embedding(monkeypatch) -> None:
    client = EmbeddingClient()

    response = CreateEmbeddingResponse(
        object="list",
        data=[
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        model="atlas-embedding",
        usage={
            "prompt_tokens": 2,
            "total_tokens": 2,
        },
    )

    mock_create = AsyncMock(return_value=response)

    monkeypatch.setattr(
        client.client.embeddings,
        "create",
        mock_create,
    )

    result = await client.embed("hello world")

    assert result == [0.1, 0.2, 0.3]

    mock_create.assert_awaited_once_with(
        model="atlas-embedding",
        input="hello world",
    )
