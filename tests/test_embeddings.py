from unittest.mock import Mock

from openai.types.create_embedding_response import CreateEmbeddingResponse

from app.rag.embeddings import EmbeddingClient


def test_embed_returns_embedding(monkeypatch):
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

    mock_create = Mock(return_value=response)

    monkeypatch.setattr(
        client.client.embeddings,
        "create",
        mock_create,
    )

    result = client.embed("hello world")

    assert result == [0.1, 0.2, 0.3]

    mock_create.assert_called_once_with(
        model="atlas-embedding",
        input="hello world",
    )
