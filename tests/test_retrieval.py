from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.rag.models import EmbeddedChunk
from app.rag.retrieval import RetrievalService


def test_retrieve_embeds_question_and_searches_vector_store() -> None:
    document_id = uuid4()
    chunk = EmbeddedChunk(
        text="Revenue increased by 12%.",
        index=0,
        embedding=[0.1, 0.2, 0.3],
        document_id=document_id,
    )

    embedding_client = Mock()
    embedding_client.embed.return_value = [0.4, 0.5, 0.6]

    vector_store = Mock()
    vector_store.search.return_value = [chunk]

    service = RetrievalService(
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    results = service.retrieve("What was the revenue increase?", limit=3)

    embedding_client.embed.assert_called_once_with("What was the revenue increase?")
    vector_store.search.assert_called_once_with(
        query_embedding=[0.4, 0.5, 0.6],
        limit=3,
    )
    assert results == [chunk]


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_retrieve_rejects_empty_question(question: str) -> None:
    embedding_client = Mock()
    vector_store = Mock()

    service = RetrievalService(
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="Question must not be empty"):
        service.retrieve(question)

    embedding_client.embed.assert_not_called()
    vector_store.search.assert_not_called()


def test_retrieve_rejects_non_positive_limit() -> None:
    embedding_client = Mock()
    vector_store = Mock()

    service = RetrievalService(
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        service.retrieve("What was the revenue increase?", limit=0)

    embedding_client.embed.assert_not_called()
    vector_store.search.assert_not_called()
