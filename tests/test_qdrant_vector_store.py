from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from qdrant_client import models

from app.core.exceptions import VectorStoreError
from app.rag.models import EmbeddedChunk
from app.rag.qdrant_vector_store import QdrantVectorStore


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def store(client: MagicMock) -> QdrantVectorStore:
    return QdrantVectorStore(client=client)


@pytest.fixture
def embedded_chunk() -> EmbeddedChunk:
    return EmbeddedChunk(
        text="Revenue increased by 12%.",
        index=0,
        embedding=[0.1, 0.2, 0.3],
        document_id=uuid4(),
    )


def test_creates_collection_when_missing(client: MagicMock) -> None:
    client.collection_exists.return_value = False

    QdrantVectorStore(client=client)

    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs

    assert kwargs["vectors_config"].distance == models.Distance.COSINE


def test_does_not_create_existing_collection(client: MagicMock) -> None:
    QdrantVectorStore(client=client)

    client.create_collection.assert_not_called()


def test_add_upserts_chunk(
    store: QdrantVectorStore,
    client: MagicMock,
    embedded_chunk: EmbeddedChunk,
) -> None:
    store.add([embedded_chunk])

    client.upsert.assert_called_once()
    kwargs = client.upsert.call_args.kwargs

    assert kwargs["collection_name"] == store.collection

    point = kwargs["points"][0]
    assert point.vector == embedded_chunk.embedding
    assert point.payload["text"] == embedded_chunk.text
    assert point.payload["index"] == embedded_chunk.index
    assert point.payload["document_id"] == str(embedded_chunk.document_id)


def test_add_empty_list_does_not_upsert(
    store: QdrantVectorStore,
    client: MagicMock,
) -> None:
    store.add([])

    client.upsert.assert_not_called()


def test_add_wraps_qdrant_error(
    store: QdrantVectorStore,
    client: MagicMock,
    embedded_chunk: EmbeddedChunk,
) -> None:
    client.upsert.side_effect = RuntimeError("Qdrant unavailable")

    with pytest.raises(VectorStoreError) as exc_info:
        store.add([embedded_chunk])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_search_rejects_non_positive_limit(
    store: QdrantVectorStore,
) -> None:
    with pytest.raises(ValueError, match="limit must be greater than 0"):
        store.search([0.1, 0.2, 0.3], limit=0)


def test_search_returns_embedded_chunks(
    store: QdrantVectorStore,
    client: MagicMock,
    embedded_chunk: EmbeddedChunk,
) -> None:
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id="point-1",
            version=1,
            score=0.95,
            payload={
                "text": embedded_chunk.text,
                "index": embedded_chunk.index,
                "document_id": str(embedded_chunk.document_id),
            },
            vector=embedded_chunk.embedding,
        )
    ]

    results = store.search([0.1, 0.2, 0.3], limit=3)

    assert len(results) == 1
    assert results[0].text == embedded_chunk.text
    assert results[0].index == embedded_chunk.index
    assert results[0].embedding == embedded_chunk.embedding
    assert results[0].document_id == embedded_chunk.document_id

    client.query_points.assert_called_once_with(
        collection_name=store.collection,
        query=[0.1, 0.2, 0.3],
        limit=3,
        with_payload=True,
        with_vectors=True,
    )


def test_search_wraps_qdrant_error(
    store: QdrantVectorStore,
    client: MagicMock,
) -> None:
    client.query_points.side_effect = RuntimeError("Qdrant unavailable")

    with pytest.raises(VectorStoreError) as exc_info:
        store.search([0.1, 0.2, 0.3])

    assert isinstance(exc_info.value.__cause__, RuntimeError)
