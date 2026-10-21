from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.rag.models import EmbeddedChunk
from app.rag.qdrant_vector_store import QdrantVectorStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_store_and_retrieve_chunk() -> None:
    settings = get_settings()
    collection = f"atlas_test_{uuid4().hex}"

    client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        await client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

        # Use the real client, but point the adapter at our isolated collection.
        store = QdrantVectorStore(client=client, collection=collection)
        await store.ensure_collection()

        document_id = uuid4()
        chunk = EmbeddedChunk(
            text="Revenue increased by 12%.",
            index=0,
            embedding=[1.0] + [0.0] * (settings.embedding_dimensions - 1),
            document_id=document_id,
            page_number=1,
        )

        await store.add([chunk])
        results = await store.search(chunk.embedding, limit=1)

        assert len(results) == 1
        assert results[0].text == chunk.text
        assert results[0].index == chunk.index
        assert results[0].document_id == document_id
        assert results[0].embedding == chunk.embedding
        assert results[0].page_number == chunk.page_number

    finally:
        await client.delete_collection(collection_name=collection)
        await client.close()
