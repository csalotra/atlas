from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import AsyncQdrantClient, models

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.rag.models import EmbeddedChunk
from app.rag.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client
        self.collection = collection or settings.qdrant_collection
        self.dimensions = settings.embedding_dimensions

    async def ensure_collection(self) -> None:
        try:
            if not await self.client.collection_exists(self.collection):
                await self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize the vector store collection."
            ) from exc

    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{chunk.document_id}:{chunk.index}",
                    )
                ),
                vector=chunk.embedding,
                payload={
                    "text": chunk.text,
                    "index": chunk.index,
                    "document_id": str(chunk.document_id),
                    "page_number": chunk.page_number,
                },
            )
            for chunk in chunks
        ]

        for start in range(0, len(points), 128):
            batch = points[start : start + 128]
            try:
                await self.client.upsert(
                    collection_name=self.collection,
                    points=batch,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to store batch starting at point {start}."
                ) from exc

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[EmbeddedChunk]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        try:
            results = await self.client.query_points(
                collection_name=self.collection,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise VectorStoreError("Failed to search the vector store.") from exc

        return [
            EmbeddedChunk(
                text=point.payload["text"],
                index=point.payload["index"],
                embedding=point.vector,
                document_id=UUID(point.payload["document_id"]),
                page_number=point.payload["page_number"],
            )
            for point in results.points
        ]
