from uuid import uuid5, NAMESPACE_URL, UUID

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.rag.models import EmbeddedChunk
from app.rag.vector_store import VectorStore
from app.core.exceptions import VectorStoreError


class QdrantVectorStore(VectorStore):
    def __init__(self, client: QdrantClient) -> None:
        settings = get_settings()
        self.client = client
        self.collection = settings.qdrant_collection
        self.dimensions = settings.embedding_dimensions

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )

    def add(self, chunks: list[EmbeddedChunk]) -> None:
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
                },
            )
            for chunk in chunks
        ]

        for start in range(0, len(points), 128):
            batch = points[start : start + 128]
            try:
                self.client.upsert(
                    collection_name=self.collection,
                    points=batch,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to store batch starting at point {start}."
                ) from exc

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[EmbeddedChunk]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            ).points
        except Exception as exc:
            raise VectorStoreError("Failed to search the vector store.") from exc

        return [
            EmbeddedChunk(
                text=point.payload["text"],
                index=point.payload["index"],
                embedding=point.vector,
                document_id=UUID(point.payload["document_id"]),
            )
            for point in results
        ]
