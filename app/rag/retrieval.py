from app.rag.embeddings import EmbeddingClient
from app.rag.models import EmbeddedChunk
from app.rag.vector_store import VectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        limit: int = 5,
    ) -> list[EmbeddedChunk]:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        query_embedding = self.embedding_client.embed(question)

        return self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
        )
