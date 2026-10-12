from abc import ABC, abstractmethod

from app.rag.models import EmbeddedChunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[EmbeddedChunk]: ...
