from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]: ...


class RecursiveChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []

        texts = self.splitter.split_text(text)

        return [Chunk(text=chunk, index=index) for index, chunk in enumerate(texts)]
