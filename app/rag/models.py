from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class EmbeddedChunk:
    text: str
    index: int
    embedding: list[float]
    document_id: UUID
