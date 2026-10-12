from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedChunk:
    text: str
    index: int
    embedding: list[float]
