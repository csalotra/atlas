from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSource:
    filename: str
    content: bytes
