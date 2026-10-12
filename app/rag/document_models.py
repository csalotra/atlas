from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import uuid


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True)
class Document:
    document_id: uuid.UUID
    filename: str
    storage_key: str
    checksum: str
    status: DocumentStatus = DocumentStatus.PENDING
    version: int = 1
    created_at: datetime | None = None
