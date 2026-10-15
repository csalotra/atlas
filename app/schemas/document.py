from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.rag.document_models import DocumentStatus


class DocumentResponse(BaseModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    created_at: datetime | None
