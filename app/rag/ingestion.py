import logging
from uuid import uuid4

from app.core.exceptions import DuplicateDocumentError
from app.rag.checksum import calculate_checksum
from app.rag.document_models import Document
from app.rag.document_source import DocumentSource
from app.rag.document_storage import DocumentStorage
from app.rag.pdf_validation import validate_pdf
from app.repositories.document import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(
        self,
        storage: DocumentStorage,
        repository: DocumentRepository,
    ) -> None:
        self.storage = storage
        self.repository = repository

    async def ingest(self, source: DocumentSource) -> Document:
        validate_pdf(source.content, source.filename)
        checksum = calculate_checksum(source.content)

        existing = await self.repository.get_by_checksum(checksum)
        if existing is not None:
            raise DuplicateDocumentError("This document has already been ingested.")

        document_id = uuid4()
        storage_key = f"documents/{document_id}.pdf"

        self.storage.save(source.content, storage_key)

        document = Document(
            document_id=document_id,
            filename=source.filename,
            storage_key=storage_key,
            checksum=checksum,
        )

        try:
            return await self.repository.create(document)
        except Exception:
            try:
                self.storage.delete(storage_key)
            except Exception:
                logger.exception(
                    "Failed to clean up stored document %s after ingestion failure",
                    storage_key,
                )
            raise
