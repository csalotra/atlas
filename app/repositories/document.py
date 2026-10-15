from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document as DocumentRecord
from app.rag.document_models import Document, DocumentStatus

from uuid import UUID
from sqlalchemy import select

from sqlalchemy.exc import IntegrityError
from app.core.exceptions import DocumentAlreadyExistsError


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @staticmethod
    def _to_domain(record: DocumentRecord) -> Document:
        return Document(
            document_id=record.document_id,
            filename=record.filename,
            storage_key=record.storage_key,
            checksum=record.checksum,
            status=DocumentStatus(record.status),
            version=record.version,
            created_at=record.created_at,
        )

    async def create(self, document: Document) -> Document:
        record = DocumentRecord(
            document_id=document.document_id,
            filename=document.filename,
            storage_key=document.storage_key,
            checksum=document.checksum,
            status=document.status,
            version=document.version,
            created_at=document.created_at,
        )

        self.session.add(record)

        try:
            await self.session.flush()
        except IntegrityError as exc:
            await self.session.rollback()
            raise DocumentAlreadyExistsError(
                f"Document with ID {document.document_id} already exists."
            ) from exc

        return self._to_domain(record)

    async def get(self, document_id: UUID) -> Document | None:
        result = await self.session.execute(
            select(DocumentRecord).where(DocumentRecord.document_id == document_id)
        )
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return self._to_domain(record)

    async def get_by_checksum(self, checksum: str) -> Document | None:
        result = await self.session.execute(
            select(DocumentRecord).where(DocumentRecord.checksum == checksum)
        )
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return self._to_domain(record)

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
    ) -> Document | None:
        document = await self.session.get(DocumentRecord, document_id)

        if document is None:
            return None

        document.status = status
        await self.session.flush()

        return self._to_domain(document)
