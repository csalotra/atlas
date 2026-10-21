from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

from app.core.exceptions import DuplicateDocumentError
from app.rag.document_models import Document, DocumentStatus
from app.rag.document_source import DocumentSource
from app.rag.ingestion import DocumentIngestionService


@pytest.fixture
def document_source() -> DocumentSource:
    return DocumentSource(
        filename="report.pdf",
        content=b"%PDF-1.7\nsample content",
    )


@pytest.mark.asyncio
async def test_ingest_pdf(document_source: DocumentSource) -> None:
    storage = Mock()
    repository = Mock()
    repository.create = AsyncMock(side_effect=lambda document: document)
    repository.get_by_checksum = AsyncMock(return_value=None)

    service = DocumentIngestionService(storage=storage, repository=repository)

    result = await service.ingest(document_source)

    assert isinstance(result.document_id, UUID)
    assert result.filename == "report.pdf"
    assert result.checksum
    assert result.status == DocumentStatus.PENDING

    storage.save.assert_called_once_with(
        document_source.content,
        result.storage_key,
    )
    repository.create.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_ingest_deletes_stored_file_when_repository_fails(
    document_source: DocumentSource,
) -> None:
    storage = Mock()
    repository = Mock()
    repository.create = AsyncMock(side_effect=RuntimeError("Database error"))
    repository.get_by_checksum = AsyncMock(return_value=None)

    service = DocumentIngestionService(storage=storage, repository=repository)

    with pytest.raises(RuntimeError, match="Database error"):
        await service.ingest(document_source)

    storage.save.assert_called_once()
    storage.delete.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_rejects_duplicate_content(
    document_source: DocumentSource,
) -> None:
    existing = Document(
        document_id=UUID(int=1),
        filename="existing_report.pdf",
        storage_key="documents/existing.pdf",
        checksum="unused",
    )

    storage = Mock()
    repository = Mock()
    repository.get_by_checksum = AsyncMock(return_value=existing)
    repository.create = AsyncMock()

    service = DocumentIngestionService(storage=storage, repository=repository)

    with pytest.raises(
        DuplicateDocumentError,
        match="This document has already been ingested.",
    ):
        await service.ingest(document_source)

    storage.save.assert_not_called()
    repository.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_preserves_repository_error_when_cleanup_fails(
    document_source: DocumentSource,
    caplog: pytest.LogCaptureFixture,
) -> None:
    storage = Mock()
    storage.delete.side_effect = RuntimeError("Cleanup error")

    repository = Mock()
    repository.create = AsyncMock(side_effect=RuntimeError("Database error"))
    repository.get_by_checksum = AsyncMock(return_value=None)

    service = DocumentIngestionService(storage=storage, repository=repository)

    with pytest.raises(RuntimeError, match="Database error"):
        await service.ingest(document_source)

    storage.delete.assert_called_once()
    assert "Failed to clean up stored document" in caplog.text
    assert "Cleanup error" in caplog.text
