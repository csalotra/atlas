from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.rag.document_models import Document, DocumentStatus
from app.rag.indexing import DocumentIndexingService


@pytest.mark.asyncio
async def test_index_document() -> None:
    document_id = uuid4()
    document = Document(
        document_id=document_id,
        filename="report.pdf",
        storage_key=f"documents/{document_id}.pdf",
        checksum="a" * 64,
    )

    storage = Mock()
    storage.read.return_value = b"pdf bytes"

    repository = Mock()
    repository.get = AsyncMock(return_value=document)
    repository.update_status = AsyncMock()

    extractor = Mock()
    extractor.extract_text.return_value = "Revenue increased by 12%."

    pipeline = Mock()
    pipeline.embed_text.return_value = ["embedded chunk"]

    vector_store = Mock()

    service = DocumentIndexingService(
        storage=storage,
        repository=repository,
        extractor=extractor,
        pipeline=pipeline,
        vector_store=vector_store,
    )

    await service.index(document_id)

    storage.read.assert_called_once_with(document.storage_key)
    extractor.extract_text.assert_called_once_with(b"pdf bytes")
    pipeline.embed_text.assert_called_once_with(
        "Revenue increased by 12%.",
        document_id,
    )
    vector_store.add.assert_called_once_with(["embedded chunk"])
    assert repository.update_status.await_args_list[0].args == (
        document_id,
        DocumentStatus.PROCESSING,
    )
    assert repository.update_status.await_args_list[1].args == (
        document_id,
        DocumentStatus.INDEXED,
    )


@pytest.mark.asyncio
async def test_index_marks_document_failed_when_extraction_fails() -> None:
    document_id = uuid4()
    document = Document(
        document_id=document_id,
        filename="report.pdf",
        storage_key=f"documents/{document_id}.pdf",
        checksum="a" * 64,
    )

    storage = Mock()
    storage.read.return_value = b"pdf bytes"

    repository = Mock()
    repository.get = AsyncMock(return_value=document)
    repository.update_status = AsyncMock()

    extractor = Mock()
    extractor.extract_text.side_effect = ValueError("Invalid PDF")

    service = DocumentIndexingService(
        storage=storage,
        repository=repository,
        extractor=extractor,
        pipeline=Mock(),
        vector_store=Mock(),
    )

    with pytest.raises(ValueError, match="Invalid PDF"):
        await service.index(document_id)

    assert repository.update_status.await_args_list[-1].args == (
        document_id,
        DocumentStatus.FAILED,
    )


@pytest.mark.asyncio
async def test_index_marks_document_failed_when_no_chunks_generated() -> None:
    document_id = uuid4()
    document = Document(
        document_id=document_id,
        filename="report.pdf",
        storage_key=f"documents/{document_id}.pdf",
        checksum="a" * 64,
    )

    repository = Mock()
    repository.get = AsyncMock(return_value=document)
    repository.update_status = AsyncMock()

    extractor = Mock()
    extractor.extract_text.return_value = "Some extracted text"

    pipeline = Mock()
    pipeline.embed_text.return_value = []

    service = DocumentIndexingService(
        storage=Mock(),
        repository=repository,
        extractor=extractor,
        pipeline=pipeline,
        vector_store=Mock(),
    )

    with pytest.raises(ValueError, match="No chunks were generated"):
        await service.index(document_id)

    assert repository.update_status.await_args_list[-1].args == (
        document_id,
        DocumentStatus.FAILED,
    )
