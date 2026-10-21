from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.rag.document_models import Document, DocumentStatus
from app.rag.indexing import DocumentIndexingService
from app.rag.models import EmbeddedChunk
from app.rag.pdf_extractor import ExtractedPage
from app.core.exceptions import InvalidDocumentError


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

    pages = [
        ExtractedPage(
            page_number=1,
            text="Revenue increased by 12%.",
        ),
        ExtractedPage(
            page_number=2,
            text="Operating costs decreased by 5%.",
        ),
    ]

    extractor = Mock()
    extractor.extract_pages.return_value = pages

    embedded_chunks = [
        EmbeddedChunk(
            text="Revenue increased by 12%.",
            index=0,
            embedding=[0.1, 0.2],
            document_id=document_id,
            page_number=1,
        ),
        EmbeddedChunk(
            text="Operating costs decreased by 5%.",
            index=1,
            embedding=[0.3, 0.4],
            document_id=document_id,
            page_number=2,
        ),
    ]

    pipeline = Mock()
    pipeline.embed_pages = AsyncMock(return_value=embedded_chunks)

    vector_store = Mock()
    vector_store.add = AsyncMock()

    service = DocumentIndexingService(
        storage=storage,
        repository=repository,
        extractor=extractor,
        pipeline=pipeline,
        vector_store=vector_store,
    )

    await service.index(document_id)

    storage.read.assert_called_once_with(document.storage_key)
    extractor.extract_pages.assert_called_once_with(b"pdf bytes")
    pipeline.embed_pages.assert_awaited_once_with(pages, document_id)
    vector_store.add.assert_awaited_once_with(embedded_chunks)

    assert embedded_chunks[0].page_number == 1
    assert embedded_chunks[1].page_number == 2

    assert repository.update_status.await_args_list[0].args == (
        document_id,
        DocumentStatus.PROCESSING,
    )
    assert repository.update_status.await_args_list[1].args == (
        document_id,
        DocumentStatus.INDEXED,
    )


@pytest.mark.asyncio
async def test_index_propagates_extraction_failure() -> None:
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
    extractor.extract_pages.side_effect = ValueError("Invalid PDF")

    service = DocumentIndexingService(
        storage=storage,
        repository=repository,
        extractor=extractor,
        pipeline=Mock(),
        vector_store=Mock(),
    )

    with pytest.raises(ValueError, match="Invalid PDF"):
        await service.index(document_id)

    repository.update_status.assert_awaited_once_with(
        document_id,
        DocumentStatus.PROCESSING,
    )


@pytest.mark.asyncio
async def test_index_propagates_when_no_chunks_generated() -> None:
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
    extractor.extract_pages.return_value = [
        ExtractedPage(
            page_number=1,
            text="Some extracted text",
        )
    ]

    pipeline = Mock()
    pipeline.embed_pages = AsyncMock(return_value=[])

    service = DocumentIndexingService(
        storage=Mock(),
        repository=repository,
        extractor=extractor,
        pipeline=pipeline,
        vector_store=Mock(),
    )

    with pytest.raises(
        InvalidDocumentError, match="No chunks were generated from the PDF."
    ):
        await service.index(document_id)

    repository.update_status.assert_awaited_once_with(
        document_id,
        DocumentStatus.PROCESSING,
    )


@pytest.mark.asyncio
async def test_index_propagates_when_pdf_has_no_extractable_text() -> None:
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
    extractor.extract_pages.return_value = [
        ExtractedPage(page_number=1, text="  "),
        ExtractedPage(page_number=2, text=""),
    ]

    pipeline = Mock()
    pipeline.embed_pages = AsyncMock()

    service = DocumentIndexingService(
        storage=Mock(),
        repository=repository,
        extractor=extractor,
        pipeline=pipeline,
        vector_store=Mock(),
    )

    with pytest.raises(InvalidDocumentError, match="No extractable text found in PDF."):
        await service.index(document_id)

    pipeline.embed_pages.assert_not_awaited()

    repository.update_status.assert_awaited_once_with(
        document_id,
        DocumentStatus.PROCESSING,
    )
