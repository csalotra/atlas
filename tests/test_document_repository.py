from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.rag.document_models import Document, DocumentStatus
from app.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_create_and_get_document(db_session: AsyncSession) -> None:
    repository = DocumentRepository(db_session)
    document_id = uuid4()

    document = Document(
        document_id=document_id,
        filename="report.pdf",
        storage_key=f"documents/{document_id}.pdf",
        checksum="a" * 64,
    )

    created = await repository.create(document)

    assert created.document_id == document_id
    assert created.filename == "report.pdf"
    assert created.status == DocumentStatus.PENDING
    assert created.created_at is not None

    retrieved = await repository.get(document_id)

    assert retrieved == created


@pytest.mark.asyncio
async def test_get_missing_document(db_session: AsyncSession) -> None:
    repository = DocumentRepository(db_session)

    result = await repository.get(uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_create_document_with_duplicate_storage_key(
    db_session: AsyncSession,
) -> None:
    repository = DocumentRepository(db_session)
    storage_key = "documents/report.pdf"

    first = Document(
        document_id=uuid4(),
        filename="report.pdf",
        storage_key=storage_key,
        checksum="a" * 64,
    )
    second = Document(
        document_id=uuid4(),
        filename="another_report.pdf",
        storage_key=storage_key,
        checksum="b" * 64,
    )

    await repository.create(first)

    with pytest.raises(IntegrityError):
        await repository.create(second)


@pytest.mark.asyncio
async def test_get_by_checksum(db_session: AsyncSession) -> None:
    repository = DocumentRepository(db_session)
    document = Document(
        document_id=uuid4(),
        filename="report.pdf",
        storage_key=f"documents/{uuid4()}.pdf",
        checksum="a" * 64,
    )

    await repository.create(document)

    result = await repository.get_by_checksum("a" * 64)

    assert result is not None
    assert result.document_id == document.document_id
    assert result.checksum == "a" * 64


@pytest.mark.asyncio
async def test_update_document_status(db_session: AsyncSession) -> None:
    repository = DocumentRepository(db_session)
    document_id = uuid4()

    document = Document(
        document_id=document_id,
        filename="report.pdf",
        storage_key=f"documents/{document_id}.pdf",
        checksum="a" * 64,
    )
    await repository.create(document)

    updated = await repository.update_status(
        document_id,
        DocumentStatus.PROCESSING,
    )

    assert updated is not None
    assert updated.status == DocumentStatus.PROCESSING

    retrieved = await repository.get(document_id)
    assert retrieved is not None
    assert retrieved.status == DocumentStatus.PROCESSING
