from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_document_ingestion_service
from app.rag.background_tasks import index_document_in_background
from app.rag.document_source import DocumentSource
from app.rag.ingestion import DocumentIngestionService
from app.schemas.document import DocumentResponse
from app.repositories.document import DocumentRepository


router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


async def read_upload_with_limit(file: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total_size = 0

    while True:
        remaining = MAX_UPLOAD_SIZE + 1 - total_size
        chunk = await file.read(min(UPLOAD_CHUNK_SIZE, remaining))

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 50 MB upload limit.",
            )

        chunks.append(chunk)

    return b"".join(chunks)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentResponse:
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A filename is required.",
            )

        if file.content_type not in (
            None,
            "application/pdf",
            "application/octet-stream",
        ):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only PDF files are supported.",
            )

        content = await read_upload_with_limit(file)

        document = await service.ingest(
            source=DocumentSource(filename=file.filename, content=content)
        )
        await session.commit()

        background_tasks.add_task(
            index_document_in_background,
            document.document_id,
        )

        return DocumentResponse(
            document_id=document.document_id,
            filename=document.filename,
            status=document.status,
            created_at=document.created_at,
        )

    except HTTPException:
        await session.rollback()
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        await file.close()


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    repository = DocumentRepository(session)
    document = await repository.get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        status=document.status,
        created_at=document.created_at,
    )
