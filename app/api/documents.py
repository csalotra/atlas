from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateDocumentError
from app.db.session import get_db
from app.dependencies import get_document_ingestion_service
from app.rag.document_source import DocumentSource
from app.rag.ingestion import DocumentIngestionService
from app.schemas.document import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
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
                detail="File exceeds the 10 MB upload limit.",
            )

        chunks.append(chunk)

    return b"".join(chunks)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
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

        return DocumentResponse(
            document_id=document.document_id,
            filename=document.filename,
            status=document.status,
            created_at=document.created_at,
        )

    except DuplicateDocumentError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except HTTPException:
        await session.rollback()
        raise
    except Exception:
        await session.rollback()
        raise
    finally:
        await file.close()
