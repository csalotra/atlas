from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_retrieval_service
from app.rag.retrieval import RetrievalService
from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResponse:
    chunks = await service.retrieve(
        question=request.question,
        limit=request.limit,
    )

    return RetrievalResponse(
        results=[
            RetrievedChunk(
                text=chunk.text,
                document_id=str(chunk.document_id),
                index=chunk.index,
                page_number=chunk.page_number,
            )
            for chunk in chunks
        ]
    )
