from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    text: str
    document_id: str
    index: int
    page_number: int | None


class RetrievalResponse(BaseModel):
    results: list[RetrievedChunk]
