from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.db.session import get_db
from app.rag.document_storage import DocumentStorage
from app.rag.ingestion import DocumentIngestionService
from app.rag.local_document_storage import LocalDocumentStorage
from app.repositories.document import DocumentRepository
from app.rag.embeddings import EmbeddingClient
from app.rag.qdrant_vector_store import QdrantVectorStore
from app.rag.retrieval import RetrievalService


def get_document_storage() -> DocumentStorage:
    settings = get_settings()
    return LocalDocumentStorage(settings.document_storage_dir)


def get_document_ingestion_service(
    session: AsyncSession = Depends(get_db),
    storage: DocumentStorage = Depends(get_document_storage),
) -> DocumentIngestionService:
    repository = DocumentRepository(session)
    return DocumentIngestionService(
        storage=storage,
        repository=repository,
    )


def get_retrieval_service() -> RetrievalService:
    settings = get_settings()
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    return RetrievalService(
        embedding_client=EmbeddingClient(),
        vector_store=QdrantVectorStore(qdrant_client),
    )
