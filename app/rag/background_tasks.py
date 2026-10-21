import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.rag.chunking import RecursiveChunker
from app.rag.embeddings import EmbeddingClient
from app.rag.indexing import DocumentIndexingService
from app.rag.local_document_storage import LocalDocumentStorage
from app.rag.pdf_extractor import PDFTextExtractor
from app.rag.pipeline import EmbeddingPipeline
from app.rag.qdrant_vector_store import QdrantVectorStore
from app.rag.document_models import DocumentStatus
from app.repositories.document import DocumentRepository
from app.core.exceptions import (
    DocumentNotFoundError,
    InvalidDocumentError,
)


logger = logging.getLogger(__name__)


async def index_document_in_background(document_id: UUID) -> None:
    print(f"DEBUG: Background indexing started for {document_id}", flush=True)
    logger.info("Starting indexing for document %s", document_id)
    settings = get_settings()
    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    try:
        async with AsyncSessionLocal() as session:
            print("DEBUG: Database session opened", flush=True)
            repository = DocumentRepository(session)
            storage = LocalDocumentStorage(settings.document_storage_dir)
            embedding_client = EmbeddingClient()

            pipeline = EmbeddingPipeline(
                chunker=RecursiveChunker(),
                embedding_client=embedding_client,
            )

            vector_store = QdrantVectorStore(qdrant_client)

            indexing_service = DocumentIndexingService(
                storage=storage,
                repository=repository,
                extractor=PDFTextExtractor(),
                pipeline=pipeline,
                vector_store=vector_store,
            )

            try:
                print("DEBUG: Ensuring Qdrant collection", flush=True)
                await vector_store.ensure_collection()
                print("DEBUG: Qdrant collection ensured", flush=True)
                print("DEBUG: Calling indexing service", flush=True)
                await indexing_service.index(document_id)
                await session.commit()
                logger.info("Indexed document %s", document_id)

            except Exception as exc:
                logger.exception(
                    "Indexing failed for document %s",
                    document_id,
                )

                try:
                    await session.rollback()
                except Exception:
                    logger.exception(
                        "Could not roll back indexing transaction for %s",
                        document_id,
                    )

                if isinstance(exc, DocumentNotFoundError):
                    raise

                # For a future retry mechanism.
                status = (
                    DocumentStatus.FAILED
                    if isinstance(exc, InvalidDocumentError)
                    else DocumentStatus.PENDING
                )

                try:
                    async with AsyncSessionLocal() as failure_session:
                        failure_repository = DocumentRepository(failure_session)
                        await failure_repository.update_status(
                            document_id,
                            status,
                        )
                        await failure_session.commit()
                except Exception:
                    logger.exception(
                        "Could not persist status %s for document %s",
                        status,
                        document_id,
                    )

                raise

    finally:
        await qdrant_client.close()
