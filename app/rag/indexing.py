import logging

from uuid import UUID
from time import perf_counter

from app.rag.document_models import DocumentStatus
from app.rag.document_storage import DocumentStorage
from app.rag.pdf_extractor import PDFTextExtractor
from app.rag.pipeline import EmbeddingPipeline
from app.rag.vector_store import VectorStore
from app.repositories.document import DocumentRepository
from app.core.exceptions import DocumentNotFoundError, InvalidDocumentError

logger = logging.getLogger(__name__)


class DocumentIndexingService:
    def __init__(
        self,
        storage: DocumentStorage,
        repository: DocumentRepository,
        extractor: PDFTextExtractor,
        pipeline: EmbeddingPipeline,
        vector_store: VectorStore,
    ) -> None:
        self.storage = storage
        self.repository = repository
        self.extractor = extractor
        self.pipeline = pipeline
        self.vector_store = vector_store

    async def index(self, document_id: UUID) -> None:
        document = await self.repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")

        await self.repository.update_status(
            document_id,
            DocumentStatus.PROCESSING,
        )

        logger.info("PDF reading started: document_id=%s", document_id)
        start = perf_counter()
        pdf_bytes = self.storage.read(document.storage_key)
        logger.info(
            "PDF reading completed: document_id=%s duration_seconds=%.2f",
            document_id,
            perf_counter() - start,
        )

        logger.info("PDF extraction started: document_id=%s", document_id)
        start = perf_counter()
        pages = self.extractor.extract_pages(pdf_bytes)
        logger.info(
            "PDF extraction completed: document_id=%s pages=%d duration_seconds=%.2f",
            document_id,
            len(pages),
            perf_counter() - start,
        )

        if not any(page.text.strip() for page in pages):
            raise InvalidDocumentError("No extractable text found in PDF.")

        logger.info("Embedding started: document_id=%s", document_id)
        start = perf_counter()
        embedded_chunks = await self.pipeline.embed_pages(
            pages,
            document_id,
        )
        logger.info(
            "Embedding completed: document_id=%s chunks=%d duration_seconds=%.2f",
            document_id,
            len(embedded_chunks),
            perf_counter() - start,
        )

        if not embedded_chunks:
            raise InvalidDocumentError("No chunks were generated from the PDF.")

        logger.info("Qdrant storage started: document_id=%s", document_id)
        start = perf_counter()
        await self.vector_store.add(embedded_chunks)
        logger.info(
            "Qdrant storage completed: document_id=%s vectors=%d duration_seconds=%.2f",
            document_id,
            len(embedded_chunks),
            perf_counter() - start,
        )

        await self.repository.update_status(
            document_id,
            DocumentStatus.INDEXED,
        )
        logger.info("Indexing completed: document_id=%s", document_id)
