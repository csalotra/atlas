from uuid import UUID

from app.rag.document_models import DocumentStatus
from app.rag.document_storage import DocumentStorage
from app.rag.pdf_extractor import PDFTextExtractor
from app.rag.pipeline import EmbeddingPipeline
from app.rag.vector_store import VectorStore
from app.repositories.document import DocumentRepository


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
            raise ValueError(f"Document not found: {document_id}")

        await self.repository.update_status(
            document_id,
            DocumentStatus.PROCESSING,
        )

        try:
            pdf_bytes = self.storage.read(document.storage_key)
            text = self.extractor.extract_text(pdf_bytes)

            if not text.strip():
                raise ValueError("No extractable text found in PDF.")

            embedded_chunks = self.pipeline.embed_text(text, document_id)

            if not embedded_chunks:
                raise ValueError("No chunks were generated from the PDF.")

            self.vector_store.add(embedded_chunks)

            await self.repository.update_status(
                document_id,
                DocumentStatus.INDEXED,
            )
        except Exception:
            await self.repository.update_status(
                document_id,
                DocumentStatus.FAILED,
            )
            raise
