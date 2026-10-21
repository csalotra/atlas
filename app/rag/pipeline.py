"""
This module generates embeddings for document text by chunking extracted
pages and sending each chunk to an embedding client.

Key Responsibilities:
- Chunking text while preserving page numbers.
- Embedding each chunk.
- Associating embedded chunks with their document and sequential index.
"""

from uuid import UUID

from app.rag.chunking import Chunker
from app.rag.embeddings import EmbeddingClient
from app.rag.models import EmbeddedChunk
from app.rag.pdf_extractor import ExtractedPage


class EmbeddingPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.chunker = chunker
        self.embedding_client = embedding_client

    async def embed_pages(
        self,
        pages: list[ExtractedPage],
        document_id: UUID,
    ) -> list[EmbeddedChunk]:
        embedded_chunks: list[EmbeddedChunk] = []
        chunk_index = 0

        for page in pages:
            chunks = self.chunker.chunk(page.text, page.page_number)

            for chunk in chunks:
                embedding = await self.embedding_client.embed(chunk.text)
                embedded_chunks.append(
                    EmbeddedChunk(
                        text=chunk.text,
                        index=chunk_index,
                        embedding=embedding,
                        document_id=document_id,
                        page_number=chunk.page_number,
                    )
                )
                chunk_index += 1

        return embedded_chunks
