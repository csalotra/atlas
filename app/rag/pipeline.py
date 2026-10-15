from app.rag.chunking import Chunker
from app.rag.embeddings import EmbeddingClient
from app.rag.models import EmbeddedChunk
from uuid import UUID


class EmbeddingPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.chunker = chunker
        self.embedding_client = embedding_client

    def embed_text(
        self,
        text: str,
        document_id: UUID,
    ) -> list[EmbeddedChunk]:
        chunks = self.chunker.chunk(text)

        return [
            EmbeddedChunk(
                text=chunk.text,
                index=chunk.index,
                embedding=self.embedding_client.embed(chunk.text),
                document_id=document_id,
            )
            for chunk in chunks
        ]
