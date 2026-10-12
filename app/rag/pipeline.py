from app.rag.chunking import Chunker
from app.rag.embeddings import EmbeddingClient
from app.rag.models import EmbeddedChunk


class EmbeddingPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedding_client: EmbeddingClient,
    ) -> None:
        self.chunker = chunker
        self.embedding_client = embedding_client

    def embed_text(self, text: str) -> list[EmbeddedChunk]:
        chunks = self.chunker.chunk(text)

        return [
            EmbeddedChunk(
                text=chunk.text,
                index=chunk.index,
                embedding=self.embedding_client.embed(chunk.text),
            )
            for chunk in chunks
        ]
