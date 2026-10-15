from unittest.mock import Mock

from app.rag.chunking import Chunk
from app.rag.models import EmbeddedChunk
from app.rag.pipeline import EmbeddingPipeline

from uuid import uuid4


def test_embed_text():
    chunker = Mock()
    embedding_client = Mock()
    document_id = uuid4()

    chunker.chunk.return_value = [
        Chunk(text="first chunk", index=0),
        Chunk(text="second chunk", index=1),
    ]

    embedding_client.embed.side_effect = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    pipeline = EmbeddingPipeline(
        chunker=chunker,
        embedding_client=embedding_client,
    )

    result = pipeline.embed_text("test document", document_id=document_id)

    assert result == [
        EmbeddedChunk(
            text="first chunk", index=0, embedding=[0.1, 0.2], document_id=document_id
        ),
        EmbeddedChunk(
            text="second chunk", index=1, embedding=[0.3, 0.4], document_id=document_id
        ),
    ]

    chunker.chunk.assert_called_once_with("test document")
    assert embedding_client.embed.call_count == 2
