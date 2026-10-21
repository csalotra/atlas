from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.rag.chunking import Chunk
from app.rag.models import EmbeddedChunk
from app.rag.pipeline import EmbeddingPipeline
from app.rag.pdf_extractor import ExtractedPage


@pytest.mark.asyncio
async def test_embed_pages_preserves_page_numbers_and_sequential_indices() -> None:
    chunker = Mock()
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        side_effect=[
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
    )
    document_id = uuid4()

    chunker.chunk.side_effect = [
        [
            Chunk(text="first chunk", index=0, page_number=1),
            Chunk(text="second chunk", index=1, page_number=1),
        ],
        [
            Chunk(text="third chunk", index=0, page_number=2),
        ],
    ]

    pipeline = EmbeddingPipeline(
        chunker=chunker,
        embedding_client=embedding_client,
    )

    pages = [
        ExtractedPage(page_number=1, text="First page text"),
        ExtractedPage(page_number=2, text="Second page text"),
    ]

    result = await pipeline.embed_pages(pages, document_id)

    assert result == [
        EmbeddedChunk(
            text="first chunk",
            index=0,
            embedding=[0.1, 0.2],
            document_id=document_id,
            page_number=1,
        ),
        EmbeddedChunk(
            text="second chunk",
            index=1,
            embedding=[0.3, 0.4],
            document_id=document_id,
            page_number=1,
        ),
        EmbeddedChunk(
            text="third chunk",
            index=2,
            embedding=[0.5, 0.6],
            document_id=document_id,
            page_number=2,
        ),
    ]

    assert chunker.chunk.call_args_list == [
        (("First page text", 1),),
        (("Second page text", 2),),
    ]
    assert embedding_client.embed.await_count == 3
