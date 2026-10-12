import pytest

from app.rag.chunking import Chunk, RecursiveChunker


def test_chunk_text_returns_chunks():
    chunker = RecursiveChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    chunks = chunker.chunk(
        "This is a long document that needs to be split into multiple chunks."
    )

    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_text_preserves_content():
    chunker = RecursiveChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    text = "This is a long document that needs to be split."

    chunks = chunker.chunk(text)

    assert chunks
    assert any("This is a long" in chunk.text for chunk in chunks)
    assert any("document" in chunk.text for chunk in chunks)
    assert any("needs to be" in chunk.text for chunk in chunks)
    assert any("split." in chunk.text for chunk in chunks)


def test_empty_text_returns_empty_list():
    chunker = RecursiveChunker()

    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_chunk_overlap_is_configured():
    chunker = RecursiveChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunker.splitter._chunk_size == 20
    assert chunker.splitter._chunk_overlap == 5


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
