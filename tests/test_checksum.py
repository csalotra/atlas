from app.rag.checksum import calculate_checksum


def test_same_content_produces_same_checksum() -> None:
    content = b"sample PDF content"

    assert calculate_checksum(content) == calculate_checksum(content)


def test_different_content_produces_different_checksum() -> None:
    assert calculate_checksum(b"document A") != calculate_checksum(b"document B")


def test_empty_content_produces_checksum() -> None:
    assert calculate_checksum(b"") == calculate_checksum(b"")
