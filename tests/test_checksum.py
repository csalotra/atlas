import pytest

from app.rag.checksum import calculate_checksum


def test_same_content_produces_same_checksum(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"sample PDF content")
    second.write_bytes(b"sample PDF content")

    assert calculate_checksum(first) == calculate_checksum(second)


def test_different_content_produces_different_checksum(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"document A")
    second.write_bytes(b"document B")

    assert calculate_checksum(first) != calculate_checksum(second)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        calculate_checksum(tmp_path / "missing.pdf")