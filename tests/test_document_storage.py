from pathlib import Path

import pytest

from app.rag.local_document_storage import LocalDocumentStorage


def test_save_and_read_document(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"sample PDF content")

    storage = LocalDocumentStorage(tmp_path / "storage")
    storage.save(source, "documents/report.pdf")

    assert storage.read("documents/report.pdf") == b"sample PDF content"


def test_delete_document(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"sample PDF content")

    storage = LocalDocumentStorage(tmp_path / "storage")
    storage.save(source, "report.pdf")
    storage.delete("report.pdf")

    with pytest.raises(FileNotFoundError):
        storage.read("report.pdf")


def test_reject_path_traversal(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path / "storage")

    with pytest.raises(ValueError):
        storage.read("../../outside.txt")


def test_missing_source_raises_error(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path / "storage")

    with pytest.raises(FileNotFoundError):
        storage.save(tmp_path / "missing.pdf", "report.pdf")
