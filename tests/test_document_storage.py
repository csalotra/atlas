from pathlib import Path

import pytest

from app.rag.local_document_storage import LocalDocumentStorage


def test_save_and_read_document(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path / "storage")
    content = b"sample PDF content"

    storage.save(content, "documents/report.pdf")

    assert storage.read("documents/report.pdf") == content


def test_delete_document(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path / "storage")
    storage.save(b"sample PDF content", "report.pdf")

    storage.delete("report.pdf")

    with pytest.raises(FileNotFoundError):
        storage.read("report.pdf")


def test_reject_path_traversal(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path / "storage")

    with pytest.raises(ValueError):
        storage.read("../../outside.txt")
