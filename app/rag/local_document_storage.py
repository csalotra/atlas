from pathlib import Path

from app.rag.document_storage import DocumentStorage


class LocalDocumentStorage(DocumentStorage):
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, storage_key: str) -> Path:
        path = (self.root_dir / storage_key).resolve()

        if not path.is_relative_to(self.root_dir):
            raise ValueError("Storage key must stay within the storage directory")

        return path

    def save(self, content: bytes, storage_key: str) -> None:
        destination = self._resolve_path(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    def read(self, storage_key: str) -> bytes:
        return self._resolve_path(storage_key).read_bytes()

    def delete(self, storage_key: str) -> None:
        path = self._resolve_path(storage_key)
        path.unlink(missing_ok=True)
