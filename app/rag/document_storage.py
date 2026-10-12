from abc import ABC, abstractmethod
from pathlib import Path


class DocumentStorage(ABC):
    @abstractmethod
    def save(self, source: Path, storage_key: str) -> None:
        """Save a source file under a storage key."""
        ...

    @abstractmethod
    def read(self, storage_key: str) -> bytes:
        """Read a stored document as bytes."""
        ...

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        """Delete a stored document."""
        ...
