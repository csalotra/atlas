from abc import ABC, abstractmethod


class DocumentStorage(ABC):
    @abstractmethod
    def save(self, content: bytes, storage_key: str) -> None:
        """Save document content under a storage key."""
        ...

    @abstractmethod
    def read(self, storage_key: str) -> bytes:
        """Read a stored document as bytes."""
        ...

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        """Delete a stored document."""
        ...
