from pathlib import Path

from app.core.exceptions import InvalidDocumentError


def validate_pdf(content: bytes, filename: str) -> None:
    if Path(filename).suffix.lower() != ".pdf":
        raise InvalidDocumentError("Only PDF files are supported.")

    if not content.startswith(b"%PDF-"):
        raise InvalidDocumentError("File does not have a valid PDF header.")
