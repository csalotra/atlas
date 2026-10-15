from pathlib import Path


def validate_pdf(content: bytes, filename: str) -> None:
    if Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if not content.startswith(b"%PDF-"):
        raise ValueError("File does not have a valid PDF header.")
