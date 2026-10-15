import pytest

from app.rag.pdf_validation import validate_pdf


def test_accepts_pdf_header() -> None:
    validate_pdf(b"%PDF-1.7\nsample content", "report.pdf")


def test_rejects_non_pdf_extension() -> None:
    with pytest.raises(ValueError, match="Only PDF"):
        validate_pdf(b"%PDF-1.7\nsample content", "report.txt")


def test_rejects_invalid_pdf_header() -> None:
    with pytest.raises(ValueError, match="PDF header"):
        validate_pdf(b"not a pdf", "report.pdf")


def test_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="PDF header"):
        validate_pdf(b"", "report.pdf")
