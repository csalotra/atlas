from pathlib import Path
from unittest.mock import MagicMock, patch

from app.rag.pdf_extractor import PDFTextExtractor


def test_extract_text_joins_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()

    with patch("app.rag.pdf_extractor.PdfReader") as reader_class:
        reader = MagicMock()
        reader.pages = [
            MagicMock(),
            MagicMock(),
        ]
        reader.pages[0].extract_text.return_value = "First page"
        reader.pages[1].extract_text.return_value = "Second page"
        reader_class.return_value = reader

        extractor = PDFTextExtractor()
        result = extractor.extract_text(pdf_path)

    assert result == "First page\n\nSecond page"
    reader_class.assert_called_once_with(pdf_path)


def test_extract_text_handles_empty_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()

    with patch("app.rag.pdf_extractor.PdfReader") as reader_class:
        reader = MagicMock()
        reader.pages = [MagicMock(), MagicMock()]
        reader.pages[0].extract_text.return_value = None
        reader.pages[1].extract_text.return_value = "Some text"
        reader_class.return_value = reader

        result = PDFTextExtractor().extract_text(pdf_path)

    assert result == "Some text"
