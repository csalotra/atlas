from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


class PDFTextExtractor:
    def extract_pages(self, source: Path | bytes) -> list[ExtractedPage]:
        pdf_source = BytesIO(source) if isinstance(source, bytes) else source
        reader = PdfReader(pdf_source)

        return [
            ExtractedPage(page_number=page_number, text=page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    def extract_text(self, source: Path | bytes) -> str:
        pages = self.extract_pages(source)
        return "\n\n".join(page.text for page in pages).strip()
