from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class PDFTextExtractor:
    def extract_text(self, source: Path | bytes) -> str:
        pdf_source = BytesIO(source) if isinstance(source, bytes) else source
        reader = PdfReader(pdf_source)

        pages = [page.extract_text() or "" for page in reader.pages]

        return "\n\n".join(pages).strip()
