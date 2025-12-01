"""PDF extraction utilities."""
import logging
import re

import fitz
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class PDFService:
    async def extract_text(self, file: UploadFile) -> str:
        try:
            pdf_bytes = await file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            logger.debug("[PDFService] extracted %s characters", len(text))
            return text
        except Exception as exc:
            logger.error("[PDFService] extraction failure: %s", exc, exc_info=True)
            raise

    @staticmethod
    def preprocess(text: str) -> str:
        ascii_text = text.encode('ascii', 'ignore').decode('ascii')
        ascii_text = re.sub(r'\s+', ' ', ascii_text)
        ascii_text = re.sub(r'[^\w\s\.,;:\-@()/]', '', ascii_text)
        return ascii_text.strip()
