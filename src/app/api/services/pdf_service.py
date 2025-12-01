import fitz  # PyMuPDF
import re
import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class PDFService:
    """Service for PDF text extraction and preprocessing"""
    
    async def extract_text(self, file: UploadFile) -> str:
        """
        Extract text from PDF file
        
        Args:
            file: Uploaded PDF file (FastAPI UploadFile)
            
        Returns:
            Extracted text string
        """
        try:
            pdf_bytes = await file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            
            doc.close()
            
            extracted_text = "\n".join(text_parts)
            logger.debug(f"Extracted {len(extracted_text)} characters from PDF")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Unicode normalization
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve structure
        text = re.sub(r'[^\w\s\.,;:\-@()\/]', '', text)
        
        return text.strip()
