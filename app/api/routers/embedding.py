import logging
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.params import File

from api.modules.embedding.service import EmbeddingService
from api.schemas.response import EmbeddingResponse
from api.modules.pdf.service import PDFService

router = APIRouter()
logger = logging.getLogger(__name__)

service = EmbeddingService()
pdf_service = PDFService()

@router.post("", response_model=EmbeddingResponse)
async def get_embedding(file: UploadFile = File(...)) -> dict:
    start = time.time()
    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        resume_embedding = service.get_embedding(clean_text)
        duration_ms = int((time.time() - start) * 1000)
        return EmbeddingResponse(
            success=True,
            embedding=resume_embedding,
            processing_time_ms=duration_ms,
        )
    except Exception as exc:
        logger.error("[ResumePipeline] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume: {exc}",
        )