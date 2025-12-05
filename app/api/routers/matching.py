import logging
import time

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.params import File

from api.modules.job_matcher.service import JobMatcherService
from api.schemas.request import JobMatchRequest
from api.schemas.response import JobMatchResponse
from api.modules.database.service import DatabaseService, get_db
from api.modules.pdf.service import PDFService
from api.routers.resume import _classify_resume, _extract_entities, _resume_metadata, _validate_pdf

router = APIRouter()
logger = logging.getLogger(__name__)

service = JobMatcherService()
pdf_service = PDFService()

@router.post('/jobs')
async def match_jobs(file: UploadFile = File(...), db: DatabaseService = Depends(get_db)) -> dict:
    start = time.time()
    size, filename = await _validate_pdf(file)
    logger.info("[ResumePipeline] Processing %s (%s bytes)", filename, size)

    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        entities = _extract_entities(raw_text=raw_text)
        category_result = _classify_resume(clean_text, entities)
        resume_embedding = service.get_embedding(clean_text)
        data = db.call_rpc(
            "get_similar_jobs_by_category",
            {
                "p_embedding": resume_embedding,
                "p_category": category_result['category'],
                "p_limit": 10
            }
        )
        print(resume_embedding)
        duration_ms = int((time.time() - start) * 1000)
        return {
            'success': True,
            'data': {
                'jobs': data,
                'processing_time_ms': duration_ms   
            },
        }
    except Exception as exc:
        logger.error("[ResumePipeline] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume: {exc}",
        )