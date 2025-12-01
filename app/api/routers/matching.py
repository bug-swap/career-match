import logging

from fastapi import APIRouter, HTTPException, status

from api.modules.job_matcher.service import JobMatcherService
from api.schemas.request import JobMatchRequest
from api.schemas.response import JobMatchResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/jobs', response_model=JobMatchResponse)
async def match_jobs(request: JobMatchRequest):
    service = JobMatcherService()
    logger.info(
        "Matching resume against %d jobs (top_k=%d)",
        len(request.jobs),
        request.top_k,
    )

    try:
        matches = service.match(
            resume_text=request.resume_text,
            jobs=[job.dict() for job in request.jobs],
            top_k=request.top_k,
        )
        return {'success': True, 'matches': matches}
    except Exception as exc:
        logger.error("Job matching failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching jobs: {exc}",
        )
