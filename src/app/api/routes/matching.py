from fastapi import APIRouter, HTTPException, status
import logging

from api.services.matcher_service import MatcherService
from api.schemas.request import JobMatchRequest
from api.schemas.response import JobMatchResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/jobs', response_model=JobMatchResponse)
async def match_jobs(request: JobMatchRequest):
    """
    Match resume with job descriptions
    
    **Request:**
    ```json
    {
        "resume_text": "...",
        "jobs": [
            {
                "job_id": "J001",
                "title": "Software Engineer",
                "description": "...",
                "required_skills": ["Python", "Java"],
                "location": "San Francisco"
            }
        ],
        "top_k": 10
    }
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "matches": [
            {
                "job_id": "J001",
                "match_score": 92.5,
                "skill_match": 88.0,
                "experience_match": 95.0,
                "education_match": true,
                "location_match": true
            }
        ]
    }
    ```
    """
    logger.info(f"Matching resume with {len(request.jobs)} jobs (top_k={request.top_k})")
    
    try:
        matcher_service = MatcherService()
        matches = matcher_service.match_jobs(
            resume_text=request.resume_text,
            jobs=[job.dict() for job in request.jobs],
            top_k=request.top_k
        )
        
        logger.info(f"Found {len(matches)} job matches")
        
        return {
            'success': True,
            'matches': matches
        }
    except Exception as e:
        logger.error(f"Error matching jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching jobs: {str(e)}"
        )
