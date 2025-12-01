from fastapi import APIRouter, HTTPException, status
import logging

from api.services.section_service import SectionService
from api.services.entity_service import EntityService
from api.services.classifier_service import ClassifierService
from api.schemas.request import TextClassifyRequest, CategoryClassifyRequest
from api.schemas.response import SectionsResponse, CategoryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/sections', response_model=SectionsResponse)
async def classify_sections(request: TextClassifyRequest):
    """
    Classify text into resume sections only
    
    **Request:**
    ```json
    {
        "text": "resume text..."
    }
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "sections": {
            "contact": "...",
            "summary": "...",
            "education": "...",
            "experience": "...",
            "skills": "..."
        }
    }
    ```
    """
    logger.info(f"Classifying sections for text ({len(request.text)} chars)")
    
    try:
        section_service = SectionService()
        sections = section_service.classify_sections(request.text)
        
        return {
            'success': True,
            'sections': sections
        }
    except Exception as e:
        logger.error(f"Error classifying sections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying sections: {str(e)}"
        )


@router.post('/category', response_model=CategoryResponse)
async def classify_category(request: CategoryClassifyRequest):
    """
    Classify resume category only
    
    **Request:**
    ```json
    {
        "text": "resume text...",
        "skills": ["Python", "Java"]  // optional
    }
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "classification": {
            "category": "Software Engineer",
            "confidence": 0.87,
            "top_3": [...]
        }
    }
    ```
    """
    logger.info(f"Classifying category for text ({len(request.text)} chars)")
    
    try:
        classifier_service = ClassifierService()
        entities = {'skills': request.skills or []}
        classification = classifier_service.classify_resume(request.text, entities)
        
        return {
            'success': True,
            'classification': classification
        }
    except Exception as e:
        logger.error(f"Error classifying category: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying category: {str(e)}"
        )