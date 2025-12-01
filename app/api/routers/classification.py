import logging

from fastapi import APIRouter, HTTPException, status

from api.modules.resume_classifier.service import ResumeClassifierService
from api.modules.section_classifier.service import SectionClassifierService
from api.schemas.request import CategoryClassifyRequest, TextClassifyRequest
from api.schemas.response import CategoryResponse, SectionsResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/sections', response_model=SectionsResponse)
async def classify_sections(request: TextClassifyRequest):
	service = SectionClassifierService()
	try:
		sections = service.classify(request.text)
		return {'success': True, 'sections': sections}
	except Exception as exc:
		logger.error("Section classification failed: %s", exc, exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Error classifying sections: {exc}",
		)


@router.post('/category', response_model=CategoryResponse)
async def classify_category(request: CategoryClassifyRequest):
	service = ResumeClassifierService()
	try:
		classification = service.classify(
			request.text,
			{'skills': request.skills or []},
		)
		return {'success': True, 'classification': classification}
	except Exception as exc:
		logger.error("Category classification failed: %s", exc, exc_info=True)
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Error classifying category: {exc}",
		)