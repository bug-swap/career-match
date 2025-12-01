"""Resume processing endpoints using the new module services."""
from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.modules.entity_extractor.service import EntityExtractorService
from api.modules.pdf.service import PDFService
from api.modules.resume_classifier.service import ResumeClassifierService
from api.modules.section_classifier.service import SectionClassifierService
from api.schemas.request import TextParseRequest
from api.schemas.response import (
    CategoryOnlyResponse,
    EntitiesOnlyResponse,
    ExtractedTextResponse,
    ResumeParseResponse,
    SectionsOnlyResponse,
)
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

pdf_service = PDFService()
section_service = SectionClassifierService()
entity_service = EntityExtractorService()
classifier_service = ResumeClassifierService()


async def _validate_pdf(file: UploadFile) -> Tuple[int, str]:
    if not file.filename.lower().endswith('.pdf'):
        logger.warning("Invalid file type uploaded: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are allowed.",
        )

    contents = await file.read()
    size = len(contents)
    if size > settings.MAX_UPLOAD_SIZE:
        logger.warning("File too large: %s bytes", size)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large. Maximum size is "
                f"{settings.MAX_UPLOAD_SIZE / (1024 * 1024):.0f}MB"
            ),
        )

    await file.seek(0)
    return size, file.filename


def _entities_metadata(entities: Dict, duration_ms: int) -> Dict:
    skills = len(entities.get('skills', []))
    companies = len(entities.get('companies', []))
    total = sum(len(values) for values in entities.values() if isinstance(values, list))
    return {
        'total_entities': total,
        'skills_count': skills,
        'companies_count': companies,
        'processing_time_ms': duration_ms,
    }


def _sections_metadata(sections: Dict[str, str], clean_text: str, duration_ms: int) -> Dict:
    return {
        'section_count': len(sections),
        'total_char_count': len(clean_text),
        'processing_time_ms': duration_ms,
    }


def _classification_metadata(duration_ms: int) -> Dict:
    return {'processing_time_ms': duration_ms}


def _text_metadata(raw_text: str, clean_text: str, duration_ms: int) -> Dict:
    return {
        'raw_char_count': len(raw_text),
        'clean_char_count': len(clean_text),
        'word_count': len(clean_text.split()),
        'line_count': len(clean_text.split('\n')),
        'processing_time_ms': duration_ms,
    }


def _resume_metadata(clean_text: str, duration_ms: int) -> Dict:
    return {
        'word_count': len(clean_text.split()),
        'char_count': len(clean_text),
        'processing_time_ms': duration_ms,
    }


def _classify_sections(text: str) -> Dict[str, str]:
    sections = section_service.classify(text)
    logger.debug("[SectionClassifier] identified %s sections", len(sections))
    return sections


def _extract_entities(raw_text) -> Dict:
    entities = entity_service.extract(raw_text)
    logger.debug(
        "[EntityExtractor] extracted %s skills",
        len(entities.get('skills', [])),
    )
    return entities


def _classify_resume(text: str, entities: Dict) -> Dict:
    classification = classifier_service.classify(text, entities)
    logger.debug(
        "[ResumeClassifier] predicted %s (confidence=%.2f)",
        classification.get('category'),
        classification.get('confidence'),
    )
    return classification


@router.post('/parse', response_model=ResumeParseResponse)
async def parse_resume(file: UploadFile = File(...)):
    start = time.time()
    size, filename = await _validate_pdf(file)
    logger.info("[ResumePipeline] Processing %s (%s bytes)", filename, size)

    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        sections = _classify_sections(raw_text)
        entities = _extract_entities(raw_text)
        classification = _classify_resume(clean_text, entities)

        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'data': {
                'contact': entities.get('contact', {}),
                'sections': sections,
                'entities': entities.get('entities', {}),
                'classification': classification,
                'metadata': _resume_metadata(clean_text, duration_ms),
            },
        }
    except Exception as exc:
        logger.error("[ResumePipeline] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume: {exc}",
        )


@router.post('/extract-text', response_model=ExtractedTextResponse)
async def extract_text(file: UploadFile = File(...)):
    start = time.time()
    _, filename = await _validate_pdf(file)
    logger.info("[TextExtraction] Processing %s", filename)

    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'raw_text': raw_text,
            'clean_text': clean_text,
            'metadata': _text_metadata(raw_text, clean_text, duration_ms),
        }
    except Exception as exc:
        logger.error("[TextExtraction] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting text: {exc}",
        )


@router.post('/classify-sections', response_model=SectionsOnlyResponse)
async def classify_sections(file: UploadFile = File(...)):
    start = time.time()
    _, filename = await _validate_pdf(file)
    logger.info("[SectionClassification] Processing %s", filename)

    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        sections = _classify_sections(raw_text)
        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'sections': sections,
            'metadata': _sections_metadata(sections, clean_text, duration_ms),
        }
    except Exception as exc:
        logger.error("[SectionClassification] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying sections: {exc}",
        )


@router.post('/extract-entities', response_model=EntitiesOnlyResponse)
async def extract_entities(file: UploadFile = File(...)):
    start = time.time()
    _, filename = await _validate_pdf(file)
    logger.info("[EntityExtraction] Processing %s", filename)

    try:
        raw_text = await pdf_service.extract_text(file)
        entities = _extract_entities(raw_text=raw_text)
        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'contact': entities.get('contact', {}),
            'entities': entities.get('entities', {}),
            'metadata': _entities_metadata(entities, duration_ms),
        }
    except Exception as exc:
        logger.error("[EntityExtraction] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting entities: {exc}",
        )


@router.post('/classify-category', response_model=CategoryOnlyResponse)
async def classify_category(file: UploadFile = File(...)):
    start = time.time()
    _, filename = await _validate_pdf(file)
    logger.info("[CategoryClassification] Processing %s", filename)

    try:
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess(raw_text)
        sections = _classify_sections(raw_text)
        entities = _extract_entities(sections)
        classification = _classify_resume(clean_text, entities)
        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'classification': classification,
            'metadata': _classification_metadata(duration_ms),
        }
    except Exception as exc:
        logger.error("[CategoryClassification] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying category: {exc}",
        )


@router.post('/parse/text', response_model=ResumeParseResponse)
async def parse_resume_text(request: TextParseRequest):
    start = time.time()
    text = request.text
    logger.info("[ResumePipeline/Text] Processing text length=%s", len(text))

    try:
        sections = _classify_sections(text)
        entities = _extract_entities(sections)
        classification = _classify_resume(text, entities)
        duration_ms = int((time.time() - start) * 1000)

        return {
            'success': True,
            'data': {
                'contact': entities.get('contact', {}),
                'sections': sections,
                'entities': entities.get('entities', {}),
                'classification': classification,
                'metadata': _resume_metadata(text, duration_ms),
            },
        }
    except Exception as exc:
        logger.error("[ResumePipeline/Text] failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {exc}",
        )
