from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import Optional
import time
import logging

from api.services.pdf_service import PDFService
from api.services.section_service import SectionService
from api.services.entity_service import EntityService
from api.services.classifier_service import ClassifierService
from api.schemas.request import TextParseRequest
from api.schemas.response import (
    ResumeParseResponse, 
    SectionsOnlyResponse,
    EntitiesOnlyResponse,
    CategoryOnlyResponse,
    ExtractedTextResponse
)
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


async def validate_and_read_pdf(file: UploadFile) -> tuple[bytes, str]:
    """
    Validate PDF file and return contents
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        Tuple of (file_contents, filename)
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Only PDF files are allowed. Got: {file.filename}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE:
        logger.warning(f"File too large: {len(contents)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
        )
    
    await file.seek(0)  # Reset file pointer
    return contents, file.filename


@router.post('/parse', response_model=ResumeParseResponse)
async def parse_resume_complete(
    file: UploadFile = File(..., description="PDF file to parse")
):
    """
    **CENTRAL ROUTE** - Parse a resume PDF through the COMPLETE ML pipeline
    
    This endpoint runs all ML models sequentially:
    1. PDF Text Extraction
    2. Text Preprocessing  
    3. Section Classification
    4. Entity Extraction (NER)
    5. Resume Category Classification
    
    **Request:**
    - `file`: PDF file (multipart/form-data) - Required
    
    **Response:**
    - Complete resume parsing results including:
      - Extracted text and sections
      - Entities (skills, companies, education, etc.)
      - Resume category classification
      - Processing metadata
    
    **Raises:**
    - 400: Invalid file type or missing file
    - 413: File too large (>16MB)
    - 500: Processing error
    """
    start_time = time.time()
    
    # Validate and read PDF
    contents, filename = await validate_and_read_pdf(file)
    logger.info(f"[COMPLETE PIPELINE] Processing resume: {filename} ({len(contents)} bytes)")
    
    try:
        # Stage 0: PDF Extraction
        pdf_service = PDFService()
        raw_text = await pdf_service.extract_text(file)
        
        # Stage 1: Preprocessing
        clean_text = pdf_service.preprocess_text(raw_text)
        logger.debug(f"Extracted {len(clean_text)} characters from PDF")
        
        # Stage 2: Section Classification
        section_service = SectionService()
        sections = section_service.classify_sections(clean_text)
        logger.debug(f"Identified {len(sections)} sections")
        
        # Stage 3: Entity Extraction
        entity_service = EntityService()
        entities = entity_service.extract_entities(sections)
        logger.debug(f"Extracted entities: {len(entities.get('skills', []))} skills")
        
        # Stage 4: Category Classification
        classifier_service = ClassifierService()
        classification = classifier_service.classify_resume(clean_text, entities)
        logger.debug(f"Classified as: {classification['category']}")
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Build response
        response = {
            'success': True,
            'data': {
                'contact': entities.get('contact', {}),
                'sections': sections,
                'entities': {
                    'companies': entities.get('companies', []),
                    'job_titles': entities.get('job_titles', []),
                    'skills': entities.get('skills', []),
                    'education': entities.get('education', []),
                    'dates': entities.get('dates', []),
                    'locations': entities.get('locations', [])
                },
                'classification': {
                    'category': classification['category'],
                    'confidence': classification['confidence'],
                    'top_3': classification['top_3']
                },
                'metadata': {
                    'word_count': len(clean_text.split()),
                    'char_count': len(clean_text),
                    'processing_time_ms': processing_time
                }
            }
        }
        
        logger.info(f"[COMPLETE PIPELINE] Successfully processed in {processing_time}ms")
        return response
    
    except Exception as e:
        logger.error(f"[COMPLETE PIPELINE] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume: {str(e)}"
        )


@router.post('/extract-text', response_model=ExtractedTextResponse)
async def extract_text_from_pdf(
    file: UploadFile = File(..., description="PDF file to extract text from")
):
    """
    **MODEL 1: PDF Text Extraction** - Extract and preprocess text from PDF
    
    This endpoint only runs PDF text extraction and preprocessing.
    
    **Request:**
    - `file`: PDF file (multipart/form-data) - Required
    
    **Response:**
    - Raw extracted text
    - Preprocessed/cleaned text
    - Text statistics
    
    **Raises:**
    - 400: Invalid file type
    - 413: File too large
    - 500: Extraction error
    """
    start_time = time.time()
    
    # Validate and read PDF
    contents, filename = await validate_and_read_pdf(file)
    logger.info(f"[TEXT EXTRACTION] Processing: {filename}")
    
    try:
        pdf_service = PDFService()
        
        # Extract text
        raw_text = await pdf_service.extract_text(file)
        
        # Preprocess
        clean_text = pdf_service.preprocess_text(raw_text)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"[TEXT EXTRACTION] Extracted {len(clean_text)} chars in {processing_time}ms")
        
        return {
            'success': True,
            'raw_text': raw_text,
            'clean_text': clean_text,
            'metadata': {
                'raw_char_count': len(raw_text),
                'clean_char_count': len(clean_text),
                'word_count': len(clean_text.split()),
                'line_count': len(clean_text.split('\n')),
                'processing_time_ms': processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"[TEXT EXTRACTION] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting text: {str(e)}"
        )


@router.post('/classify-sections', response_model=SectionsOnlyResponse)
async def classify_sections_from_pdf(
    file: UploadFile = File(..., description="PDF file to classify sections from")
):
    """
    **MODEL 2: Section Classification** - Extract text and classify into resume sections
    
    This endpoint runs PDF extraction + section classification model.
    
    **Request:**
    - `file`: PDF file (multipart/form-data) - Required
    
    **Response:**
    - Classified resume sections (contact, education, experience, skills, etc.)
    - Processing metadata
    
    **Raises:**
    - 400: Invalid file type
    - 413: File too large
    - 500: Classification error
    """
    start_time = time.time()
    
    # Validate and read PDF
    contents, filename = await validate_and_read_pdf(file)
    logger.info(f"[SECTION CLASSIFICATION] Processing: {filename}")
    
    try:
        # Extract and preprocess text
        pdf_service = PDFService()
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess_text(raw_text)
        
        # Classify sections
        section_service = SectionService()
        sections = section_service.classify_sections(clean_text)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"[SECTION CLASSIFICATION] Found {len(sections)} sections in {processing_time}ms")
        
        return {
            'success': True,
            'sections': sections,
            'metadata': {
                'section_count': len(sections),
                'total_char_count': len(clean_text),
                'processing_time_ms': processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"[SECTION CLASSIFICATION] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying sections: {str(e)}"
        )


@router.post('/extract-entities', response_model=EntitiesOnlyResponse)
async def extract_entities_from_pdf(
    file: UploadFile = File(..., description="PDF file to extract entities from")
):
    """
    **MODEL 3: Entity Extraction (NER)** - Extract entities using spaCy NER model
    
    This endpoint runs PDF extraction + section classification + entity extraction.
    
    **Request:**
    - `file`: PDF file (multipart/form-data) - Required
    
    **Response:**
    - Extracted entities:
      - Contact information (name, email, phone, LinkedIn, GitHub)
      - Companies, job titles, skills
      - Education institutions
      - Dates and locations
    - Processing metadata
    
    **Raises:**
    - 400: Invalid file type
    - 413: File too large
    - 500: Extraction error
    """
    start_time = time.time()
    
    # Validate and read PDF
    contents, filename = await validate_and_read_pdf(file)
    logger.info(f"[ENTITY EXTRACTION] Processing: {filename}")
    
    try:
        # Extract and preprocess text
        pdf_service = PDFService()
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess_text(raw_text)
        
        # Classify sections
        section_service = SectionService()
        sections = section_service.classify_sections(clean_text)
        
        # Extract entities
        entity_service = EntityService()
        entities = entity_service.extract_entities(sections)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        total_entities = sum([
            len(entities.get('companies', [])),
            len(entities.get('job_titles', [])),
            len(entities.get('skills', [])),
            len(entities.get('education', [])),
            len(entities.get('dates', [])),
            len(entities.get('locations', []))
        ])
        
        logger.info(f"[ENTITY EXTRACTION] Extracted {total_entities} entities in {processing_time}ms")
        
        return {
            'success': True,
            'contact': entities.get('contact', {}),
            'entities': {
                'companies': entities.get('companies', []),
                'job_titles': entities.get('job_titles', []),
                'skills': entities.get('skills', []),
                'education': entities.get('education', []),
                'dates': entities.get('dates', []),
                'locations': entities.get('locations', [])
            },
            'metadata': {
                'total_entities': total_entities,
                'skills_count': len(entities.get('skills', [])),
                'companies_count': len(entities.get('companies', [])),
                'processing_time_ms': processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"[ENTITY EXTRACTION] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting entities: {str(e)}"
        )


@router.post('/classify-category', response_model=CategoryOnlyResponse)
async def classify_category_from_pdf(
    file: UploadFile = File(..., description="PDF file to classify category from")
):
    """
    **MODEL 4: Resume Category Classification** - Classify resume into job categories
    
    This endpoint runs PDF extraction + section classification + entity extraction + 
    category classification using ML classifier.
    
    **Request:**
    - `file`: PDF file (multipart/form-data) - Required
    
    **Response:**
    - Resume category classification
    - Confidence scores
    - Top 3 predictions
    - Processing metadata
    
    **Raises:**
    - 400: Invalid file type
    - 413: File too large
    - 500: Classification error
    """
    start_time = time.time()
    
    # Validate and read PDF
    contents, filename = await validate_and_read_pdf(file)
    logger.info(f"[CATEGORY CLASSIFICATION] Processing: {filename}")
    
    try:
        # Extract and preprocess text
        pdf_service = PDFService()
        raw_text = await pdf_service.extract_text(file)
        clean_text = pdf_service.preprocess_text(raw_text)
        
        # Classify sections
        section_service = SectionService()
        sections = section_service.classify_sections(clean_text)
        
        # Extract entities (needed for classification)
        entity_service = EntityService()
        entities = entity_service.extract_entities(sections)
        
        # Classify category
        classifier_service = ClassifierService()
        classification = classifier_service.classify_resume(clean_text, entities)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"[CATEGORY CLASSIFICATION] Classified as '{classification['category']}' in {processing_time}ms")
        
        return {
            'success': True,
            'classification': {
                'category': classification['category'],
                'confidence': classification['confidence'],
                'top_3': classification['top_3']
            },
            'metadata': {
                'processing_time_ms': processing_time
            }
        }
    
    except Exception as e:
        logger.error(f"[CATEGORY CLASSIFICATION] Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error classifying category: {str(e)}"
        )


@router.post('/parse/text', response_model=ResumeParseResponse)
async def parse_text(request: TextParseRequest):
    """
    Parse resume from raw text instead of PDF (legacy endpoint)
    
    **Request:**
    ```json
    {
        "text": "resume text content..."
    }
    ```
    
    **Response:** Same as /parse
    
    **Raises:**
    - 422: Invalid request body
    - 500: Processing error
    """
    start_time = time.time()
    text = request.text
    
    logger.info(f"Processing resume from text ({len(text)} characters)")
    
    try:
        # Process through pipeline (skip PDF extraction)
        section_service = SectionService()
        sections = section_service.classify_sections(text)
        
        entity_service = EntityService()
        entities = entity_service.extract_entities(sections)
        
        classifier_service = ClassifierService()
        classification = classifier_service.classify_resume(text, entities)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = {
            'success': True,
            'data': {
                'contact': entities.get('contact', {}),
                'sections': sections,
                'entities': {
                    'companies': entities.get('companies', []),
                    'job_titles': entities.get('job_titles', []),
                    'skills': entities.get('skills', []),
                    'education': entities.get('education', []),
                    'dates': entities.get('dates', []),
                    'locations': entities.get('locations', [])
                },
                'classification': {
                    'category': classification['category'],
                    'confidence': classification['confidence'],
                    'top_3': classification['top_3']
                },
                'metadata': {
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'processing_time_ms': processing_time
                }
            }
        }
        
        logger.info(f"Successfully processed text in {processing_time}ms")
        return response
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )


@router.post('/parse/text', response_model=ResumeParseResponse)
async def parse_text(request: TextParseRequest):
    """
    Parse resume from raw text instead of PDF
    
    **Request:**
    ```json
    {
        "text": "resume text content..."
    }
    ```
    
    **Response:** Same as /parse
    
    **Raises:**
    - 422: Invalid request body
    - 500: Processing error
    """
    start_time = time.time()
    text = request.text
    
    logger.info(f"Processing resume from text ({len(text)} characters)")
    
    try:
        # Process through pipeline (skip PDF extraction)
        section_service = SectionService()
        sections = section_service.classify_sections(text)
        
        entity_service = EntityService()
        entities = entity_service.extract_entities(sections)
        
        classifier_service = ClassifierService()
        classification = classifier_service.classify_resume(text, entities)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = {
            'success': True,
            'data': {
                'contact': entities.get('contact', {}),
                'sections': sections,
                'entities': {
                    'companies': entities.get('companies', []),
                    'job_titles': entities.get('job_titles', []),
                    'skills': entities.get('skills', []),
                    'education': entities.get('education', []),
                    'dates': entities.get('dates', []),
                    'locations': entities.get('locations', [])
                },
                'classification': {
                    'category': classification['category'],
                    'confidence': classification['confidence'],
                    'top_3': classification['top_3']
                },
                'metadata': {
                    'word_count': len(text.split()),
                    'char_count': len(text),
                    'processing_time_ms': processing_time
                }
            }
        }
        
        logger.info(f"Successfully processed text in {processing_time}ms")
        return response
    
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )