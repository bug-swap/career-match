from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")


class ModelsStatusResponse(BaseModel):
    """Models loading status response"""
    models: Dict[str, str] = Field(..., description="Status of each model")
    all_loaded: bool = Field(..., description="Whether all models are loaded")


class ContactInfo(BaseModel):
    """Contact information schema"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class EntityInfo(BaseModel):
    """Extracted entities schema"""
    job_titles: List[str] = Field(default_factory=list, description="Extracted job titles")
    companies: List[str] = Field(default_factory=list, description="Extracted company names")
    work_dates: List[str] = Field(default_factory=list, description="Extracted work experience dates")
    skills: List[str] = Field(default_factory=list, description="Extracted skills")
    
    # Education
    degrees: List[str] = Field(default_factory=list, description="Extracted degrees")
    majors: List[str] = Field(default_factory=list, description="Extracted majors")
    institutions: List[str] = Field(default_factory=list, description="Extracted educational institutions")
    graduation_years: List[str] = Field(default_factory=list, description="Extracted graduation years")
    gpa: Optional[str] = Field(None, description="Extracted GPA")
    
    # Other
    certifications: List[str] = Field(default_factory=list, description="Extracted certifications")
    projects: List[str] = Field(default_factory=list, description="Extracted projects")
    publications: List[str] = Field(default_factory=list, description="Extracted publications")
    languages: List[str] = Field(default_factory=list, description="Languages known")
    summary: Optional[str] = Field(None, description="Summary or objective statement")  



class ClassificationDetail(BaseModel):
    """Single classification detail"""
    category: str = Field(..., description="Category name")
    confidence: float = Field(..., description="Confidence score")


class Classification(BaseModel):
    """Classification result schema"""
    category: str = Field(..., description="Top predicted category")
    confidence: float = Field(..., description="Confidence score for top category")
    top_3: List[ClassificationDetail] = Field(..., description="Top 3 predictions")


class Metadata(BaseModel):
    """Processing metadata schema"""
    word_count: int = Field(..., description="Number of words in resume")
    char_count: int = Field(..., description="Number of characters in resume")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ResumeData(BaseModel):
    """Resume parsing data"""
    contact: Dict[str, Any] = Field(default_factory=dict, description="Contact information")
    sections: Dict[str, Any] = Field(default_factory=dict, description="Resume sections (structured)")
    entities: EntityInfo = Field(..., description="Extracted entities")
    classification: Classification = Field(..., description="Resume classification")
    metadata: Metadata = Field(..., description="Processing metadata")


class ResumeParseResponse(BaseModel):
    """Resume parse response schema"""
    success: bool = Field(..., description="Whether parsing was successful")
    data: ResumeData = Field(..., description="Parsed resume data")


# ============================================================================
# Individual Model Responses
# ============================================================================

class TextMetadata(BaseModel):
    """Text extraction metadata"""
    raw_char_count: int = Field(..., description="Character count in raw text")
    clean_char_count: int = Field(..., description="Character count in cleaned text")
    word_count: int = Field(..., description="Word count in cleaned text")
    line_count: int = Field(..., description="Line count in cleaned text")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class ExtractedTextResponse(BaseModel):
    """Response for text extraction endpoint"""
    success: bool = Field(..., description="Whether extraction was successful")
    raw_text: str = Field(..., description="Raw extracted text from PDF")
    clean_text: str = Field(..., description="Preprocessed/cleaned text")
    metadata: TextMetadata = Field(..., description="Text statistics and metadata")


class SectionsMetadata(BaseModel):
    """Sections classification metadata"""
    section_count: int = Field(..., description="Number of sections identified")
    total_items: int = Field(..., description="Total number of items across all sections")
    total_char_count: int = Field(..., description="Total character count")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class SectionsOnlyResponse(BaseModel):
    """Response for section classification endpoint"""
    success: bool = Field(..., description="Whether classification was successful")
    sections: Dict[str, Any] = Field(..., description="Classified resume sections (structured)")
    metadata: SectionsMetadata = Field(..., description="Classification metadata")


class EntitiesMetadata(BaseModel):
    """Entity extraction metadata"""
    total_entities: int = Field(..., description="Total number of entities extracted")
    skills_count: int = Field(..., description="Number of skills extracted")
    companies_count: int = Field(..., description="Number of companies extracted")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class EntitiesOnlyResponse(BaseModel):
    """Response for entity extraction endpoint"""
    success: bool = Field(..., description="Whether extraction was successful")
    contact: Dict[str, Any] = Field(default_factory=dict, description="Contact information")
    entities: EntityInfo = Field(..., description="Extracted entities")
    metadata: EntitiesMetadata = Field(..., description="Extraction metadata")


class CategoryMetadata(BaseModel):
    """Category classification metadata"""
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class CategoryOnlyResponse(BaseModel):
    """Response for category classification endpoint"""
    success: bool = Field(..., description="Whether classification was successful")
    classification: Classification = Field(..., description="Resume category classification")
    metadata: CategoryMetadata = Field(..., description="Classification metadata")


# ============================================================================
# Other Responses (for /classify and /match routes)
# ============================================================================

class SectionsResponse(BaseModel):
    """Sections classification response"""
    success: bool = Field(..., description="Whether classification was successful")
    sections: Dict[str, Any] = Field(..., description="Classified sections (structured)")


class CategoryResponse(BaseModel):
    """Category classification response"""
    success: bool = Field(..., description="Whether classification was successful")
    classification: Classification = Field(..., description="Classification result")

class EmbeddingResponse(BaseModel):
    """Embedding response"""
    success: bool = Field(..., description="Whether embedding was successful")
    embedding: List[float] = Field(..., description="Generated embedding vector")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    