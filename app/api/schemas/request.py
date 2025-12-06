from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class TextParseRequest(BaseModel):
    """Request schema for parsing text resume"""
    text: str = Field(..., min_length=50, description="Resume text content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "John Doe\nSoftware Engineer\nEmail: john@example.com..."
            }
        }


class TextClassifyRequest(BaseModel):
    """Request schema for text classification into sections"""
    text: str = Field(..., min_length=50, description="Text to classify into sections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Education\nBachelor of Science in Computer Science..."
            }
        }


class CategoryClassifyRequest(BaseModel):
    """Request schema for category classification"""
    text: str = Field(..., min_length=50, description="Resume text for category classification")
    skills: Optional[List[str]] = Field(default=None, description="Optional list of skills")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Experienced software engineer with 5 years in Python...",
                "skills": ["Python", "Java", "AWS"]
            }
        }


class JobDescription(BaseModel):
    """Job description model"""
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description")
    required_skills: Optional[List[str]] = Field(default=None, description="Required skills")
    location: Optional[str] = Field(default=None, description="Job location")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "J001",
                "title": "Senior Software Engineer",
                "description": "We are looking for an experienced software engineer...",
                "required_skills": ["Python", "Django", "PostgreSQL"],
                "location": "San Francisco, CA"
            }
        }


class EmbeddingResponse(BaseModel):
    """Response schema for embedding extraction"""
    success: bool = Field(..., description="Whether embedding extraction was successful")
    embedding: List[float] = Field(..., description="Extracted embedding vector")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "embedding": [0.123, 0.456, 0.789, ...],
                "processing_time_ms": 150
            }
        }