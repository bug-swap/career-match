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


class JobMatchRequest(BaseModel):
    """Request schema for job matching"""
    resume_text: str = Field(..., min_length=100, description="Resume text content")
    jobs: List[JobDescription] = Field(..., min_items=1, description="List of job descriptions")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of top matches to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Experienced software engineer with 5 years...",
                "jobs": [
                    {
                        "job_id": "J001",
                        "title": "Senior Software Engineer",
                        "description": "Looking for experienced Python developer...",
                        "required_skills": ["Python", "Django"],
                        "location": "Remote"
                    }
                ],
                "top_k": 5
            }
        }