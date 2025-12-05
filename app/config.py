import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Union, Optional


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings"""
    
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True,
        extra='ignore'
    )
    
    # App settings
    APP_NAME: str = "Career Match ML Service"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    REPO_PATH : Path = BASE_DIR.parent
    MODELS_DIR: Path = BASE_DIR / 'artifacts'
    SECTION_CLASSIFIER_PATH: Path = MODELS_DIR / 'section_classifier'
    ENTITY_EXTRACTOR_PATH: Path = MODELS_DIR / 'entity_extractor'
    RESUME_CLASSIFIER_PATH: Path = MODELS_DIR / 'resume_classifier'
    JOB_MATCHER_PATH: Path = MODELS_DIR / 'job_matcher'
    
    # Upload settings
    MAX_UPLOAD_SIZE: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: set = {'pdf'}
    
    # Processing settings
    PROCESSING_TIMEOUT: int = 60  # seconds
    
    # CORS settings - can be comma-separated string or list
    CORS_ORIGINS: Union[str, List[str]] = "*"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # 'json' or 'text'
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from string or list"""
        if isinstance(v, str):
            # Split by comma and strip whitespace
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('DEBUG', mode='before')
    @classmethod
    def parse_debug(cls, v):
        """Parse DEBUG from string or bool"""
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes')
        return bool(v)


# Create global settings instance
settings = Settings()