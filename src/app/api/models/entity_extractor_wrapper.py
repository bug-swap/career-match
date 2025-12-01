"""
Wrapper for EntityExtractor model
Handles loading and provides type-safe interface
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

logger = logging.getLogger(__name__)


class EntityExtractorModel:
    """Wrapper class for EntityExtractor model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._load()
    
    def _load(self):
        """Load the EntityExtractor model"""
        try:
            # Add models directory to path
            models_dir = self.model_path.parent.parent.parent / 'models'
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            
            logger.info(f"[EntityExtractor] Loading from {self.model_path}")
            logger.info(f"[EntityExtractor] Path exists: {self.model_path.exists()}")
            
            if not self.model_path.exists():
                logger.error(f"[EntityExtractor] Model path does not exist: {self.model_path}")
                return
            
            # List files
            files = list(self.model_path.glob('*'))
            logger.info(f"[EntityExtractor] Files found: {[f.name for f in files]}")
            
            # Import and load
            from entity_extractor.model import EntityExtractor
            self.model = EntityExtractor.load(self.model_path)
            
            logger.info(f"[EntityExtractor] ✅ Loaded successfully")
            logger.info(f"[EntityExtractor] Using spaCy: {self.model.use_spacy}")
            logger.info(f"[EntityExtractor] Using regex: {self.model.use_regex}")
            
        except Exception as e:
            logger.error(f"[EntityExtractor] ❌ Failed to load: {e}", exc_info=True)
            self.model = None
    
    def extract(self, text: str) -> Dict:
        """
        Extract entities from text
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with extracted entities
        """
        if not self.model:
            logger.error("[EntityExtractor] Model not loaded, cannot extract")
            return {
                'contact': {},
                'companies': [],
                'job_titles': [],
                'skills': [],
                'education': [],
                'dates': [],
                'locations': []
            }
        
        try:
            # Call model's extract method which returns ExtractedEntities
            result = self.model.extract(text)
            
            # Convert to dict format
            return {
                'contact': {
                    'name': result.name,
                    'email': result.email,
                    'phone': result.phone,
                    'linkedin': result.linkedin,
                    'github': result.github
                },
                'companies': result.companies,
                'job_titles': result.job_titles,
                'skills': result.skills,
                'education': result.institutions,
                'dates': result.dates,
                'locations': result.locations
            }
        except Exception as e:
            logger.error(f"[EntityExtractor] Error during extraction: {e}", exc_info=True)
            return {
                'contact': {},
                'companies': [],
                'job_titles': [],
                'skills': [],
                'education': [],
                'dates': [],
                'locations': []
            }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
