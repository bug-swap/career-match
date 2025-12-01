"""
Wrapper for SectionClassifier model
Handles loading and provides type-safe interface
"""
import logging
from pathlib import Path
from typing import Dict, Optional
import sys

logger = logging.getLogger(__name__)


class SectionClassifierModel:
    """Wrapper class for SectionClassifier model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._load()
    
    def _load(self):
        """Load the SectionClassifier model"""
        try:
            # Add models directory to path
            models_dir = self.model_path.parent.parent.parent / 'models'
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            
            logger.info(f"[SectionClassifier] Loading from {self.model_path}")
            logger.info(f"[SectionClassifier] Path exists: {self.model_path.exists()}")
            
            if not self.model_path.exists():
                logger.error(f"[SectionClassifier] Model path does not exist: {self.model_path}")
                return
            
            # List files
            files = list(self.model_path.glob('*'))
            logger.info(f"[SectionClassifier] Files found: {[f.name for f in files]}")
            
            # Import and load
            from section_classifier.model import SectionClassifier
            self.model = SectionClassifier.load(self.model_path)
            
            logger.info(f"[SectionClassifier] ✅ Loaded successfully")
            logger.info(f"[SectionClassifier] is_fitted: {self.model.is_fitted}")
            logger.info(f"[SectionClassifier] Classes: {self.model.SECTION_CLASSES}")
            
        except Exception as e:
            logger.error(f"[SectionClassifier] ❌ Failed to load: {e}", exc_info=True)
            self.model = None
    
    def segment_resume(self, text: str, min_confidence: float = 0.3) -> Dict[str, str]:
        """
        Segment resume text into sections
        
        Args:
            text: Resume text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary of section name -> section content
        """
        if not self.model:
            logger.error("[SectionClassifier] Model not loaded, cannot segment")
            return {}
        
        try:
            return self.model.segment_resume(text, min_confidence=min_confidence)
        except Exception as e:
            logger.error(f"[SectionClassifier] Error during segmentation: {e}", exc_info=True)
            return {}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and hasattr(self.model, 'is_fitted') and self.model.is_fitted
