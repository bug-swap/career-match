"""
Wrapper for ResumeClassifier model
Handles loading and provides type-safe interface
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import torch

logger = logging.getLogger(__name__)


class ResumeClassifierModel:
    """Wrapper class for ResumeClassifier model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self._load()
    
    def _load(self):
        """Load the ResumeClassifier model"""
        try:
            # Add models directory to path
            models_dir = self.model_path.parent.parent.parent / 'models'
            if str(models_dir) not in sys.path:
                sys.path.insert(0, str(models_dir))
            
            logger.info(f"[ResumeClassifier] Loading from {self.model_path}")
            logger.info(f"[ResumeClassifier] Path exists: {self.model_path.exists()}")
            
            if not self.model_path.exists():
                logger.error(f"[ResumeClassifier] Model path does not exist: {self.model_path}")
                return
            
            # List files
            files = list(self.model_path.glob('*'))
            logger.info(f"[ResumeClassifier] Files found: {[f.name for f in files]}")
            
            # Import and load
            from resume_classifier.model import ResumeClassifier
            self.model = ResumeClassifier.load(self.model_path, device=torch.device('cpu'))
            
            # Set to eval mode
            if self.model and hasattr(self.model, 'model'):
                self.model.model.eval()
            
            logger.info(f"[ResumeClassifier] ✅ Loaded successfully")
            logger.info(f"[ResumeClassifier] is_fitted: {self.model.is_fitted}")
            logger.info(f"[ResumeClassifier] Classes: {self.model.num_classes}")
            
        except Exception as e:
            logger.error(f"[ResumeClassifier] ❌ Failed to load: {e}", exc_info=True)
            self.model = None
    
    def predict(self, text: str, skills: Optional[List[str]] = None, top_k: int = 3) -> Dict:
        """
        Classify resume into job category
        
        Args:
            text: Resume text
            skills: Optional list of skills
            top_k: Number of top predictions
            
        Returns:
            Dictionary with category, confidence, and top_k predictions
        """
        if not self.model:
            logger.error("[ResumeClassifier] Model not loaded, cannot predict")
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}]
            }
        
        try:
            # Call model's predict method which returns ClassificationResult
            result = self.model.predict(text, skills=skills, top_k=top_k)
            
            # Convert to dict format
            return {
                'category': result.category,
                'confidence': float(result.confidence),
                'top_3': [
                    {'category': cat, 'confidence': float(conf)}
                    for cat, conf in result.top_k
                ]
            }
        except Exception as e:
            logger.error(f"[ResumeClassifier] Error during prediction: {e}", exc_info=True)
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}]
            }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and hasattr(self.model, 'is_fitted') and self.model.is_fitted
