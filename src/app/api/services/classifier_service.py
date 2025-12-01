import logging
from api.models.loader import ModelLoader

logger = logging.getLogger(__name__)


class ClassifierService:
    """Service for classifying resume into job categories"""
    
    def __init__(self):
        self.loader = ModelLoader.get_instance()
        # Load the complete ResumeClassifier wrapper
        self.model = self.loader.get_model('resume_classifier')
    
    def classify_resume(self, text: str, entities: dict) -> dict:
        """
        Classify resume into job category using ResumeClassifier.predict()
        
        Args:
            text: Resume text
            entities: Extracted entities (contains 'skills')
            
        Returns:
            {
                'category': str,
                'confidence': float,
                'top_3': list of dicts
            }
        """
        logger.info(f"[ClassifierService] Starting resume classification, text length: {len(text)}")
        
        if not self.model or not self.model.is_loaded():
            logger.error("[ClassifierService] Resume classifier model not loaded!")
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}]
            }
        
        try:
            # Extract skills for model input
            skills = entities.get('skills', [])
            logger.info(f"[ClassifierService] Using {len(skills)} skills for classification")
            
            # Use the wrapper's predict method
            logger.info("[ClassifierService] Calling wrapper.predict()")
            result = self.model.predict(text, skills=skills if skills else None, top_k=3)
            
            logger.info(f"[ClassifierService] Predicted category: {result['category']} (confidence: {result['confidence']:.4f})")
            logger.debug(f"[ClassifierService] Top 3 predictions: {result['top_3']}")
            
            return result
        
        except Exception as e:
            logger.error(f"[ClassifierService] Error in resume classification: {e}", exc_info=True)
            logger.error(f"[ClassifierService] Text preview: {text[:200]}...")
            # Return fallback result
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}]
            }
