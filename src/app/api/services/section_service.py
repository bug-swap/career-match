import logging
from api.models.loader import ModelLoader

logger = logging.getLogger(__name__)


class SectionService:
    """Service for classifying resume text into sections"""
    
    def __init__(self):
        self.loader = ModelLoader.get_instance()
        # Load the complete SectionClassifier model (includes vectorizer internally)
        self.model = self.loader.get_model('section_classifier')
    
    def classify_sections(self, text: str) -> dict:
        """
        Classify resume text into sections using SectionClassifier.segment_resume()
        
        Args:
            text: Preprocessed resume text
            
        Returns:
            Dictionary with section names as keys and text content as values
        """
        logger.info(f"[SectionService] Starting section classification, text length: {len(text)}")
        
        if not self.model or not self.model.is_loaded():
            logger.error("[SectionService] Model not loaded! Cannot classify sections")
            return {}
        
        contact = []
        education = []
        experience = []
        skills = []
        person_skills = []
        try:
            # Use the wrapper's segment_resume method
            logger.info("[SectionService] Calling wrapper.segment_resume()")
            sections = self.model.segment_resume(text, min_confidence=0.3)
            
            logger.info(f"[SectionService] Raw sections from model: {list(sections.keys())}")
            logger.debug(f"[SectionService] Section lengths: {[(k, len(v)) for k, v in sections.items()]}")
            
            # Convert section keys to lowercase for consistency
            sections = {k.lower(): v for k, v in sections.items()}
            
            logger.info(f"[SectionService] Final sections: {list(sections.keys())}")
            return sections
        
        except Exception as e:
            logger.error(f"[SectionService] Error in section classification: {e}", exc_info=True)
            logger.error(f"[SectionService] Text preview: {text[:200]}...")
            return {}
