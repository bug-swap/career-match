import logging
from api.models.loader import ModelLoader

logger = logging.getLogger(__name__)


class EntityService:
    """Service for extracting entities from resume sections"""
    
    # Regex patterns for fallback extraction
    EMAIL_PATTERN = r'[\w\.-]+@[\w\.-]+\.\w+'
    PHONE_PATTERN = r'\+?\d[\d\-\(\)\s]{8,}\d'
    LINKEDIN_PATTERN = r'(?:linkedin\.com/in/|linkedin\.com/pub/)[\w\-]+'
    GITHUB_PATTERN = r'(?:github\.com/)[\w\-]+'
    DATE_PATTERN = r'(?:\d{4}|\w+\s+\d{4})\s*(?:-|to|–)\s*(?:\d{4}|\w+\s+\d{4}|present|current)'
    
    def __init__(self):
        self.loader = ModelLoader.get_instance()
        self.nlp = self.loader.get_model('entity_extractor')
    
    def extract_entities(self, sections: dict) -> dict:
        """
        Extract entities from resume sections using EntityExtractor.extract()
        
        Args:
            sections: Dictionary of section text
            
        Returns:
            Dictionary of extracted entities
        """
        logger.info(f"[EntityService] Starting entity extraction from {len(sections)} sections")
        
        if not self.nlp or not self.nlp.is_loaded():
            logger.error("[EntityService] Entity extractor model not loaded!")
            return {
                'contact': {},
                'companies': [],
                'job_titles': [],
                'skills': [],
                'education': [],
                'dates': [],
                'locations': []
            }
        
        # Combine all sections into full text for extraction
        full_text = '\n\n'.join([
            f"{section.upper()}\n{content}" 
            for section, content in sections.items()
        ])
        
        logger.debug(f"[EntityService] Combined text length: {len(full_text)}")
        
        try:
            # Use the wrapper's extract method
            logger.info("[EntityService] Calling wrapper.extract()")
            entities = self.nlp.extract(full_text)
            
            # Filter out None values from contact
            if 'contact' in entities:
                entities['contact'] = {k: v for k, v in entities['contact'].items() if v}
            
            logger.info(f"[EntityService] Successfully extracted entities")
            return entities
        
        except Exception as e:
            logger.error(f"[EntityService] Error in entity extraction: {e}", exc_info=True)
            logger.error(f"[EntityService] Text preview: {full_text[:200]}...")
            # Return empty structure on error
            return {
                'contact': {},
                'companies': [],
                'job_titles': [],
                'skills': [],
                'education': [],
                'dates': [],
                'locations': []
            }