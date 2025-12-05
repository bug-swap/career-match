import logging
from typing import Dict

from api.core.base_service import BaseService

from .wrapper import ResumeClassifierModel

logger = logging.getLogger(__name__)


class ResumeClassifierService(BaseService):
    def __init__(self):
        super().__init__()
        self.classifier = self.get_model('resume_classifier', ResumeClassifierModel)

    def classify(self, text: str, entities: Dict = None) -> Dict:
        """
        Classify resume text into category
        
        Args:
            text: Resume text
            entities: Extracted entities (kept for backwards compatibility, not used by new model)
        
        Returns:
            {category, confidence, top_3}
        """
        logger.info("[ResumeClassifierService] classifying length=%s", len(text))
        
        if not self.classifier or not self.classifier.is_loaded():
            logger.error("[ResumeClassifierService] model unavailable")
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}],
            }

        try:
            # New model uses SBERT embeddings - no need for skills
            return self.classifier.predict(text, top_k=3)
        except Exception as exc:
            logger.error("[ResumeClassifierService] failure: %s", exc, exc_info=True)
            self.log_preview(text, "ResumeClassifierService")
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}],
            }