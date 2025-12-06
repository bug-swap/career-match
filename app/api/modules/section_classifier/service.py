import logging
from typing import Dict, Any

from api.core.base_service import BaseService

from .wrapper import SectionClassifierModel

logger = logging.getLogger(__name__)


class SectionClassifierService(BaseService):
    def __init__(self):
        super().__init__()
        self.classifier = self.get_model('section_classifier', SectionClassifierModel)

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify resume text into structured sections

        Returns:
            Dictionary with section names as keys and structured data as values
            (lists for experience/projects/publications, dict for contact, etc.)
        """
        logger.info("[SectionClassifierService] classifying text length=%s", len(text))
        if not self.classifier or not self.classifier.is_loaded():
            logger.error("[SectionClassifierService] model unavailable")
            return {}
        try:
            return self.classifier.segment(text)
        except Exception as exc:
            logger.error("[SectionClassifierService] failure: %s", exc, exc_info=True)
            self.log_preview(text, "SectionClassifierService")
            return {}
