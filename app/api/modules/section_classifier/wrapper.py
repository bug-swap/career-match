import logging
from pathlib import Path
from typing import Dict

from api.core.base_model import BaseModelWrapper
from models import SectionClassifier

logger = logging.getLogger(__name__)


class SectionClassifierModel(BaseModelWrapper):
    MODEL_NAME = "SectionClassifier"

    def __init__(self, model_path: Path = None):
        """
        Initialize the section classifier wrapper

        Note: model_path is optional since this is now rule-based
        """
        # For rule-based classifier, we don't need model artifacts
        super().__init__(model_path if model_path else Path("."))

    def _preload_nlp(self):
        """Preload spaCy NLP model for entity extraction"""
        try:
            import spacy
            logger.info("[SectionClassifier] Preloading spaCy NLP model...")
            try:
                spacy.load("en_core_web_lg")
                logger.info("[SectionClassifier] spaCy model loaded successfully")
            except OSError:
                logger.warning("[SectionClassifier] spaCy model not found, attempting download...")
                from spacy.cli import download
                download("en_core_web_lg")
                spacy.load("en_core_web_lg")
                logger.info("[SectionClassifier] spaCy model downloaded and loaded")
        except ImportError:
            logger.warning("[SectionClassifier] spaCy not installed - using pattern-based extraction only")
        except Exception as e:
            logger.error("[SectionClassifier] Failed to load spaCy model: %s", e)

    def _load_model(self):
        self._preload_nlp()
        return SectionClassifier()

    def on_loaded(self) -> None:
        """Called after model is loaded"""
        if not self.model:
            return

        # Check if NLP is available
        from models.section_classifier.model import SPACY_AVAILABLE
        nlp_status = "enabled" if SPACY_AVAILABLE else "disabled (pattern-based only)"

        logger.info(
            "[SectionClassifier] Rule-based classifier ready | sections=%s | NLP=%s",
            getattr(self.model, "SECTIONS", []),
            nlp_status,
        )

    def segment(self, text: str, min_confidence: float = 0.2, parse: bool = True) -> Dict:
        """
        Segment resume text into sections

        Args:
            text: Resume text to segment
            min_confidence: Minimum confidence threshold (default 0.2)
            parse: If True, parse sections into structured data (default True)

        Returns:
            Dictionary mapping section names to their content (parsed or raw)
        """
        if not self.is_loaded():
            logger.warning("[SectionClassifier] Model not loaded, returning empty dict")
            return {}
        try:
            model = self.require_model()
            sections = model.segment_resume(text)

            if parse and sections:
                sections = model.parse_sections(sections)

            return sections
        except Exception as exc:
            logger.error("[SectionClassifier] segmentation failure: %s", exc, exc_info=True)
            return {}

    def is_loaded(self) -> bool:
        """Check if classifier is loaded and ready"""
        return super().is_loaded() and getattr(self.model, 'is_fitted', False)
