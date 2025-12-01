import logging
from pathlib import Path
from typing import Dict

from api.core.base_model import BaseModelWrapper
from models import SectionClassifier

logger = logging.getLogger(__name__)


class SectionClassifierModel(BaseModelWrapper):
    MODEL_NAME = "SectionClassifier"

    def __init__(self, model_path: Path):
        super().__init__(model_path)

    def _load_model(self):
        return SectionClassifier.load(self.model_path)

    def on_loaded(self) -> None:
        if not self.model:
            return
        logger.info(
            "[SectionClassifier] fitted=%s classes=%s",
            getattr(self.model, "is_fitted", False),
            getattr(self.model, "SECTION_CLASSES", []),
        )

    def segment(self, text: str, min_confidence: float = 0.3) -> Dict[str, str]:
        if not self.is_loaded():
            return {}
        try:
            model = self.require_model()
            return model.segment_resume(text, min_confidence=min_confidence)
        except Exception as exc:
            logger.error("[SectionClassifier] segmentation failure: %s", exc, exc_info=True)
            return {}

    def is_loaded(self) -> bool:
        return super().is_loaded() and getattr(self.model, 'is_fitted', False)
