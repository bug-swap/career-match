import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from api.core.base_model import BaseModelWrapper
from models import ResumeClassifier

logger = logging.getLogger(__name__)


class ResumeClassifierModel(BaseModelWrapper):
    MODEL_NAME = "ResumeClassifier"

    def __init__(self, model_path: Path):
        super().__init__(model_path)

    def _load_model(self):
        model = ResumeClassifier.load(self.model_path, device=torch.device('cpu'))
        if model and hasattr(model, 'model'):
            model.model.eval()
        return model

    def on_loaded(self) -> None:
        if not self.model:
            return
        logger.info(
            "[ResumeClassifier] fitted=%s classes=%s",
            getattr(self.model, 'is_fitted', False),
            getattr(self.model, 'num_classes', 'unknown'),
        )

    def predict(self, text: str, skills: Optional[List[str]] = None, top_k: int = 3) -> Dict:
        if not self.is_loaded():
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}],
            }
        try:
            classifier = self.require_model()
            result = classifier.predict(text, skills=skills, top_k=top_k)
            return {
                'category': result.category,
                'confidence': float(result.confidence),
                'top_3': [
                    {'category': cat, 'confidence': float(conf)}
                    for cat, conf in result.top_k
                ],
            }
        except Exception as exc:
            logger.error("[ResumeClassifier] prediction failure: %s", exc, exc_info=True)
            return {
                'category': 'UNKNOWN',
                'confidence': 0.0,
                'top_3': [{'category': 'UNKNOWN', 'confidence': 0.0}],
            }

    def is_loaded(self) -> bool:
        return super().is_loaded() and getattr(self.model, 'is_fitted', False)
