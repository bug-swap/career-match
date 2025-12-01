import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from api.core.base_model import BaseModelWrapper
from models import JobMatcher

logger = logging.getLogger(__name__)


class JobMatcherModel(BaseModelWrapper):
    MODEL_NAME = "JobMatcher"

    def __init__(self, model_path: Path):
        super().__init__(model_path)

    def _load_model(self):
        model = JobMatcher.load(self.model_path, device=torch.device('cpu'))
        if model and getattr(model, 'encoder', None):
            model.encoder.eval()
        return model

    def on_loaded(self) -> None:
        if not self.model:
            return
        logger.info(
            "[JobMatcher] fitted=%s embedding_dim=%s",
            getattr(self.model, 'is_fitted', False),
            getattr(self.model, 'embedding_dim', 'unknown'),
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded():
            return np.array([])
        try:
            return self.require_model().encode(texts)
        except Exception as exc:
            logger.error("[JobMatcher] encode failure: %s", exc, exc_info=True)
            return np.array([])

    def encode_single(self, text: str) -> np.ndarray:
        if not self.is_loaded():
            return np.array([])
        try:
            return self.require_model().encode_single(text)
        except Exception as exc:
            logger.error("[JobMatcher] encode_single failure: %s", exc, exc_info=True)
            return np.array([])

    def similarity(self, resume_embedding: np.ndarray, job_embeddings: np.ndarray) -> np.ndarray:
        if not self.is_loaded():
            return np.array([])
        try:
            return self.require_model().compute_similarity(resume_embedding, job_embeddings)
        except Exception as exc:
            logger.error("[JobMatcher] similarity failure: %s", exc, exc_info=True)
            return np.array([])

    def is_loaded(self) -> bool:
        return super().is_loaded() and getattr(self.model, 'is_fitted', False)
