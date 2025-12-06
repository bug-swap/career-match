import logging
from typing import List

from api.core.base_service import BaseService

from .wrapper import EmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingService(BaseService):
    def __init__(self):
        super().__init__()
        self.matcher = self.get_model('job_matcher', EmbeddingModel)


    def get_embedding(self, text: str) -> List[float]:
        if not self.matcher or not self.matcher.is_loaded():
            logger.warning("[JobEmbeddorService] model unavailable")
            return []
        try:
            embedding = self.matcher.encode_single(text)
            return embedding.tolist()
        except Exception as exc:
            logger.error("[JobEmbeddorService] get_embedding failure: %s", exc, exc_info=True)
            return []
