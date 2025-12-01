import logging
from typing import Dict, Optional

from api.core.base_service import BaseService
from config import settings

from .wrapper import EntityExtractorModel

logger = logging.getLogger(__name__)


class EntityExtractorService(BaseService):
    def __init__(self, extractor: Optional[EntityExtractorModel] = None):
        super().__init__()
        self.extractor = extractor or self.get_model('entity_extractor', EntityExtractorModel)
        if not self.extractor:
            self.extractor = self._load_inline()

    def _load_inline(self) -> Optional[EntityExtractorModel]:
        try:
            logger.info("[EntityExtractorService] Lazily loading EntityExtractorModel")
            return EntityExtractorModel(settings.ENTITY_EXTRACTOR_PATH)
        except Exception as exc:
            logger.error("[EntityExtractorService] Inline load failed: %s", exc, exc_info=True)
            return None

    def extract(self, raw_text: str) -> Dict:
        if not self.extractor or not self.extractor.is_loaded():
            logger.error("[EntityExtractorService] model unavailable")
            return EntityExtractorModel.empty_response()

        if not isinstance(raw_text, str):
            logger.warning("[EntityExtractorService] invalid raw_text payload: %s", type(raw_text))
            return EntityExtractorModel.empty_response()

        if not raw_text:
            logger.warning("[EntityExtractorService] no raw_text provided")
            return EntityExtractorModel.empty_response()
        try:
            return self.extractor.extract(raw_text)
        except Exception as exc:
            logger.error("[EntityExtractorService] failure: %s", exc, exc_info=True)
            self.log_preview(raw_text, "EntityExtractorService")
            return EntityExtractorModel.empty_response()
