"""Lightweight base class for API services."""
from __future__ import annotations

import logging
from typing import Optional, Type, TypeVar

from api.core.base_model import BaseModelWrapper
from api.core.loader import ModelLoader

logger = logging.getLogger(__name__)

WrapperType = TypeVar('WrapperType', bound=BaseModelWrapper)


class BaseService:
    def __init__(self, loader: Optional[ModelLoader] = None):
        self.loader = loader or ModelLoader.get_instance()

    def get_model(self, name: str, expected: Type[WrapperType]) -> Optional[WrapperType]:
        model = self.loader.get_model(name)
        if model and isinstance(model, expected):
            return model
        logger.error("[%s] Missing or invalid model '%s'", self.__class__.__name__, name)
        return None

    @staticmethod
    def log_preview(text: str, label: str, length: int = 200) -> None:
        logger.error("[%s] Text preview: %s...", label, text[:length])
