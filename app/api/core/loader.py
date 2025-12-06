"""Singleton loader for inference wrappers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Type, TypeVar

from config import settings

from api.core.base_model import BaseModelWrapper
from api.modules.entity_extractor.wrapper import EntityExtractorModel
from api.modules.section_classifier.wrapper import SectionClassifierModel
from api.modules.resume_classifier.wrapper import ResumeClassifierModel
from api.modules.embedding.wrapper import EmbeddingModel

logger = logging.getLogger(__name__)

WrapperType = TypeVar('WrapperType', bound=BaseModelWrapper)


class ModelLoader:
    _instance: "ModelLoader" | None = None

    def __init__(self):
        self._models: Dict[str, BaseModelWrapper] = {}

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_all()
        return cls._instance

    def _load_all(self) -> None:
        logger.info("=" * 80)
        logger.info("Loading ML models")
        logger.info("=" * 80)

        registry = {
            'section_classifier': (SectionClassifierModel, settings.SECTION_CLASSIFIER_PATH),
            'entity_extractor': (EntityExtractorModel, settings.ENTITY_EXTRACTOR_PATH),
            'resume_classifier': (ResumeClassifierModel, settings.RESUME_CLASSIFIER_PATH),
            'job_matcher': (EmbeddingModel, settings.JOB_MATCHER_PATH),
        }

        for key, (wrapper_cls, path) in registry.items():
            self._register(key, wrapper_cls, path)

        self._log_summary()

    def _register(self, name: str, wrapper_cls: Type[WrapperType], path: Path) -> None:
        logger.info("Registering %s from %s", name, path)
        self._models[name] = wrapper_cls(path)

    def _log_summary(self) -> None:
        logger.info("=" * 80)
        logger.info("Model loading summary:")
        for name, wrapper in self._models.items():
            status = '✅ Loaded' if wrapper.is_loaded() else '❌ Failed'
            logger.info("  %-20s %s", name.replace('_', ' ').title(), status)
        logger.info("=" * 80)

    def get_model(self, name: str) -> BaseModelWrapper | None:
        return self._models.get(name)

    def get_status(self) -> Dict[str, str]:
        status = {
            key: ('loaded' if wrapper.is_loaded() else 'not loaded')
            for key, wrapper in self._models.items()
        }
        status['all_loaded'] = 'loaded' if all(val == 'loaded' for val in status.values()) else 'not loaded'
        return status
