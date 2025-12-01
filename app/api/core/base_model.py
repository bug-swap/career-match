"""Common functionality for model wrappers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseModelWrapper:
    """Provides uniform loading, logging, and readiness checks."""

    MODEL_NAME = "BaseModel"

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self._ensure_models_path()
        self._load()

    def _ensure_models_path(self) -> None:
        models_dir = self.model_path.parent.parent.parent / "models"
        models_dir_str = str(models_dir)
        if models_dir_str not in sys.path:
            sys.path.insert(0, models_dir_str)

    def _load(self) -> None:
        prefix = f"[{self.MODEL_NAME}]"
        try:
            logger.info("%s Loading from %s", prefix, self.model_path)
            if not self.model_path.exists():
                logger.error("%s Missing path: %s", prefix, self.model_path)
                return

            files = [f.name for f in self.model_path.glob("*")]
            if files:
                logger.debug("%s Files: %s", prefix, files)

            self.model = self._load_model()
            if self.is_loaded():
                self.on_loaded()
                logger.info("%s ✅ ready", prefix)
            else:
                logger.error("%s Loaded object failed readiness check", prefix)
        except Exception as exc:  # pragma: no cover
            logger.error("%s ❌ %s", prefix, exc, exc_info=True)
            self.model = None

    def _load_model(self):  # pragma: no cover
        raise NotImplementedError

    def on_loaded(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return self.model is not None

    def require_model(self):
        if not self.is_loaded():
            raise RuntimeError(f"{self.MODEL_NAME} not loaded")
        return self.model
