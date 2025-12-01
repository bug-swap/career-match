import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from api.core.base_model import BaseModelWrapper
from app import BASE_DIR
from config import Settings, settings
from models.entity_extractor.model import EntityExtractor

logger = logging.getLogger(__name__)


class EntityExtractorModel(BaseModelWrapper):
    MODEL_NAME = "EntityExtractor"
    DEFAULT_SPACY_MODEL = "en_core_web_md"

    @staticmethod
    def empty_response() -> Dict:
        return {
            'contact': {
                'name': '',
                'email': '',
                'phone': '',
                'linkedin': '',
                'github': '',
            },
            'entities': {
                'job_titles': [],
                'companies': [],
                'skills': [],
                'work_dates': [],
                'degrees': [],
                'majors': [],
                'institutions': [],
                'graduation_years': [],
                'certifications': [],
                'projects': [],
                'publications': [],
                'languages': [],
                'summary': None,
            },
        }

    def _load_model(self):
        skills_path = BASE_DIR / "data" / "raw" / "structured" / "person_skills.csv"
        model = EntityExtractor.from_skills_csv(skills_path, spacy_model=self.DEFAULT_SPACY_MODEL)
        logger.info(
            "[EntityExtractor] Loaded persisted skills from %s",
            self.model_path,
        )
        return model

    def on_loaded(self) -> None:
        if not self.model:
            return
        logger.info(
            "[EntityExtractor] spaCy=%s skills=%s",
            bool(getattr(self.model, 'nlp', None)),
            len(getattr(self.model, 'skills_set', [])),
        )

    def extract(self, text: str) -> Dict:
        if not self.is_loaded():
            return self.empty_response()

        extractor = self.require_model()
        result = extractor.extract(text)
        data = result.to_dict()
        payload = self.empty_response()
        payload['contact'] = {
            'name': data.get('name', ''),
            'email': data.get('email', ''),
            'phone': data.get('phone', ''),
            'linkedin': data.get('linkedin', ''),
            'github': data.get('github', ''),
        }
        payload['entities'] = {
            'job_titles': data.get('job_titles', []),
            'companies': data.get('companies', []),
            'skills': data.get('skills', []),
            'work_dates': data.get('work_dates', []),
            'degrees': data.get('degrees', []),
            'majors': data.get('majors', []),
            'institutions': data.get('institutions', []),
            'graduation_years': data.get('graduation_years', []),
            'certifications': data.get('certifications', []),
            'projects': data.get('projects', []),
            'publications': data.get('publications', []),
            'languages': data.get('languages', []),
            'summary': data.get('summary', None),
        }
        return payload
