import logging
from pathlib import Path
from config import settings
from .section_classifier_wrapper import SectionClassifierModel
from .entity_extractor_wrapper import EntityExtractorModel
from .resume_classifier_wrapper import ResumeClassifierModel
from .job_matcher_wrapper import JobMatcherModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class to load and manage all ML models"""
    
    _instance = None
    _models = {}
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Load all ML models into memory using wrapper classes"""
        try:
            logger.info("=" * 80)
            logger.info("Loading ML models...")
            logger.info("=" * 80)
            
            # Load Section Classifier
            section_path = settings.SECTION_CLASSIFIER_PATH
            self._models['section_classifier'] = SectionClassifierModel(section_path)
            
            # Load Entity Extractor
            entity_path = settings.ENTITY_EXTRACTOR_PATH
            self._models['entity_extractor'] = EntityExtractorModel(entity_path)
            
            # Load Resume Classifier
            resume_path = settings.RESUME_CLASSIFIER_PATH
            self._models['resume_classifier'] = ResumeClassifierModel(resume_path)
            
            # Load Job Matcher
            matcher_path = settings.JOB_MATCHER_PATH
            self._models['job_matcher'] = JobMatcherModel(matcher_path)
            
            # Summary
            logger.info("=" * 80)
            logger.info("Model loading summary:")
            logger.info(f"  Section Classifier: {'✅ Loaded' if self._models['section_classifier'].is_loaded() else '❌ Failed'}")
            logger.info(f"  Entity Extractor:   {'✅ Loaded' if self._models['entity_extractor'].is_loaded() else '❌ Failed'}")
            logger.info(f"  Resume Classifier:  {'✅ Loaded' if self._models['resume_classifier'].is_loaded() else '❌ Failed'}")
            logger.info(f"  Job Matcher:        {'✅ Loaded' if self._models['job_matcher'].is_loaded() else '❌ Failed'}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {e}", exc_info=True)
            raise
    
    def get_model(self, model_name: str):
        """Get a specific model by name"""
        return self._models.get(model_name)
    
    def get_status(self) -> dict:
        """Get loading status of all models"""
        status = {
            'models': {
                'section_classifier': 'loaded' if 'section_classifier' in self._models and self._models['section_classifier'].is_loaded() else 'not loaded',
                'entity_extractor': 'loaded' if 'entity_extractor' in self._models and self._models['entity_extractor'].is_loaded() else 'not loaded',
                'resume_classifier': 'loaded' if 'resume_classifier' in self._models and self._models['resume_classifier'].is_loaded() else 'not loaded',
                'job_matcher': 'loaded' if 'job_matcher' in self._models and self._models['job_matcher'].is_loaded() else 'not loaded'
            }
        }
        status['all_loaded'] = all(v == 'loaded' for v in status['models'].values())
        return status