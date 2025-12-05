import logging
from typing import List

from api.core.base_service import BaseService

from .wrapper import JobMatcherModel

logger = logging.getLogger(__name__)


class JobMatcherService(BaseService):
    def __init__(self):
        super().__init__()
        self.matcher = self.get_model('job_matcher', JobMatcherModel)

    def match(self, resume_text: str, jobs: List[dict], top_k: int = 10) -> List[dict]:
        if not self.matcher or not self.matcher.is_loaded():
            logger.warning("[JobMatcherService] model unavailable")
            return []
        if not jobs:
            return []

        try:
            resume_embedding = self.matcher.encode_single(resume_text)
            job_texts = [f"{job['title']}. {job['description']}" for job in jobs]
            job_embeddings = self.matcher.encode(job_texts)
            similarities = self.matcher.similarity(resume_embedding, job_embeddings)

            matches = []
            for idx, job in enumerate(jobs):
                base_score = float(similarities[idx]) * 100 if similarities.size else 0.0
                skill_score = self._skill_match(resume_text, job.get('required_skills', []))
                total = 0.7 * base_score + 0.3 * (skill_score * 100)
                matches.append({
                    'job_id': job['job_id'],
                    'match_score': round(total, 2),
                    'skill_match': round(skill_score * 100, 2),
                    'experience_match': round(base_score, 2),
                    'education_match': True,
                    'location_match': True,
                })

            matches.sort(key=lambda item: item['match_score'], reverse=True)
            return matches[:top_k]
        except Exception as exc:
            logger.error("[JobMatcherService] failure: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _skill_match(resume_text: str, required_skills: List[str]) -> float:
        if not required_skills:
            return 0.5
        resume_lower = resume_text.lower()
        hits = sum(1 for skill in required_skills if skill.lower() in resume_lower)
        return hits / len(required_skills)
    

    def get_embedding(self, text: str) -> List[float]:
        if not self.matcher or not self.matcher.is_loaded():
            logger.warning("[JobMatcherService] model unavailable")
            return []
        try:
            embedding = self.matcher.encode_single(text)
            return embedding.tolist()
        except Exception as exc:
            logger.error("[JobMatcherService] get_embedding failure: %s", exc, exc_info=True)
            return []
