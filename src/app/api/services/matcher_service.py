import logging
from api.models.loader import ModelLoader

logger = logging.getLogger(__name__)


class MatcherService:
    """Service for matching resumes with job descriptions"""
    
    def __init__(self):
        self.loader = ModelLoader.get_instance()
        # Load the complete JobMatcher wrapper
        self.model = self.loader.get_model('job_matcher')
    
    def match_jobs(self, resume_text: str, jobs: list, top_k: int = 10) -> list:
        """
        Match resume with job descriptions using JobMatcher.encode() and cosine similarity
        
        Args:
            resume_text: Resume text
            jobs: List of job dicts with keys: job_id, title, description, 
                  required_skills, location
            top_k: Number of top matches to return
            
        Returns:
            List of job matches sorted by score
        """
        if not self.model or not self.model.is_loaded():
            logger.warning("Job matcher model not loaded")
            return []
        
        try:
            # Use the wrapper's encode methods
            resume_embedding = self.model.encode_single(resume_text)
            
            # Encode all job descriptions
            job_texts = [f"{job['title']}. {job['description']}" for job in jobs]
            job_embeddings = self.model.encode(job_texts)
            
            # Calculate similarities
            similarities = self.model.compute_similarity(resume_embedding, job_embeddings)
            
            # Build matches with additional scoring
            matches = []
            for i, job in enumerate(jobs):
                base_score = float(similarities[i]) * 100
                
                # Add skill match bonus
                skill_score = self._calculate_skill_match(
                    resume_text, 
                    job.get('required_skills', [])
                )
                
                # Weighted total score
                total_score = 0.7 * base_score + 0.3 * (skill_score * 100)
                
                matches.append({
                    'job_id': job['job_id'],
                    'match_score': round(total_score, 2),
                    'skill_match': round(skill_score * 100, 2),
                    'experience_match': round(base_score, 2),  # Use embedding similarity
                    'education_match': True,  # Placeholder
                    'location_match': True    # Placeholder
                })
            
            # Sort by score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return matches[:top_k]
        
        except Exception as e:
            logger.error(f"Error in job matching: {e}", exc_info=True)
            return []
    
    def _calculate_skill_match(self, resume_text: str, required_skills: list) -> float:
        """
        Calculate skill match score (Jaccard similarity)
        
        Args:
            resume_text: Resume text
            required_skills: List of required skills
            
        Returns:
            Skill match score (0-1)
        """
        if not required_skills:
            return 0.5
        
        resume_text_lower = resume_text.lower()
        
        matched_skills = sum(
            1 for skill in required_skills 
            if skill.lower() in resume_text_lower
        )
        
        return matched_skills / len(required_skills) if required_skills else 0.0