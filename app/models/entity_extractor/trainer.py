"""
Entity Extractor Trainer
Loads skills from dataset for lookup matching
No NER training needed - uses pre-trained spaCy
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .model import EntityExtractor, ExtractedEntities

logger = logging.getLogger(__name__)


class EntityExtractorTrainer:
    """
    Prepares EntityExtractor by loading skills/companies/etc from dataset
    
    No actual training - just builds lookup sets from your CSV
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("data/artifacts/entity_extractor"),
        spacy_model: str = "en_core_web_sm",
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.spacy_model = spacy_model
        self.extractor: Optional[EntityExtractor] = None
    
    def load_skills_from_csv(
        self,
        csv_path: Optional[Path] = None,
        skills_columns: List[str] = None,
    ) -> Set[str]:
        """
        Load skills from CSV dataset
        
        Args:
            csv_path: Path to CSV with skills
            skills_columns: Columns containing skills
        """
        if csv_path is None:
            # Try common paths
            candidates = [
                self.data_dir / "raw" / "skills.csv",
                self.data_dir / "processed" / "skills.csv",
                self.data_dir / "skills.csv",
            ]
            for path in candidates:
                if path.exists():
                    csv_path = path
                    break
        
        if csv_path is None or not Path(csv_path).exists():
            logger.warning("No skills CSV found, using default skills")
            return set()
        
        logger.info(f"Loading skills from {csv_path}")
        
        skills = set()
        
        try:
            df = pd.read_csv(csv_path)
            
            # If no columns specified, try to find skill-related columns
            if skills_columns is None:
                skills_columns = [c for c in df.columns if 'skill' in c.lower()]
                if not skills_columns:
                    skills_columns = df.columns.tolist()
            
            for col in skills_columns:
                if col in df.columns:
                    for value in df[col].dropna():
                        # Handle comma-separated skills
                        if isinstance(value, str):
                            for skill in value.split(','):
                                skill = skill.strip().lower()
                                if skill and len(skill) > 1:
                                    skills.add(skill)
            
            logger.info(f"Loaded {len(skills)} unique skills")
            
        except Exception as e:
            logger.error(f"Error loading skills: {e}")
        
        return skills
    
    def load_from_resume_dataset(
        self,
        csv_path: Optional[Path] = None,
    ) -> Set[str]:
        """
        Load skills from your resume dataset columns:
        - skills
        - related_skils_in_job
        - skills_required
        - certification_skills
        """
        if csv_path is None:
            csv_path = self.data_dir / "raw" / "structured" / "resume_data.csv"
        
        if not Path(csv_path).exists():
            logger.warning(f"Dataset not found: {csv_path}")
            return set()
        
        logger.info(f"Loading from resume dataset: {csv_path}")
        
        skills = set()
        
        try:
            df = pd.read_csv(csv_path)
            
            skill_columns = [
                'skills',
                'related_skils_in_job',
                'skills_required',
                'certification_skills',
            ]
            
            for col in skill_columns:
                if col in df.columns:
                    for value in df[col].dropna():
                        if isinstance(value, str):
                            # Handle various separators
                            for sep in [',', ';', '|', '\n']:
                                value = value.replace(sep, ',')
                            for skill in value.split(','):
                                skill = skill.strip().lower()
                                if skill and 1 < len(skill) < 50:
                                    skills.add(skill)
            
            logger.info(f"Extracted {len(skills)} unique skills from dataset")
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
        
        return skills
    
    def train(
        self,
        skills_csv: Optional[Path] = None,
        resume_dataset_csv: Optional[Path] = None,
    ) -> EntityExtractor:
        """
        Build EntityExtractor with skills from dataset
        
        Args:
            skills_csv: Optional path to skills.csv
            resume_dataset_csv: Optional path to resume dataset
        """
        logger.info("="*60)
        logger.info("Building Entity Extractor")
        logger.info("="*60)
        
        # Collect skills from all sources
        all_skills = set()
        
        # From skills.csv
        if skills_csv:
            all_skills.update(self.load_skills_from_csv(skills_csv))
        
        # From resume dataset
        if resume_dataset_csv:
            all_skills.update(self.load_from_resume_dataset(resume_dataset_csv))
        else:
            # Try default path
            all_skills.update(self.load_from_resume_dataset())
        
        logger.info(f"Total skills: {len(all_skills)}")
        
        # Create extractor
        self.extractor = EntityExtractor(
            spacy_model=self.spacy_model,
            skills_list=list(all_skills),
        )
        self.extractor.save(self.artifacts_dir)
        
        logger.info("="*60)
        logger.info(f"Entity Extractor saved to {self.artifacts_dir}")
        logger.info("="*60)
        
        return self.extractor
    
    def test(self, texts: Optional[List[str]] = None):
        """Test extractor on sample texts"""
        if self.extractor is None:
            try:
                self.extractor = EntityExtractor.load(str(self.artifacts_dir), self.spacy_model)
            except FileNotFoundError:
                logger.error(
                    "No persisted EntityExtractor at %s. Train or provide skills before testing.",
                    self.artifacts_dir,
                )
                return
        
        if texts is None:
            texts = [
                """
                Jane Smith
                jane.smith@gmail.com | (415) 555-1234
                San Francisco, CA
                linkedin.com/in/janesmith
                
                EXPERIENCE
                
                Senior Data Scientist at Google
                Jan 2021 - Present
                
                Machine Learning Engineer at Meta
                2019 - 2020
                
                EDUCATION
                
                Ph.D. in Computer Science
                Stanford University, 2019
                GPA: 3.95
                
                SKILLS
                Python, TensorFlow, PyTorch, SQL, AWS
                
                CERTIFICATIONS
                AWS Machine Learning Specialty
                
                LANGUAGES
                English, Spanish, Mandarin
                """,
            ]
        
        logger.info("\nTest Extractions:")
        logger.info("-" * 50)
        
        for i, text in enumerate(texts):
            result = self.extractor.extract(text)
            
            logger.info(f"\n--- Resume {i+1} ---")
            logger.info(f"Name: {result.name}")
            logger.info(f"Email: {result.email}")
            logger.info(f"Phone: {result.phone}")
            logger.info(f"Job Titles: {result.job_titles}")
            logger.info(f"Companies: {result.companies}")
            logger.info(f"Institutions: {result.institutions}")
            logger.info(f"Degrees: {result.degrees}")
            logger.info(f"Skills: {result.skills[:10] if result.skills else []}")
            logger.info(f"Languages: {result.languages}")


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = EntityExtractorTrainer()
    trainer.train()
    trainer.test()