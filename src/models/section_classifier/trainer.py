import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from .model import SectionClassifier
from .rules import SectionRules

logger = logging.getLogger(__name__)


class SectionClassifierTrainer:
    """
    Trainer for Section Classifier
    Generates training data and trains the model
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("data/artifacts/section_classifier"),
        algorithm: str = "logistic_regression",
        rule_weight: float = 0.3,
        tfidf_max_features: int = 3000
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.algorithm = algorithm
        self.rule_weight = rule_weight
        self.tfidf_max_features = tfidf_max_features
        
        self.rules = SectionRules()
        self.model: Optional[SectionClassifier] = None
    
    def generate_data_from_resume_csv(
        self,
        csv_path: Optional[Path] = None,
        text_column: str = "Resume_text"
    ) -> Tuple[List[str], List[str]]:
        """Generate training data from resume CSV using rule-based segmentation"""
        
        if csv_path is None:
            csv_path = self.data_dir / "processed" / "Resume.csv"
        
        if not csv_path.exists():
            logger.warning(f"Resume CSV not found: {csv_path}")
            return [], []
        
        logger.info(f"Loading resumes from {csv_path}")
        df = pd.read_csv(csv_path)
        
        texts, labels = [], []
        
        for _, row in df.iterrows():
            resume_text = str(row.get(text_column, ""))
            if not resume_text or resume_text == "nan":
                continue
            
            segments = self.rules.segment_resume(resume_text)
            
            for section, content in segments.items():
                if content and len(content.strip()) > 20:
                    texts.append(content.strip())
                    labels.append(section)
        
        logger.info(f"Generated {len(texts)} samples from resume CSV")
        return texts, labels
    
    def generate_data_from_structured(
        self,
        structured_dir: Optional[Path] = None
    ) -> Tuple[List[str], List[str]]:
        """Generate training data from structured CSVs"""
        
        if structured_dir is None:
            structured_dir = self.data_dir / "raw" / "structured"
        
        if not structured_dir.exists():
            logger.warning(f"Structured dir not found: {structured_dir}")
            return [], []
        
        texts, labels = [], []
        
        # Education
        edu_path = structured_dir / "education.csv"
        if edu_path.exists():
            education_df = pd.read_csv(edu_path)
            for _, row in education_df.iterrows():
                institution = str(row.get("institution", ""))
                program = str(row.get("program", ""))
                start_date = str(row.get("start_date", ""))
                location = str(row.get("location", ""))
                
                parts = [program, institution]
                if start_date:
                    parts.append(f"Started in {start_date}")
                if location:
                    parts.append(f"in {location}")
                
                text = ' '.join(parts)
                if len(text) > 2:
                    texts.append(text)
                    labels.append("education")
            logger.info(f"Added {len(education_df)} education samples")
        
        # Experience
        exp_path = structured_dir / "experience.csv"
        if exp_path.exists():
            experience_df = pd.read_csv(exp_path)
            for _, row in experience_df.iterrows():
                title = str(row.get("title", ""))
                firm = str(row.get("firm", ""))
                start_date = str(row.get("start_date", ""))
                end_date = str(row.get("end_date", ""))
                location = str(row.get("location", ""))
                
                parts = [title]
                if firm:
                    parts.append(f"worked at {firm}")
                if start_date or end_date:
                    date_range = f"{start_date} - {end_date}".strip(" -")
                    parts.append(f"({date_range})")
                if location:
                    parts.append(f"in {location}")
                
                text = ' '.join(parts)
                if len(text) > 2:
                    texts.append(text)
                    labels.append("experience")
            logger.info(f"Added {len(experience_df)} experience samples")
        
        # People / Contact
        people_path = structured_dir / "people.csv"
        if people_path.exists():
            people_df = pd.read_csv(people_path)
            for _, row in people_df.iterrows():
                email = str(row.get("email", ""))
                phone = str(row.get("phone", ""))
                linkedin = str(row.get("linkedin", ""))
                
                parts = []
                if email:
                    parts.append(f"Email: {email}")
                if phone:
                    parts.append(f"Phone: {phone}")
                if linkedin:
                    parts.append(f"LinkedIn: {linkedin}")
                
                text = ' '.join(parts)
                if len(text) > 0:
                    texts.append(text)
                    labels.append("contact")
            logger.info(f"Added {len(people_df)} contact samples")

        # Person Skills
        person_skills_path = structured_dir / "person_skills.csv"
        if person_skills_path.exists():
            person_skills_df = pd.read_csv(person_skills_path)
            for _, row in person_skills_df.iterrows():
                skill = str(row.get("skill", ""))
                texts.append(skill)
                labels.append("person_skills")
            logger.info(f"Added {len(person_skills_df)} person skills samples")

        # Skills
        skills_path = structured_dir / "skills.csv"
        if skills_path.exists():
            skills_df = pd.read_csv(skills_path)
            for _, row in skills_df.iterrows():
                skill = str(row.get("skill", ""))
                texts.append(skill)
                labels.append("skills")
            logger.info(f"Added {len(skills_df)} skills samples")

        # Abilities
        abilities_path = structured_dir / "abilities.csv"
        if abilities_path.exists():
            df = pd.read_csv(abilities_path)
            for _, row in df.iterrows():
                ability = str(row.get("ability", ""))
                texts.append(ability)
                labels.append("skills")
            logger.info(f"Added {len(df)} abilities samples")

        return texts, labels
    
    def train(
        self,
        use_resume_csv: bool = True,
        use_structured: bool = True,
        use_synthetic: bool = True,
        augment_factor: int = 5
    ) -> Dict[str, float]:
        """
        Train the section classifier
        
        Args:
            use_resume_csv: Use data from Resume.csv
            use_structured: Use data from structured CSVs
            use_synthetic: Use synthetic data
            augment_factor: Multiply synthetic data for balance
        """
        logger.info("=" * 60)
        logger.info("Training Section Classifier")
        logger.info("=" * 60)
        
        texts, labels = [], []
        
        # Collect data from all sources
        if use_resume_csv:
            t, l = self.generate_data_from_resume_csv()
            texts.extend(t)
            labels.extend(l)
        
        if use_structured:
            t, l = self.generate_data_from_structured()
            texts.extend(t)
            labels.extend(l)
        
        if not texts:
            logger.error("No training data found!")
            raise ValueError("No training data available")
        
        # Log distribution
        label_counts = pd.Series(labels).value_counts()
        logger.info(f"Total samples: {len(texts)}")
        logger.info(f"Label distribution:\n{label_counts}")
        
        # Create and train model
        self.model = SectionClassifier(
            algorithm=self.algorithm,
            tfidf_max_features=self.tfidf_max_features,
            rule_weight=self.rule_weight
        )
        
        metrics = self.model.fit(texts, labels)
        
        # Save model
        self.model.save(self.artifacts_dir)
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to {self.artifacts_dir}")
        logger.info("=" * 60)
        
        return metrics
    
    def test(self, texts: Optional[List[str]] = None):
        """Test the trained model"""
        if self.model is None:
            self.model = SectionClassifier.load(self.artifacts_dir)
        
        if texts is None:
            texts = [
                "john.doe@email.com, phone: 555-1234",
                "Bachelor of Science in Computer Science from MIT, GPA 3.8",
                "Software Engineer at Google, 2020-2023.",
                "Python, Java, TensorFlow, PyTorch, SQL, AWS, Docker",
                "Experienced professional seeking new opportunities"
            ]
        
        logger.info("\nTest Predictions:")
        for text in texts:
            pred = self.model.predict(text)
            logger.info(f"  '{text[:50]}...' -> {pred.section} ({pred.confidence:.2f})")


# Entry point for training
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = SectionClassifierTrainer()
    # trainer.train()
    trainer.test()