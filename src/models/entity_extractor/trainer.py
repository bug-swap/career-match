"""
Entity Extractor Trainer
Generates training data and trains custom spaCy NER
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from .model import EntityExtractor, SPACY_AVAILABLE

if SPACY_AVAILABLE:
    import spacy
    from spacy.tokens import DocBin
    from spacy.training import Example

logger = logging.getLogger(__name__)


class EntityExtractorTrainer:
    """
    Trainer for Entity Extractor
    Generates training data from structured CSVs and trains spaCy NER
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        artifacts_dir: Path = Path("data/artifacts/entity_extractor"),
        base_model: str = "en_core_web_sm",
        n_iter: int = 20
    ):
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.base_model = base_model
        self.n_iter = n_iter
        
        self.extractor: Optional[EntityExtractor] = None
    
    def generate_training_data_from_structured(
        self,
        structured_dir: Optional[Path] = None,
        max_per_source: int = 2000
    ) -> List[Tuple[str, Dict]]:
        """
        Generate NER training data from structured CSVs
        
        Args:
            structured_dir: Path to structured CSVs
            max_per_source: Max samples per CSV file
        
        Returns:
            List of (text, {"entities": [(start, end, label), ...]})
        """
        if structured_dir is None:
            structured_dir = self.data_dir / "raw" / "structured"
        
        training_data = []
        
        # People CSV -> PERSON, EMAIL, PHONE, LINKEDIN
        people_path = structured_dir / "people.csv"
        if people_path.exists():
            df = pd.read_csv(people_path)
            if len(df) > max_per_source:
                df = df.sample(n=max_per_source, random_state=42)
            
            for _, row in df.iterrows():
                name = str(row.get("name", ""))
                email = str(row.get("email", ""))
                phone = str(row.get("phone", ""))
                linkedin = str(row.get("linkedin", ""))
                
                if name and name != "nan":
                    # Create sentence with annotations
                    text = f"{name}"
                    entities = [(0, len(name), "PERSON")]
                    
                    if email and email != "nan":
                        start = len(text) + 3
                        text += f" | {email}"
                        entities.append((start, start + len(email), "EMAIL"))
                    
                    if phone and phone != "nan":
                        start = len(text) + 3
                        text += f" | {phone}"
                        entities.append((start, start + len(phone), "PHONE"))
                    
                    training_data.append((text, {"entities": entities}))
            
            logger.info(f"Generated {len(df)} samples from people.csv")
        
        # Education CSV -> INSTITUTION, DEGREE
        edu_path = structured_dir / "education.csv"
        if edu_path.exists():
            df = pd.read_csv(edu_path)
            if len(df) > max_per_source:
                df = df.sample(n=max_per_source, random_state=42)
            
            for _, row in df.iterrows():
                institution = str(row.get("institution", ""))
                program = str(row.get("program", ""))
                location = str(row.get("location", ""))
                
                if institution and institution != "nan":
                    entities = []
                    text = institution
                    entities.append((0, len(institution), "INSTITUTION"))
                    
                    if program and program != "nan":
                        start = len(text) + 2
                        text += f", {program}"
                        entities.append((start, start + len(program), "DEGREE"))
                    
                    if location and location != "nan":
                        start = len(text) + 2
                        text += f", {location}"
                        entities.append((start, start + len(location), "LOCATION"))
                    
                    training_data.append((text, {"entities": entities}))
            
            logger.info(f"Generated {len(df)} samples from education.csv")
        
        # Experience CSV -> COMPANY, JOB_TITLE
        exp_path = structured_dir / "experience.csv"
        if exp_path.exists():
            df = pd.read_csv(exp_path)
            if len(df) > max_per_source:
                df = df.sample(n=max_per_source, random_state=42)
            
            for _, row in df.iterrows():
                title = str(row.get("title", ""))
                firm = str(row.get("firm", ""))
                location = str(row.get("location", ""))
                
                if title and title != "nan" and firm and firm != "nan":
                    # "Software Engineer at Google"
                    text = f"{title} at {firm}"
                    entities = [
                        (0, len(title), "JOB_TITLE"),
                        (len(title) + 4, len(title) + 4 + len(firm), "COMPANY")
                    ]
                    
                    if location and location != "nan":
                        start = len(text) + 2
                        text += f", {location}"
                        entities.append((start, start + len(location), "LOCATION"))
                    
                    training_data.append((text, {"entities": entities}))
            
            logger.info(f"Generated {len(df)} samples from experience.csv")
        
        # Skills CSV -> SKILL
        skills_path = structured_dir / "skills.csv"
        person_skills_path = structured_dir / "person_skills.csv"
        abilities_path = structured_dir / "abilities.csv"
        skill_sources = [skills_path, person_skills_path, abilities_path]
        for skill_path in skill_sources:
            if skill_path.exists():
                df = pd.read_csv(skill_path)
                if len(df) > max_per_source:
                    df = df.sample(n=max_per_source, random_state=42)
                
                for _, row in df.iterrows():
                    skill = str(row.get("skill", ""))
                    if skill and skill != "nan":
                        text = skill
                        entities = [(0, len(skill), "SKILL")]
                        training_data.append((text, {"entities": entities}))
                
                logger.info(f"Generated {len(df)} samples from {skill_path.name}")

        
        return training_data
    
    def train_spacy_ner(
        self,
        training_data: List[Tuple[str, Dict]],
        output_dir: Optional[Path] = None
    ) -> None:
        """Train custom spaCy NER model"""
        
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available. Cannot train NER.")
            return
        
        if output_dir is None:
            output_dir = self.artifacts_dir / "spacy_model"
        
        logger.info(f"Training spaCy NER with {len(training_data)} samples")
        
        # Load base model
        nlp = spacy.load(self.base_model)
        
        # Get or create NER pipe
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        # Add labels
        for _, annotations in training_data:
            for start, end, label in annotations.get("entities", []):
                ner.add_label(label)
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.begin_training()
            
            for iteration in range(self.n_iter):
                random.shuffle(training_data)
                losses = {}
                
                # Batch training
                for text, annotations in training_data:
                    try:
                        doc = nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        nlp.update([example], drop=0.35, losses=losses)
                    except Exception as e:
                        logger.warning(f"Skipping sample: {e}")
                        continue
                
                logger.info(f"Iteration {iteration}, Loss: {losses.get('ner', 0):.4f}")
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    def train(
        self,
        use_structured: bool = True,
        use_synthetic: bool = True,
        max_samples: int = 20000,
        sample_divider: int = 6
    ) -> None:
        """
        Train the entity extractor
        
        Args:
            use_structured: Use data from structured CSVs
            use_synthetic: Use synthetic data
            max_samples: Maximum training samples (samples if exceeded)
        """
        logger.info("=" * 60)
        logger.info("Training Entity Extractor")
        logger.info("=" * 60)
        
        training_data = []
        
        if use_structured:
            data = self.generate_training_data_from_structured(max_per_source= max_samples // sample_divider)
            training_data.extend(data)
        
        if not training_data:
            logger.error("No training data found!")
            raise ValueError("No training data available")
        
        logger.info(f"Total samples before sampling: {len(training_data)}")
        
        # Sample if too many
        if len(training_data) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(training_data)} samples")
            training_data = random.sample(training_data, max_samples)
        
        logger.info(f"Training samples: {len(training_data)}")
        
        # Train spaCy NER
        if SPACY_AVAILABLE:
            self.train_spacy_ner(training_data)
        else:
            logger.warning("spaCy not available. Skipping NER training.")
        
        # Save extractor config
        self.extractor = EntityExtractor()
        self.extractor.save(self.artifacts_dir)
        
        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to {self.artifacts_dir}")
        logger.info("=" * 60)
    
    def test(self, texts: Optional[List[str]] = None):
        """Test the trained model"""
        if self.extractor is None:
            self.extractor = EntityExtractor.load(self.artifacts_dir)
        
        # Load custom NER
        custom_ner_path = self.artifacts_dir / "spacy_model"
        if custom_ner_path.exists():
            self.extractor.load_custom_ner(custom_ner_path)
        
        if texts is None:
            texts = [
                "John Doe | john@email.com | (555) 123-4567,Software Engineer at Google, 2020-Present, Bachelor of Science from MIT, 2018, GPA 3.8, Works with Python, Java",
                "Jane Smith | jane@email.com | (555) 987-6543,Data Scientist at Facebook, 2019-Present, Master of Science from Stanford, 2017, GPA 3.9, Works with R, Python",
                "Alice Johnson | alice@email.com | (555) 555-5555,Product Manager at Amazon, 2018-Present, MBA from Harvard, 2016, GPA 3.7, Works with Agile, Scrum",
            ]
        
        logger.info("\nTest Extractions:")
        for text in texts:
            result = self.extractor.extract(text)
            logger.info(f"\nInput: '{text}'")
            logger.info(f"  Name: {result.name}")
            logger.info(f"  Email: {result.email}")
            logger.info(f"  Phone: {result.phone}")
            logger.info(f"  LinkedIn: {result.linkedin}")
            logger.info(f"  Companies: {result.companies}")
            logger.info(f"  Job Titles: {result.job_titles}")
            logger.info(f"  Institutions: {result.institutions}")
            logger.info(f"  Degrees: {result.degrees}")
            logger.info(f"  Skills: {result.skills}")


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    trainer = EntityExtractorTrainer()
    trainer.train(max_samples=20_000)
    trainer.test()