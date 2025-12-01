import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .patterns import EntityPatterns, EntityMatch

logger = logging.getLogger(__name__)

# Try to import spacy
try:
    import spacy
    from spacy.tokens import DocBin
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Using regex-only extraction.")


@dataclass
class ExtractedEntities:
    """Container for extracted entities"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    
    companies: List[str] = field(default_factory=list)
    job_titles: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    degrees: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    
    raw_entities: List[EntityMatch] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "linkedin": self.linkedin,
            "github": self.github,
            "companies": self.companies,
            "job_titles": self.job_titles,
            "institutions": self.institutions,
            "degrees": self.degrees,
            "skills": self.skills,
            "dates": self.dates,
            "locations": self.locations,
        }


class EntityExtractor:
    """
    Entity Extractor for Resumes
    
    Combines:
    - spaCy NER (for PERSON, ORG, GPE, DATE)
    - Custom regex patterns (for EMAIL, PHONE, LINKEDIN, etc.)
    """
    
    ENTITY_LABELS = [
        "PERSON", "EMAIL", "PHONE", "LINKEDIN", "GITHUB",
        "COMPANY", "JOB_TITLE", "INSTITUTION", "DEGREE",
        "SKILL", "DATE", "LOCATION"
    ]
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_spacy: bool = True,
        use_regex: bool = True
    ):
        self.spacy_model_name = spacy_model
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_regex = use_regex
        
        self.nlp = None
        self.patterns = EntityPatterns()
        self.custom_ner = None  # Custom trained NER
        
        if self.use_spacy:
            self._load_spacy()
    
    def _load_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load(self.spacy_model_name)
            logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
        except OSError:
            logger.warning(f"spaCy model '{self.spacy_model_name}' not found. Downloading...")
            spacy.cli.download(self.spacy_model_name)
            self.nlp = spacy.load(self.spacy_model_name)
    
    def _extract_with_spacy(self, text: str) -> List[EntityMatch]:
        """Extract entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Map spaCy labels to our labels
        label_map = {
            "PERSON": "PERSON",
            "ORG": "COMPANY",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "NORP": "LOCATION",  # Nationalities, religious, political groups
            "FAC": "LOCATION"    # Buildings, airports, highways, bridges, etc.
        }
        
        for ent in doc.ents:
            if ent.label_ in label_map:
                entities.append(EntityMatch(
                    text=ent.text,
                    label=label_map[ent.label_],
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8
                ))
        
        return entities
    
    def _extract_with_regex(self, text: str) -> List[EntityMatch]:
        """Extract entities using regex patterns"""
        return self.patterns.extract_all(text)
    
    def _extract_name(self, text: str, entities: List[EntityMatch]) -> Optional[str]:
        """Extract person name (usually first PERSON entity or first line)"""
        # Try spaCy PERSON entities first
        for e in entities:
            if e.label == "PERSON":
                return e.text
        
        # Fallback: first line that looks like a name
        lines = text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Simple heuristic: 2-4 words, no special characters
            words = first_line.split()
            if 2 <= len(words) <= 4:
                if all(w[0].isupper() and w.isalpha() for w in words):
                    return first_line
        
        return None
    
    def _merge_entities(
        self, 
        spacy_entities: List[EntityMatch], 
        regex_entities: List[EntityMatch]
    ) -> List[EntityMatch]:
        """Merge and deduplicate entities from both sources"""
        all_entities = spacy_entities + regex_entities
        
        if not all_entities:
            return []
        
        # Sort by start position
        all_entities.sort(key=lambda e: (e.start, -e.confidence))
        
        # Remove duplicates and overlapping
        result = []
        seen_texts = set()
        last_end = -1
        
        for entity in all_entities:
            text_lower = entity.text.lower().strip()
            
            # Skip if duplicate text or overlapping
            if text_lower in seen_texts:
                continue
            if entity.start < last_end:
                continue
            
            result.append(entity)
            seen_texts.add(text_lower)
            last_end = entity.end
        
        return result
    
    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from text
        
        Args:
            text: Resume text
            
        Returns:
            ExtractedEntities object
        """
        # Get entities from both sources
        spacy_entities = self._extract_with_spacy(text) if self.use_spacy else []
        regex_entities = self._extract_with_regex(text) if self.use_regex else []
        
        # Merge entities
        all_entities = self._merge_entities(spacy_entities, regex_entities)
        
        # Build result
        result = ExtractedEntities(raw_entities=all_entities)
        
        # Extract name
        result.name = self._extract_name(text, all_entities)
        
        # Organize by type
        for entity in all_entities:
            if entity.label == "EMAIL" and not result.email:
                result.email = entity.text
            elif entity.label == "PHONE" and not result.phone:
                result.phone = entity.text
            elif entity.label == "LINKEDIN" and not result.linkedin:
                result.linkedin = entity.text
            elif entity.label == "GITHUB" and not result.github:
                result.github = entity.text
            elif entity.label == "COMPANY":
                if entity.text not in result.companies:
                    result.companies.append(entity.text)
            elif entity.label == "JOB_TITLE":
                if entity.text not in result.job_titles:
                    result.job_titles.append(entity.text)
            elif entity.label == "INSTITUTION":
                if entity.text not in result.institutions:
                    result.institutions.append(entity.text)
            elif entity.label == "DEGREE":
                if entity.text not in result.degrees:
                    result.degrees.append(entity.text)
            elif entity.label == "SKILL":
                if entity.text not in result.skills:
                    result.skills.append(entity.text)
            elif entity.label == "DATE":
                if entity.text not in result.dates:
                    result.dates.append(entity.text)
            elif entity.label == "LOCATION":
                if entity.text not in result.locations:
                    result.locations.append(entity.text)
        
        return result
    
    def extract_batch(self, texts: List[str]) -> List[ExtractedEntities]:
        """Extract entities from multiple texts"""
        return [self.extract(text) for text in texts]
    
    def load_custom_ner(self, path: Union[str, Path]) -> None:
        """Load custom trained NER model"""
        path = Path(path)
        if path.exists() and SPACY_AVAILABLE:
            self.custom_ner = spacy.load(path)
            logger.info(f"Loaded custom NER from {path}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save extractor config (custom NER saved separately during training)"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        import json
        config = {
            "spacy_model": self.spacy_model_name,
            "use_spacy": self.use_spacy,
            "use_regex": self.use_regex,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "EntityExtractor":
        """Load extractor from path"""
        path = Path(path)
        
        import json
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        extractor = cls(**config)
        
        # Load custom NER if exists
        custom_ner_path = path / "spacy_model"
        if custom_ner_path.exists():
            extractor.load_custom_ner(custom_ner_path)
        
        return extractor


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = EntityExtractor()
    
    test_text = """
    John Doe
    john.doe@gmail.com | (555) 123-4567
    linkedin.com/in/johndoe | github.com/johndoe
    San Francisco, CA
    
    EXPERIENCE
    Senior Software Engineer at Google, 2020 - Present
    - Built scalable ML systems
    
    Software Engineer at Microsoft, Jan 2018 - Dec 2019
    - Developed cloud infrastructure
    
    EDUCATION
    Bachelor of Science in Computer Science
    Stanford University, 2017, GPA: 3.85/4.0
    """
    
    result = extractor.extract(test_text)
    
    print(f"Name: {result.name}")
    print(f"Email: {result.email}")
    print(f"Phone: {result.phone}")
    print(f"LinkedIn: {result.linkedin}")
    print(f"Companies: {result.companies}")
    print(f"Job Titles: {result.job_titles}")
    print(f"Skills: {result.skills}")
    print(f"Institutions: {result.institutions}")
    print(f"Dates: {result.dates}")