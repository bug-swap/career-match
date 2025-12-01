import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EntityMatch:
    """Single entity match"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.9


class EntityPatterns:
    """Regex-based entity extraction patterns"""
    
    PATTERNS: Dict[str, List[str]] = {
        "EMAIL": [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        ],
        "PHONE": [
            r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            r"\+\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}",
            r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}",
        ],
        "LINKEDIN": [
            r"linkedin\.com/in/[\w-]+",
            r"linkedin\.com/pub/[\w/-]+",
        ],
        "GITHUB": [
            r"github\.com/[\w-]+",
        ],
        "URL": [
            r"https?://[^\s<>\"{}|\\^`\[\]]+",
        ],
        "DATE": [
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b",
            r"\b\d{1,2}/\d{4}\b",
            r"\b\d{4}\s*[-–]\s*(?:\d{4}|[Pp]resent|[Cc]urrent)\b",
            r"\b(?:19|20)\d{2}\b",
        ],
        "DEGREE": [
            r"\b(?:Bachelor|Master|Doctor|Ph\.?D\.?|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|MBA|M\.?Eng|B\.?Eng)\.?\s*(?:of\s+)?(?:Science|Arts|Engineering|Business|Administration)?\b",
            r"\bAssociate(?:'s)?\s+(?:Degree|of\s+\w+)\b",
        ],
        "GPA": [
            r"\bGPA:?\s*(\d\.\d{1,2})\s*(?:/\s*4\.0)?\b",
            r"\b(\d\.\d{1,2})\s*/\s*4\.0\s*GPA\b",
        ],
    }
    
    # Common job titles
    JOB_TITLES = [
        "Software Engineer", "Senior Software Engineer", "Staff Engineer",
        "Data Scientist", "Senior Data Scientist", "Machine Learning Engineer",
        "Product Manager", "Senior Product Manager", "Program Manager",
        "DevOps Engineer", "Site Reliability Engineer", "SRE",
        "Frontend Developer", "Backend Developer", "Full Stack Developer",
        "Data Analyst", "Business Analyst", "Data Engineer",
        "UX Designer", "UI Designer", "Product Designer",
        "Engineering Manager", "Technical Lead", "Tech Lead",
        "CTO", "CEO", "VP of Engineering", "Director of Engineering",
        "Intern", "Software Intern", "Research Intern",
        "Consultant", "Technical Consultant", "Solutions Architect",
        "QA Engineer", "Test Engineer", "Quality Assurance",
        "Project Manager", "Scrum Master", "Agile Coach",
    ]
    
    # Common companies
    KNOWN_COMPANIES = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Facebook",
        "Netflix", "Tesla", "Uber", "Lyft", "Airbnb", "Twitter",
        "LinkedIn", "Salesforce", "Adobe", "Oracle", "IBM", "Intel",
        "NVIDIA", "AMD", "Cisco", "VMware", "Dropbox", "Slack",
        "Spotify", "Pinterest", "Snap", "TikTok", "ByteDance",
        "Goldman Sachs", "Morgan Stanley", "JPMorgan", "Deloitte",
        "McKinsey", "BCG", "Bain", "Accenture", "PwC", "EY", "KPMG",
    ]
    
    # Common universities
    KNOWN_INSTITUTIONS = [
        "MIT", "Stanford", "Harvard", "Berkeley", "Caltech",
        "Carnegie Mellon", "Princeton", "Yale", "Columbia", "Cornell",
        "University of Michigan", "University of Texas", "UCLA", "USC",
        "Georgia Tech", "University of Washington", "NYU", "Duke",
        "Northwestern", "University of Illinois", "Penn State",
        "Ohio State", "University of Florida", "Texas A&M",
    ]
    
    def __init__(self, skills_list: Optional[List[str]] = None):
        self.skills_list = skills_list or []
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.compiled = {}
        for label, patterns in self.PATTERNS.items():
            self.compiled[label] = [re.compile(p, re.IGNORECASE) for p in patterns]

        # Compile job title pattern
        titles_escaped = [re.escape(t) for t in self.JOB_TITLES]
        self.job_title_pattern = re.compile(
            r"\b(" + "|".join(titles_escaped) + r")\b", re.IGNORECASE
        )

        # Compile company pattern
        companies_escaped = [re.escape(c) for c in self.KNOWN_COMPANIES]
        self.company_pattern = re.compile(
            r"\b(" + "|".join(companies_escaped) + r")\b", re.IGNORECASE
        )

        # Compile institution pattern
        institutions_escaped = [re.escape(i) for i in self.KNOWN_INSTITUTIONS]
        self.institution_pattern = re.compile(
            r"\b(" + "|".join(institutions_escaped) + r")\b", re.IGNORECASE
        )

        # Compile skills pattern if skills_list is provided
        if self.skills_list:
            # Sort by length descending to match longer skills first
            skills_sorted = sorted(set(self.skills_list), key=lambda x: -len(x))
            skills_escaped = [re.escape(skill.strip()) for skill in skills_sorted if skill.strip()]
            if skills_escaped:
                self.skills_pattern = re.compile(r"\\b(" + "|".join(skills_escaped) + r")\\b", re.IGNORECASE)
            else:
                self.skills_pattern = None
        else:
            self.skills_pattern = None
    
    def extract_all(self, text: str) -> List[EntityMatch]:
        """Extract all entities from text using regex and skills lookup"""
        entities = []

        # Pattern-based extraction
        for label, patterns in self.compiled.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(EntityMatch(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))

        # Job titles
        for match in self.job_title_pattern.finditer(text):
            entities.append(EntityMatch(
                text=match.group(),
                label="JOB_TITLE",
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))

        # Companies
        for match in self.company_pattern.finditer(text):
            entities.append(EntityMatch(
                text=match.group(),
                label="COMPANY",
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))

        # Institutions
        for match in self.institution_pattern.finditer(text):
            entities.append(EntityMatch(
                text=match.group(),
                label="INSTITUTION",
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))

        # Skills (from skills.csv)
        if self.skills_pattern:
            for match in self.skills_pattern.finditer(text):
                entities.append(EntityMatch(
                    text=match.group(),
                    label="SKILL",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))

        # Remove duplicates and overlapping
        entities = self._remove_overlapping(entities)

        return entities
    
    def _remove_overlapping(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return []
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        result = []
        last_end = -1
        
        for entity in entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
        
        return result
    
    def extract_by_type(self, text: str, entity_type: str) -> List[str]:
        """Extract entities of a specific type"""
        entities = self.extract_all(text)
        return [e.text for e in entities if e.label == entity_type]


# Quick test
if __name__ == "__main__":
    patterns = EntityPatterns()
    
    test_text = """
    John Doe
    john.doe@email.com | (555) 123-4567
    linkedin.com/in/johndoe | github.com/johndoe
    
    Software Engineer at Google, 2020 - Present
    Senior Developer at Microsoft, Jan 2018 - Dec 2019
    
    Bachelor of Science in Computer Science
    MIT, 2017, GPA: 3.85/4.0
    """
    
    entities = patterns.extract_all(test_text)
    for e in entities:
        print(f"{e.label}: '{e.text}' ({e.confidence:.2f})")