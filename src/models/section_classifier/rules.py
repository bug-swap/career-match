import re
from typing import Dict, List, Tuple


class SectionRules:
    """Rule-based section classification using keyword patterns"""
    
    SECTION_KEYWORDS: Dict[str, List[str]] = {
        "contact": [
            "email", "phone", "address", "linkedin", "github",
            "portfolio", "website", "mobile", "tel", "contact"
        ],
        "summary": [
            "summary", "objective", "profile", "about me", "overview",
            "professional summary", "career objective", "personal statement"
        ],
        "education": [
            "education", "university", "college", "school", "degree",
            "bachelor", "master", "phd", "diploma", "gpa", "graduate",
            "b.s.", "b.a.", "m.s.", "m.a.", "mba"
        ],
        "experience": [
            "experience", "employment", "work history", "professional experience",
            "work experience", "positions", "responsibilities", "achievements"
        ],
        "skills": [
            "skills", "technical skills", "competencies", "expertise",
            "technologies", "tools", "programming", "software", "frameworks"
        ],
        "projects": [
            "projects", "personal projects", "academic projects", "portfolio"
        ],
        "certifications": [
            "certifications", "certificates", "licenses", "certified"
        ],
        "awards": [
            "awards", "honors", "achievements", "recognition", "scholarships"
        ],
        "other": [
            "interests", "hobbies", "volunteer", "activities", "references"
        ]
    }
    
    HEADER_PATTERNS = [
        r"^[A-Z][A-Z\s&]+$",
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?$",
    ]
    
    def __init__(self, weight: float = 0.3):
        self.weight = weight
        self._compile_patterns()
    
    def _compile_patterns(self):
        self.header_patterns = [re.compile(p) for p in self.HEADER_PATTERNS]
        self.keyword_patterns = {}
        for section, keywords in self.SECTION_KEYWORDS.items():
            patterns = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) 
                       for kw in keywords]
            self.keyword_patterns[section] = patterns
    
    def score_text(self, text: str) -> Dict[str, float]:
        """Score text for each section type"""
        text_lower = text.lower()
        scores = {}
        for section, patterns in self.keyword_patterns.items():
            matches = sum(1 for p in patterns if p.search(text_lower))
            scores[section] = min(matches / max(len(patterns) * 0.3, 1), 1.0)
        return scores
    
    def is_header(self, line: str) -> bool:
        """Check if line is a section header"""
        line = line.strip()
        if len(line) < 3 or len(line) > 50:
            return False
        return any(p.match(line) for p in self.header_patterns)
    
    def detect_header_section(self, header: str) -> Tuple[str, float]:
        """Detect section type from header text"""
        header_lower = header.lower().strip()
        for section in self.SECTION_KEYWORDS.keys():
            if section in header_lower:
                return section, 1.0
        scores = self.score_text(header)
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0.3:
                return best, scores[best]
        return "other", 0.5
    
    def segment_resume(self, text: str) -> Dict[str, str]:
        """Segment resume text into sections"""
        lines = text.split('\n')
        sections = {}
        current_section = "other"
        current_content = []
        
        for line in lines:
            if self.is_header(line):
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections[current_section] = sections.get(current_section, '') + '\n' + content
                current_section, _ = self.detect_header_section(line)
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections[current_section] = sections.get(current_section, '') + '\n' + content
        
        return {k: v.strip() for k, v in sections.items()}