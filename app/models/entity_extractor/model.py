"""
Entity Extractor - Comprehensive Resume Parser
Regex: Structured data (email, phone, URLs)
spaCy: NER (names, orgs, locations, dates)
Section-aware: Context-based extraction
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")


@dataclass
class ExtractedEntities:
    """All extracted entities from resume"""
    
    # Contact Info
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    
    # Professional
    job_titles: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    work_dates: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    
    # Education
    degrees: List[str] = field(default_factory=list)
    majors: List[str] = field(default_factory=list)
    institutions: List[str] = field(default_factory=list)
    graduation_years: List[str] = field(default_factory=list)
    gpa: Optional[str] = None
    
    # Other
    certifications: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            # Contact
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "linkedin": self.linkedin,
            "github": self.github,
            "portfolio": self.portfolio,
            # Professional
            "job_titles": self.job_titles,
            "companies": self.companies,
            "work_dates": self.work_dates,
            "skills": self.skills,
            # Education
            "degrees": self.degrees,
            "majors": self.majors,
            "institutions": self.institutions,
            "graduation_years": self.graduation_years,
            "gpa": self.gpa,
            # Other
            "certifications": self.certifications,
            "projects": self.projects,
            "publications": self.publications,
            "languages": self.languages,
            "summary": self.summary,
        }


class EntityExtractor:
    """
    Comprehensive Resume Entity Extractor
    
    Extraction Strategy:
    1. Regex → email, phone, linkedin, github, urls, gpa
    2. spaCy NER → PERSON, ORG, GPE, DATE
    3. Section Detection → understand context for better extraction
    4. Skills Lookup → match against provided skills list
    """
    
    # ============ REGEX PATTERNS ============
    
    EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    PHONE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    LINKEDIN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.I)
    GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.I)
    URL = re.compile(r"https?://[^\s<>\"]+", re.I)
    GPA = re.compile(r"(?:GPA|CGPA)[:\s]*(\d\.\d{1,2})(?:\s*/\s*4\.?0?)?", re.I)
    
    # Date patterns
    DATE_RANGE = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|"
        r"\d{1,2}/\d{4}|\d{4})\s*[-–—to]+\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|"
        r"\d{1,2}/\d{4}|\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)",
        re.I
    )
    YEAR = re.compile(r"\b(19|20)\d{2}\b")
    
    # Degree patterns
    DEGREE = re.compile(
        r"\b(Bachelor|Master|Doctor|Ph\.?D\.?|B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|"
        r"MBA|M\.?Eng|B\.?Eng|B\.?Tech|M\.?Tech|B\.?Com|M\.?Com|Associate|Diploma|"
        r"High School|Secondary)\b[^.]*",
        re.I
    )
    
    # Section headers
    SECTION_PATTERNS = {
        "experience": re.compile(r"^(?:work\s+)?experience|employment|work\s+history|professional\s+experience", re.I | re.M),
        "education": re.compile(r"^education|academic|qualification", re.I | re.M),
        "skills": re.compile(r"^(?:technical\s+)?skills|technologies|competencies|expertise", re.I | re.M),
        "projects": re.compile(r"^projects|personal\s+projects|academic\s+projects", re.I | re.M),
        "certifications": re.compile(r"^certifications?|licenses?|credentials", re.I | re.M),
        "publications": re.compile(r"^publications?|papers|research", re.I | re.M),
        "languages": re.compile(r"^languages?", re.I | re.M),
        "summary": re.compile(r"^(?:professional\s+)?summary|objective|profile|about", re.I | re.M),
    }
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        skills_list: Optional[List[str]] = None,
    ):
        self.nlp = None
        self.skills_set: Set[str] = set()
        
        if SPACY_AVAILABLE:
            self._load_spacy(spacy_model)
        
        if skills_list:
            self.skills_set = {s.lower().strip() for s in skills_list if s and len(s.strip()) > 1}
            logger.info(f"Loaded {len(self.skills_set)} skills")
    
    @classmethod
    def from_skills_csv(cls, csv_path: str, spacy_model: str = "en_core_web_sm") -> "EntityExtractor":
        """
        Create extractor with skills loaded from CSV
        
        Args:
            csv_path: Path to CSV with skills (tries columns: skills, skill, name)
        """
        import pandas as pd
        
        skills = []
        try:
            df = pd.read_csv(csv_path)
            
            # Find skill column
            skill_cols = [c for c in df.columns if 'skill' in c.lower()]
            if not skill_cols:
                skill_cols = df.columns.tolist()
            
            for col in skill_cols:
                for val in df[col].dropna():
                    if isinstance(val, str):
                        for s in val.split(','):
                            s = s.strip()
                            if s and len(s) > 1:
                                skills.append(s)
            
            logger.info(f"Loaded {len(set(skills))} skills from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading skills: {e}")
        
        return cls(spacy_model=spacy_model, skills_list=skills)
    
    def _load_spacy(self, model_name: str):
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy: {model_name}")
        except OSError:
            logger.info(f"Downloading {model_name}...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entities from resume text"""
        result = ExtractedEntities()
        
        # 1. Split into sections
        sections = self._detect_sections(text)
        
        # 2. Extract contact info (from header/full text)
        header = sections.get("header", text[:500])
        self._extract_contact(header, text, result)
        
        # 3. Extract with spaCy NER
        if self.nlp:
            self._extract_spacy(text, sections, result)
        
        # 4. Extract education
        edu_text = sections.get("education", "")
        self._extract_education(edu_text, result)
        
        # 5. Extract experience
        exp_text = sections.get("experience", "")
        self._extract_experience(exp_text, result)
        
        # 6. Extract skills
        skills_text = sections.get("skills", text)
        self._extract_skills(skills_text, result)
        
        # 7. Extract other sections
        self._extract_certifications(sections.get("certifications", ""), result)
        self._extract_projects(sections.get("projects", ""), result)
        self._extract_publications(sections.get("publications", ""), result)
        self._extract_languages(sections.get("languages", ""), result)
        self._extract_summary(sections.get("summary", ""), result)
        
        return result
    
    # ============ SECTION DETECTION ============
    
    def _detect_sections(self, text: str) -> Dict[str, str]:
        """Split resume into sections based on headers"""
        sections = {}
        lines = text.split('\n')
        
        # Find section boundaries
        section_starts = []
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.match(line_clean):
                    section_starts.append((i, section_name))
                    break
        
        # Extract header (before first section)
        if section_starts:
            first_section_line = section_starts[0][0]
            sections["header"] = '\n'.join(lines[:first_section_line])
        else:
            sections["header"] = text[:500]
        
        # Extract each section
        for idx, (line_num, section_name) in enumerate(section_starts):
            if idx + 1 < len(section_starts):
                end_line = section_starts[idx + 1][0]
            else:
                end_line = len(lines)
            sections[section_name] = '\n'.join(lines[line_num:end_line])
        
        return sections
    
    # ============ CONTACT EXTRACTION ============
    
    def _extract_contact(self, header: str, full_text: str, result: ExtractedEntities):
        """Extract contact information"""
        # Email
        match = self.EMAIL.search(full_text)
        result.email = match.group() if match else None
        
        # Phone
        match = self.PHONE.search(full_text)
        result.phone = match.group() if match else None
        
        # LinkedIn
        match = self.LINKEDIN.search(full_text)
        result.linkedin = match.group() if match else None
        
        # GitHub
        match = self.GITHUB.search(full_text)
        result.github = match.group() if match else None
        
        # Portfolio (other URLs)
        urls = self.URL.findall(full_text)
        for url in urls:
            if "linkedin" not in url.lower() and "github" not in url.lower():
                result.portfolio = url
                break
        
        # Address - look for city, state pattern or full address in header
        address_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}(?:\s+\d{5})?)|"
            r"(\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln)[^,]*,\s*[^,]+,\s*[A-Z]{2}\s*\d{5})",
            re.I
        )
        match = address_pattern.search(header)
        if match:
            result.address = match.group().strip()
    
    # ============ SPACY EXTRACTION ============
    
    def _extract_spacy(self, text: str, sections: Dict[str, str], result: ExtractedEntities):
        """Extract entities using spaCy NER"""
        # Process header for name
        header = sections.get("header", text[:500])
        header_doc = self.nlp(header)
        
        for ent in header_doc.ents:
            if ent.label_ == "PERSON" and not result.name:
                name = ent.text.strip()
                if 1 < len(name.split()) <= 4 and not any(c.isdigit() for c in name):
                    result.name = name
                    break
        
        # If no name found via NER, try first line heuristic
        if not result.name:
            first_line = header.strip().split('\n')[0].strip()
            words = first_line.split()
            if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w):
                if not any(c.isdigit() for c in first_line) and '@' not in first_line:
                    result.name = first_line
        
        # Process education section for institutions
        edu_text = sections.get("education", "")
        if edu_text:
            edu_doc = self.nlp(edu_text)
            for ent in edu_doc.ents:
                if ent.label_ == "ORG":
                    inst = ent.text.strip()
                    # Filter out non-institutions
                    if self._is_institution(inst):
                        if inst not in result.institutions:
                            result.institutions.append(inst)
    
    def _is_institution(self, text: str) -> bool:
        """Check if text looks like an educational institution"""
        text_lower = text.lower()
        # Must contain education-related keywords or be known pattern
        edu_keywords = ["university", "college", "institute", "school", "academy", 
                       "polytechnic", "mit", "stanford", "harvard", "berkeley"]
        return any(kw in text_lower for kw in edu_keywords) or len(text) > 3
    
    def _is_job_title(self, text: str) -> bool:
        """Check if text looks like a job title"""
        text_lower = text.lower()
        title_keywords = ["engineer", "developer", "manager", "analyst", "designer",
                         "lead", "director", "architect", "specialist", "consultant",
                         "intern", "scientist", "administrator", "coordinator", "officer",
                         "executive", "president", "head", "chief", "vp", "senior", "junior",
                         "associate", "principal", "staff"]
        return any(kw in text_lower for kw in title_keywords)
    
    # ============ EDUCATION EXTRACTION ============
    
    def _extract_education(self, edu_text: str, result: ExtractedEntities):
        """Extract education details"""
        if not edu_text:
            return
        
        # Degrees
        for match in self.DEGREE.finditer(edu_text):
            degree = match.group().strip()
            if degree and degree not in result.degrees:
                result.degrees.append(degree)
        
        # Majors - look for "in [Major]" or "of [Major]"
        major_pattern = re.compile(r"(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+(?:Science|Engineering|Arts|Business))?)", re.I)
        for match in major_pattern.finditer(edu_text):
            major = match.group(1).strip()
            if major and major not in result.majors and len(major) > 3:
                result.majors.append(major)
        
        # Graduation years
        years = self.YEAR.findall(edu_text)
        result.graduation_years = list(dict.fromkeys(years))
        
        # GPA
        match = self.GPA.search(edu_text)
        if match:
            result.gpa = match.group(1)
        
        # Institutions - parse lines looking for university names
        lines = edu_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for institution keywords
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["university", "college", "institute", "school", "academy"]):
                # Extract the institution name
                # Remove dates, GPA, degree info
                inst = re.sub(r'\d{4}', '', line)
                inst = re.sub(r'GPA[:\s]*\d\.\d+', '', inst, flags=re.I)
                inst = re.sub(r'[-–—,].*$', '', inst)
                inst = inst.strip()
                if inst and inst not in result.institutions and len(inst) > 3:
                    result.institutions.append(inst)
    
    # ============ EXPERIENCE EXTRACTION ============
    
    def _extract_experience(self, exp_text: str, result: ExtractedEntities):
        """Extract work experience details"""
        if not exp_text:
            return
        
        # Date ranges
        for match in self.DATE_RANGE.finditer(exp_text):
            date_range = match.group().strip()
            if date_range and date_range not in result.work_dates:
                result.work_dates.append(date_range)
        
        # Parse "Job Title at Company" or "Job Title, Company" patterns
        title_company_patterns = [
            # "Senior Data Scientist at Google"
            re.compile(r"([A-Z][a-zA-Z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Lead|Director|Architect|Specialist|Consultant|Intern|Scientist|Administrator|Coordinator|Officer|Executive|President))\s+(?:at|@)\s+([A-Z][a-zA-Z\s&.,]+)", re.I),
            # "Software Engineer - Google" or "Software Engineer | Google"
            re.compile(r"([A-Z][a-zA-Z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Lead|Director|Architect|Specialist|Consultant|Intern|Scientist))\s*[-|–—]\s*([A-Z][a-zA-Z\s&.,]+)"),
            # "Google - Software Engineer" (company first)
            re.compile(r"^([A-Z][a-zA-Z\s&.,]+?)\s*[-|–—]\s*([A-Z][a-zA-Z\s]+(?:Engineer|Developer|Manager|Analyst|Designer|Lead|Director|Architect|Specialist|Consultant|Intern|Scientist))$", re.M),
        ]
        
        for pattern in title_company_patterns:
            for match in pattern.finditer(exp_text):
                g1, g2 = match.group(1).strip(), match.group(2).strip()
                
                # Determine which is title and which is company
                if self._is_job_title(g1):
                    title, company = g1, g2
                elif self._is_job_title(g2):
                    title, company = g2, g1
                else:
                    continue
                
                # Clean and add
                title = re.sub(r'\s+', ' ', title).strip()
                company = re.sub(r'\s+', ' ', company).strip()
                company = re.sub(r'[,.]$', '', company)  # Remove trailing punctuation
                
                if title and title not in result.job_titles and len(title) > 3:
                    result.job_titles.append(title)
                if company and company not in result.companies and len(company) > 1:
                    if not self._is_job_title(company):  # Double check
                        result.companies.append(company)
        
        # Also look for standalone job titles at start of lines
        lines = exp_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('-', '•', '*', '–')):
                continue
            
            # Check if line is a job title (capitalized, contains title keyword)
            if self._is_job_title(line) and len(line) < 60:
                # Remove company part if present
                for sep in [' at ', ' @ ', ' - ', ' | ', ', ']:
                    if sep in line:
                        parts = line.split(sep)
                        for part in parts:
                            part = part.strip()
                            if self._is_job_title(part) and part not in result.job_titles:
                                result.job_titles.append(part)
                            elif not self._is_job_title(part) and part not in result.companies:
                                if len(part) > 1:
                                    result.companies.append(part)
                        break
                else:
                    # No separator, check if whole line is a title
                    if line not in result.job_titles:
                        result.job_titles.append(line)
    
    # ============ SKILLS EXTRACTION ============
    
    def _extract_skills(self, text: str, result: ExtractedEntities):
        """Extract skills from text"""
        if not self.skills_set:
            return
        
        text_lower = text.lower()
        found = []
        
        for skill in self.skills_set:
            # Word boundary match
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found.append(skill)
        
        result.skills = found
    
    # ============ OTHER SECTIONS ============
    
    def _extract_certifications(self, text: str, result: ExtractedEntities):
        """Extract certifications"""
        if not text:
            return
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Skip header line
        for line in lines[1:]:
            if len(line) > 5 and not self.SECTION_PATTERNS["certifications"].match(line):
                result.certifications.append(line)
    
    def _extract_projects(self, text: str, result: ExtractedEntities):
        """Extract project titles"""
        if not text:
            return
        
        # Look for project titles (usually bold or at start of line)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Project title is usually short and at start
            if 5 < len(line) < 80 and not line.startswith(('-', '•', '*')):
                if not self.SECTION_PATTERNS["projects"].match(line):
                    # Check if it looks like a title (capitalized, no full sentences)
                    if line[0].isupper() and line.count('.') < 2:
                        result.projects.append(line)
    
    def _extract_publications(self, text: str, result: ExtractedEntities):
        """Extract publication titles"""
        if not text:
            return
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines[1:]:
            if len(line) > 10 and not self.SECTION_PATTERNS["publications"].match(line):
                result.publications.append(line)
    
    def _extract_languages(self, text: str, result: ExtractedEntities):
        """Extract languages"""
        if not text:
            return
        
        # Common languages
        common_languages = {
            "english", "spanish", "french", "german", "chinese", "mandarin",
            "japanese", "korean", "hindi", "arabic", "portuguese", "russian",
            "italian", "dutch", "polish", "turkish", "vietnamese", "thai",
            "indonesian", "malay", "tamil", "telugu", "bengali", "punjabi",
            "urdu", "gujarati", "marathi", "kannada", "malayalam", "swedish",
            "norwegian", "danish", "finnish", "greek", "hebrew", "persian",
        }
        
        text_lower = text.lower()
        for lang in common_languages:
            if lang in text_lower:
                result.languages.append(lang.title())
    
    def _extract_summary(self, text: str, result: ExtractedEntities):
        """Extract summary/objective"""
        if not text:
            return
        
        lines = text.split('\n')
        # Skip header, take rest as summary
        summary_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line and not any(p.match(line) for p in self.SECTION_PATTERNS.values()):
                summary_lines.append(line)
        
        if summary_lines:
            result.summary = ' '.join(summary_lines)[:500]  # Limit length
    
    # ============ BATCH & SAVE/LOAD ============
    
    def extract_batch(self, texts: List[str]) -> List[ExtractedEntities]:
        """Extract from multiple texts"""
        return [self.extract(text) for text in texts]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # read from data/structured/skills.csv for skills
    skills_csv = "data/raw/structured/skills.csv"
    extractor = EntityExtractor.from_skills_csv(str(skills_csv))
    
    test_resume = """
    Pratik Pujari
    +1 303 6206112 | pratikpujari1000@gmail.com | github.com/Dracula-101
    Education
    University of Colorado Boulder
    Boulder, CO
    Master of Science in Computer Science
    Aug. 2025 – Present
    Sardar Patel Institute of Technology
    Mumbai, India
    Bachelor of Technology in Computer Engineering
    GPA: 8.6/10.0, Oct. 2020 – May 2024
    Experience
    Software Developer Intern
    Nov. 2023 – Dec. 2024
    Aim4U Software Solutions Private Limited
    Mumbai, India
    • Built Dermalens AI healthcare app using Flutter and AWS Lambda, achieving 300+ downloads
    • Implemented ML skin disease detection with 90-95% accuracy using computer vision models
    • Integrated Agora SDK for real-time video calls, enabling doctor-patient consultations within the app
    • Presented product demos to investors and participated in 3+ startup competitions for funding validation
    • Co-developed Autobuddys Flutter app with memory games for autistic children, reaching 250+ downloads
    • Added performance tracking and analytics features to monitor user progress and game completion rates
    Research and Software Intern
    Jan. 2023 – June 2023
    Acuradyne Systems Private Limited
    Mumbai, India
    • Researched Bluetooth Classic vs BLE protocols, comparing HC-05 and ESP32C3 for medical device
    applications
    • Developed Flutter mobile app for blood pressure monitoring with real-time data charts and visualization
    • Built auto-connect/disconnect features for BLE devices with power optimization to extend battery life
    • Created data pipeline connecting piezoelectric sensors to Android phones for health data collection
    • Optimized system to increase sampling rate from 80 to 300+ data points/second and fixed Bluetooth bugs
    • Improved battery performance in medical monitoring devices through efficient power management coding
    Projects
    Nesters | Flutter, Supabase, Firebase, GCP, Python | Playstore | AppStore
    May 2024 – Jan. 2025
    • Created a Flutter application to help fresher students connect with potential roommates
    • Integrated real-time chat using Firebase Firestore and added push notifications for instant communication
    • Developed features for users to list available rooms and search for accommodations with advanced filters
    • Implemented a marketplace for buying, selling, or renting household items to improve the overall living experience
    JetScan | Android, OpenCV, Firebase, GCP | Github | Demo
    May 2024 – Nov. 2024
    • Native Jetpack Compose Android app to scan and share documents efficiently
    • Developed an advanced square contour document detection algorithm using multi-stage edge analysis in OpenCV
    • Seamlessly integrated Google Document AI’s OCR for effortless and accurate document retrieval and processing.
    • Implemented a marketplace for buying, selling, or renting household items to improve the overall living experience
    Crop and Weed Segmentation | Python, Kaggle, TensorFlow | Paper
    Feb. 2024 – June 2024
    • Developed a ResNet-Unet model for weed detection in UAV images of sorghum fields, achieving up to a 0.929
    Sørensen-Dice Coefficient
    • Implemented data augmentation techniques (e.g., random rotations, flips) to enhance model robustness
    • Evaluated performance across different growth stages and environments, demonstrating versatility
    • Contributed to preprocessing methods like CLAHE, gamma correction to improve image quality, feature extraction
    Publications
    • Enhancing Fetal ECG Extraction: Novel UNET Architecture using Continuous Wavelet Transforms,
    International Conference on Integrated Circuits, Communication, and Computing Systems, IIIT UNA, 2024
    • CRBC- An automated approach for Handwriting OCR, 4th International Conference on Artificial
    Intelligence and Signal Processing,VIT-AP University, Vellore, 2024
    • Enhancing Shift-Invariance for Accurate Brain MRI Skull-Stripping using Adaptive Polyphase
    Pooling in Modified U-net, 2nd International Conference on Automation, Computing and Renewable Systems,
    Mount Zion College of Engineering and Technology, Tamil Nadu, 2023
    
    """
    
    result = extractor.extract(test_resume)
    
    print("\n" + "="*50)
    print("EXTRACTED ENTITIES")
    print("="*50)
    
    print("\n--- Contact ---")
    print(f"Name: {result.name}")
    print(f"Email: {result.email}")
    print(f"Phone: {result.phone}")
    print(f"Address: {result.address}")
    print(f"LinkedIn: {result.linkedin}")
    print(f"GitHub: {result.github}")
    
    print("\n--- Professional ---")
    print(f"Job Titles: {result.job_titles}")
    print(f"Companies: {result.companies}")
    print(f"Work Dates: {result.work_dates}")
    print(f"Skills: {result.skills}")
    
    print("\n--- Education ---")
    print(f"Degrees: {result.degrees}")
    print(f"Majors: {result.majors}")
    print(f"Institutions: {result.institutions}")
    print(f"Graduation Years: {result.graduation_years}")
    print(f"GPA: {result.gpa}")
    
    print("\n--- Other ---")
    print(f"Certifications: {result.certifications}")
    print(f"Projects: {result.projects}")
    print(f"Languages: {result.languages}")
    print(f"Summary: {result.summary[:100] if result.summary else 'None'}...")