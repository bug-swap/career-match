import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")
        SPACY_MODEL = "lg"
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        SPACY_MODEL = "sm"
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    logger.warning("spaCy not available")
    SPACY_AVAILABLE = False
    nlp = None
    SPACY_MODEL = None


@dataclass
class SectionPrediction:
    section: str
    confidence: float
    all_scores: Dict[str, float]


class SectionClassifier:
    SECTIONS = ["contact", "experience", "education", "project", "skills", "languages", "publication"]
    
    SECTION_KEYWORDS = {
        "contact": ["contact", "email", "phone", "mobile", "address", "linkedin", "github"],
        "experience": ["experience", "employment", "work", "job", "intern", "responsibilities", "career"],
        "education": ["education", "degree", "university", "college", "bachelor", "master", "phd", "gpa"],
        "project": ["project", "portfolio", "developed", "built", "created", "implemented", "demo"],
        "skills": ["skills", "technical", "expertise", "technologies", "programming", "proficient"],
        "languages": ["languages", "fluent", "native", "proficiency", "bilingual"],
        "publication": ["publications", "research", "published", "journal", "conference", "doi", "arxiv"]
    }

    DATE_PATTERN = r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}[/-]\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|Present|Current|present|current))\b'
    
    def __init__(self):
        self._compile_patterns()
        self.is_fitted = True
        
    def _compile_patterns(self):
        self.date_pattern = re.compile(self.DATE_PATTERN, re.IGNORECASE)
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}')
        self.url_pattern = re.compile(r'https?://[^\s]+|(?:www\.|github\.com/|linkedin\.com/in/)[\w\-./]+', re.IGNORECASE)
        self.gpa_pattern = re.compile(r'\b(?:GPA|CGPA)[\s:]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?\b', re.IGNORECASE)
        self.degree_pattern = re.compile(
            r'\b(Bachelor|Master|PhD|Doctorate|B\.?Tech|M\.?Tech|B\.?S\.?|M\.?S\.?|MBA|Diploma|Associate)\b',
            re.IGNORECASE
        )
        
    def preprocess_resume(self, text: str) -> str:
        text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2022', '-')
        text = re.sub(r'[●○■□▪▫•‣⁃◘◦⦾⦿►▸▹▻▷]', '- ', text)
        text = re.sub(r'^\s*(?:Page\s+)?\d+\s*(?:of\s*\d+)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return '\n'.join(line.rstrip() for line in text.split('\n'))
    
    def _is_header(self, line: str) -> bool:
        line = line.strip()
        if not line or len(line) < 2 or len(line) > 100:
            return False
            
        score = 0
        line_lower = line.lower()
        line_clean = re.sub(r'[*#:\-_]+', ' ', line_lower).strip()
        
        if any(section in line_clean for section in self.SECTIONS):
            score += 5
        
        for section, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in line_clean for kw in keywords[:2]):
                score += 3
                break
        
        if line.isupper() and len(line) > 3:
            score += 2
        if line.endswith(':'):
            score += 2
        if len(line) < 30:
            score += 1
        
        if self.date_pattern.search(line):
            score -= 4
        if self.email_pattern.search(line):
            score -= 4
        if line.startswith(('-', '•', '*', '>')):
            score -= 5
            
        return score >= 5
    
    def _detect_section_type(self, header: str) -> Optional[str]:
        header_lower = header.lower().strip()
        header_clean = re.sub(r'[*#:\-_]+', ' ', header_lower).strip()
        
        for section in self.SECTIONS:
            if section in header_clean:
                return section
                
        for section, keywords in self.SECTION_KEYWORDS.items():
            if any(kw in header_clean for kw in keywords[:3]):
                return section
                
        return None
    
    def segment_resume(self, resume_text: str) -> Dict[str, str]:
        resume_text = self.preprocess_resume(resume_text)
        lines = resume_text.split('\n')
        sections = {}
        boundaries = []
        
        for i, line in enumerate(lines):
            if self._is_header(line):
                section = self._detect_section_type(line)
                if section and (not boundaries or boundaries[-1][1] != section):
                    boundaries.append((i, section))
        
        for idx, (start, name) in enumerate(boundaries):
            end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
            content = '\n'.join(lines[start + 1:end]).strip()
            if content:
                sections[name] = sections.get(name, '') + ('\n\n' + content if name in sections else content)
        
        if "contact" not in sections and boundaries:
            sections["contact"] = '\n'.join(lines[:boundaries[0][0]]).strip()
            
        return sections
    
    def _extract_entities(self, text: str, filter_low_confidence: bool = True) -> Dict[str, List[str]]:
        """Enhanced entity extraction with better filtering"""
        if not SPACY_AVAILABLE or not nlp:
            return {"ORG": [], "GPE": [], "DATE": [], "PERSON": []}
            
        doc = nlp(text)
        entities = {"ORG": [], "GPE": [], "DATE": [], "PERSON": []}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                continue
                
            clean_text = ent.text.strip()
            if not clean_text or clean_text in entities[ent.label_]:
                continue
            
            if ent.label_ == "ORG":
                if any(word in clean_text.lower() for word in ['university', 'college', 'institute', 'school']):
                    entities[ent.label_].append(clean_text)
                elif any(word in clean_text.lower() for word in ['limited', 'ltd', 'inc', 'corp', 'llc', 'systems', 'solutions', 'software', 'technologies']):
                    entities[ent.label_].append(clean_text)
                elif not filter_low_confidence:
                    entities[ent.label_].append(clean_text)
            else:
                entities[ent.label_].append(clean_text)
                
        return entities
    
    def _is_likely_job_title(self, text: str) -> bool:
        """Use NLP to determine if text is a job title"""
        job_keywords = ['engineer', 'developer', 'manager', 'analyst', 'intern', 
                       'specialist', 'consultant', 'architect', 'designer', 'scientist',
                       'director', 'lead', 'senior', 'junior', 'associate', 'coordinator',
                       'researcher', 'software', 'data', 'product', 'project']
        
        text_lower = text.lower()
        if any(kw in text_lower for kw in job_keywords):
            return True
        
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            pos_tags = [token.pos_ for token in doc]
            return 'NOUN' in pos_tags or 'PROPN' in pos_tags
            
        return False
    
    def _extract_company_fallback(self, text: str) -> Optional[str]:
        """Fallback regex-based company extraction"""
        company_patterns = [
            r'([A-Z][A-Za-z0-9\s&]+(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|Limited|Ltd\.?|Inc\.?|Corp\.?|LLC|LLP|Co\.?))',
            r'([A-Z][A-Za-z0-9\s&]+(?:Technologies|Systems|Solutions|Software|Labs|Group|Consulting|Services))',
        ]
        
        for pattern in company_patterns:
            if match := re.search(pattern, text):
                return match.group(1).strip()
        
        return None
    
    def _parse_experience_items(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        items = []
        
        date_indices = []
        for i, line in enumerate(lines):
            if self.date_pattern.search(line) and not line.startswith(('-', '•', '*', '>')):
                date_indices.append(i)
        
        for date_idx in date_indices:
            date_line = lines[date_idx]
            is_present = 'present' in date_line.lower() or 'current' in date_line.lower()
            
            title = ""
            company = ""
            location = ""
            
            search_start = max(0, date_idx - 4)
            context_lines = []
            
            for i in range(date_idx - 1, search_start - 1, -1):
                if i < 0:
                    break
                line = lines[i]
                if line.startswith(('-', '•', '*', '>')):
                    continue
                if self.date_pattern.search(line):
                    break
                context_lines.insert(0, line)
            
            for line in context_lines:
                entities = self._extract_entities(line)
                
                if not company:
                    if entities["ORG"]:
                        company = entities["ORG"][0]
                        if entities["GPE"]:
                            location = entities["GPE"][0]
                    else:
                        regex_company = self._extract_company_fallback(line)
                        if regex_company:
                            company = regex_company
                            loc_entities = self._extract_entities(line)
                            if loc_entities["GPE"]:
                                location = loc_entities["GPE"][0]
            
            for line in context_lines:
                if company and company in line:
                    continue
                if self._is_likely_job_title(line) and len(line) > 2:
                    title = line
                    break
            
            if not title:
                for line in context_lines:
                    if company and company in line:
                        continue
                    if len(line) > 2:
                        title = line
                        break
            
            responsibilities = []
            for i in range(date_idx + 1, len(lines)):
                line = lines[i]
                if self.date_pattern.search(line):
                    break
                if line.startswith(('-', '•', '*', '>')):
                    resp = line.lstrip('-•*> ').strip()
                    if resp:
                        responsibilities.append(resp)
            
            items.append({
                "title": title,
                "company": company,
                "date": date_line,
                "isPresent": is_present,
                "location": location,
                "responsibilities": responsibilities
            })
        
        return items
    
    def _parse_education_items(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        items = []
        processed_indices: Set[int] = set()
        
        i = 0
        while i < len(lines):
            if i in processed_indices:
                i += 1
                continue
                
            line = lines[i]
            window_end = min(i + 6, len(lines))
            window_lines = lines[i:window_end]
            window_text = '\n'.join(window_lines)
            
            entities = self._extract_entities(window_text, filter_low_confidence=False)
            has_org = bool(entities["ORG"])
            has_degree = bool(self.degree_pattern.search(window_text))
            
            if has_org or has_degree:
                institution = ""
                degree = ""
                major = ""
                location = ""
                date = ""
                gpa = ""
                
                institution_line_idx = None
                degree_line_idx = None
                
                for j, w_line in enumerate(window_lines):
                    line_entities = self._extract_entities(w_line, filter_low_confidence=False)
                    
                    if line_entities["ORG"] and not institution:
                        institution = line_entities["ORG"][0]
                        institution_line_idx = j
                        if line_entities["GPE"]:
                            location = line_entities["GPE"][0]
                    
                    if self.degree_pattern.search(w_line) and not degree:
                        degree_line_idx = j
                        if in_match := re.search(r'(.+?)\s+in\s+(.+?)(?:\s*(?:GPA|,|$))', w_line, re.IGNORECASE):
                            degree = in_match.group(1).strip()
                            major_text = in_match.group(2).strip()
                            major = self.gpa_pattern.sub('', major_text).strip(' ,')
                            major = self.date_pattern.sub('', major).strip(' ,')
                        else:
                            degree_match = self.degree_pattern.search(w_line)
                            degree = degree_match.group(0)
                    
                    if not location and line_entities["GPE"] and j != institution_line_idx:
                        location = line_entities["GPE"][0]
                    
                    if not date and (date_match := self.date_pattern.search(w_line)):
                        date = date_match.group(0)
                    
                    if not gpa and (gpa_match := self.gpa_pattern.search(w_line)):
                        gpa = f"{gpa_match.group(1)}/{gpa_match.group(2)}" if gpa_match.group(2) else gpa_match.group(1)
                
                if institution.startswith(degree):
                    institution = ""
                
                if degree or institution:
                    items.append({
                        "degree": degree,
                        "major": major,
                        "institution": institution,
                        "location": location,
                        "date": date,
                        "gpa": gpa
                    })
                    
                    for idx in range(i, min(i + 4, len(lines))):
                        processed_indices.add(idx)
                
                i = window_end
            else:
                i += 1
        
        return items
    
    def _parse_project_items(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        items = []
        current = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            is_bullet = line.startswith(('-', '•', '*', '>'))
            
            if is_bullet:
                if current:
                    current["description"].append(line.lstrip('-•*> ').strip())
                i += 1
                continue
            
            has_pipe = '|' in line
            has_date = bool(self.date_pattern.search(line))
            next_line_is_bullet = (i + 1 < len(lines) and lines[i + 1].startswith(('-', '•', '*', '>')))
            
            is_project_header = has_pipe and not is_bullet
            
            if is_project_header:
                if current:
                    items.append(current)
                
                parts = [p.strip() for p in line.split('|')]
                name = parts[0].strip()
                
                url_list = []
                for url_match in self.url_pattern.finditer(line):
                    url_list.append(url_match.group())
                
                url = url_list[0] if url_list else ""
                
                for u in url_list:
                    name = name.replace(u, '').strip()
                
                tech = ""
                date_str = ""
                
                if len(parts) >= 2:
                    for j in range(1, len(parts)):
                        part = parts[j]
                        
                        for u in url_list:
                            part = part.replace(u, '').strip()
                        
                        if self.date_pattern.search(part):
                            if date_match := self.date_pattern.search(part):
                                date_str = date_match.group(0)
                            part = self.date_pattern.sub('', part).strip()
                        
                        part = re.sub(r'\b(Github|Demo|Playstore|AppStore|Paper|Link)\b', '', part, flags=re.IGNORECASE).strip()
                        part = re.sub(r'\s+', ' ', part).strip()
                        
                        if part and len(part) > 2 and not tech:
                            tech = part
                
                current = {
                    "name": name.strip(),
                    "url": url,
                    "description": [],
                    "technologies": tech,
                    "date": date_str
                }
            
            i += 1
        
        if current:
            items.append(current)
        
        return items
    
    def _parse_publication_items(self, text: str) -> List[Dict]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        items = []
        current_pub = []
        
        for line in lines:
            line = line.lstrip('-•* ').strip()
            if not line:
                continue
            
            if re.search(r',\s*\d{4}\s*$', line) or (current_pub and len(' '.join(current_pub)) > 100):
                current_pub.append(line)
                full_text = ' '.join(current_pub)
                
                doi = ""
                if doi_match := re.search(r'doi:\s*(10\.\S+)', full_text, re.IGNORECASE):
                    doi = doi_match.group(1)
                
                year = ""
                if year_match := re.search(r'\b(\d{4})\b', full_text):
                    year = year_match.group(1)
                
                items.append({"title": full_text.strip(), "year": year, "doi": doi})
                current_pub = []
            else:
                current_pub.append(line)
        
        if current_pub:
            full_text = ' '.join(current_pub)
            doi = ""
            if doi_match := re.search(r'doi:\s*(10\.\S+)', full_text, re.IGNORECASE):
                doi = doi_match.group(1)
            year = ""
            if year_match := re.search(r'\b(\d{4})\b', full_text):
                year = year_match.group(1)
            items.append({"title": full_text.strip(), "year": year, "doi": doi})
        
        return items
    
    def _parse_skills_items(self, text: str) -> List[str]:
        skills = []
        text = text.replace('\n', '|||')
        
        for part in text.split(','):
            for item in part.split('|||'):
                item = item.strip().lstrip('-•* ').strip()
                item = re.sub(r'^[A-Za-z/&\s]+:\s*', '', item)
                if item and len(item) > 1:
                    skills.append(item)
        
        seen = set()
        return [s for s in skills if not (s.lower() in seen or seen.add(s.lower()))]
    
    def _parse_languages_items(self, text: str) -> List[Dict]:
        items = []
        lines = text.split('\n') if '\n' in text else text.split(',')
        
        for line in lines:
            line = line.strip().lstrip('-•* ').strip()
            if not line:
                continue
            
            proficiency = ""
            language = line
            
            if match := re.search(r'(.+?)\s*[\(\-]\s*(.+?)[\)]?$', line):
                language = match.group(1).strip()
                proficiency = match.group(2).strip().rstrip(')')
            
            items.append({"language": language, "proficiency": proficiency})
        
        return items
    
    def _parse_contact_items(self, text: str) -> Dict:
        contact = {"name": "", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
        
        if email := self.email_pattern.search(text):
            contact["email"] = email.group()
        if phone := self.phone_pattern.search(text):
            contact["phone"] = phone.group()
        
        for url_match in self.url_pattern.finditer(text):
            url = url_match.group()
            if 'linkedin' in url.lower():
                contact["linkedin"] = url
            elif 'github' in url.lower():
                contact["github"] = url
            elif not contact["website"]:
                contact["website"] = url
        
        lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
        for line in lines:
            if self.email_pattern.search(line) or self.url_pattern.search(line):
                continue
            if self.phone_pattern.search(line) and len(line) < 25:
                continue
            
            if not contact["name"]:
                name = self.email_pattern.sub('', line)
                name = self.phone_pattern.sub('', name)
                name = self.url_pattern.sub('', name)
                name = re.sub(r'[\|,]\s*', ' ', name).strip()
                
                if len(name) > 2 and not any(c.isdigit() for c in name[:5]):
                    contact["name"] = name
                    break
        
        return contact
    
    def parse_sections(self, sections: Dict[str, str]) -> Dict:
        parsed = {}
        
        parsers = {
            "experience": self._parse_experience_items,
            "education": self._parse_education_items,
            "project": self._parse_project_items,
            "publication": self._parse_publication_items,
            "skills": self._parse_skills_items,
            "languages": self._parse_languages_items,
            "contact": self._parse_contact_items
        }
        
        for section_name, content in sections.items():
            parser = parsers.get(section_name)
            parsed[section_name] = parser(content) if parser else content
        
        return parsed
    
    @classmethod
    def load(cls, path=None):
        logger.info(f"Loading SectionClassifier with spaCy model: {SPACY_MODEL}")
        return cls()