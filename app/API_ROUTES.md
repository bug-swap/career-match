# FastAPI ML Service - API Route Summary

## 🎯 Main Resume Processing Routes (All accept PDF via multipart/form-data)

### 1. **Central Route - Complete ML Pipeline**
```
POST /api/v1/resume/parse
```
**Input:** PDF file (multipart/form-data)  
**Pipeline:** PDF Extract → Preprocess → Sections → Entities → Category  
**Output:** Everything (sections + entities + classification + metadata)

**Use Case:** When you need complete resume analysis in one request

---

### 2. **Model 1 - Text Extraction**
```
POST /api/v1/resume/extract-text
```
**Input:** PDF file (multipart/form-data)  
**Pipeline:** PDF Extract → Preprocess  
**Output:** Raw text, cleaned text, text statistics

**Use Case:** When you only need to extract text from PDF

---

### 3. **Model 2 - Section Classification**
```
POST /api/v1/resume/classify-sections
```
**Input:** PDF file (multipart/form-data)  
**Pipeline:** PDF Extract → Preprocess → Sections  
**Output:** Classified resume sections (contact, education, experience, skills, etc.)

**Use Case:** When you need resume organized into sections

---

### 4. **Model 3 - Entity Extraction (NER)**
```
POST /api/v1/resume/extract-entities
```
**Input:** PDF file (multipart/form-data)  
**Pipeline:** PDF Extract → Preprocess → Sections → Entities  
**Output:** Contact info, companies, job titles, skills, education, dates, locations

**Use Case:** When you need to extract specific entities like skills, companies, dates

---

### 5. **Model 4 - Category Classification**
```
POST /api/v1/resume/classify-category
```
**Input:** PDF file (multipart/form-data)  
**Pipeline:** PDF Extract → Preprocess → Sections → Entities → Category  
**Output:** Resume category, confidence scores, top 3 predictions

**Use Case:** When you need to classify resume into job categories

---

## 📊 ML Pipeline Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE ML PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PDF Upload (multipart/form-data)                          │
│           ↓                                                     │
│  2. Text Extraction                    ← /extract-text         │
│           ↓                                                     │
│  3. Text Preprocessing                                         │
│           ↓                                                     │
│  4. Section Classification             ← /classify-sections    │
│           ↓                                                     │
│  5. Entity Extraction (NER)            ← /extract-entities     │
│           ↓                                                     │
│  6. Category Classification            ← /classify-category    │
│           ↓                                                     │
│  7. Complete Response                  ← /parse               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Quick Test Commands

### Test Central Route (All Models)
```bash
curl -X POST "http://localhost:8000/api/v1/resume/parse" \
  -F "file=@resume.pdf"
```

### Test Individual Models
```bash
# Model 1: Text Extraction
curl -X POST "http://localhost:8000/api/v1/resume/extract-text" \
  -F "file=@resume.pdf"

# Model 2: Section Classification
curl -X POST "http://localhost:8000/api/v1/resume/classify-sections" \
  -F "file=@resume.pdf"

# Model 3: Entity Extraction
curl -X POST "http://localhost:8000/api/v1/resume/extract-entities" \
  -F "file=@resume.pdf"

# Model 4: Category Classification
curl -X POST "http://localhost:8000/api/v1/resume/classify-category" \
  -F "file=@resume.pdf"
```

## 📝 Response Format Comparison

### Central Route (`/parse`)
```json
{
  "success": true,
  "data": {
    "contact": {...},
    "sections": {...},
    "entities": {...},
    "classification": {...},
    "metadata": {...}
  }
}
```

### Text Extraction (`/extract-text`)
```json
{
  "success": true,
  "raw_text": "...",
  "clean_text": "...",
  "metadata": {
    "raw_char_count": 5432,
    "clean_char_count": 5120,
    "word_count": 856,
    "processing_time_ms": 145
  }
}
```

### Section Classification (`/classify-sections`)
```json
{
  "success": true,
  "sections": {
    "contact": "...",
    "education": "...",
    "experience": "..."
  },
  "metadata": {
    "section_count": 5,
    "processing_time_ms": 234
  }
}
```

### Entity Extraction (`/extract-entities`)
```json
{
  "success": true,
  "contact": {...},
  "entities": {
    "companies": [...],
    "skills": [...],
    "job_titles": [...]
  },
  "metadata": {
    "total_entities": 15,
    "processing_time_ms": 312
  }
}
```

### Category Classification (`/classify-category`)
```json
{
  "success": true,
  "classification": {
    "category": "Software Engineer",
    "confidence": 0.92,
    "top_3": [...]
  },
  "metadata": {
    "processing_time_ms": 398
  }
}
```

## 🚀 Performance Characteristics

| Route | Models Run | Avg Time | Use When |
|-------|-----------|----------|----------|
| `/parse` | All 4 | ~400-600ms | Need complete analysis |
| `/extract-text` | 1 | ~100-150ms | Just need text |
| `/classify-sections` | 2 | ~200-250ms | Need sections only |
| `/extract-entities` | 3 | ~300-350ms | Need entities only |
| `/classify-category` | 4 | ~350-450ms | Need category only |

## 🎓 Additional Features

- ✅ All routes accept PDF via multipart/form-data
- ✅ Automatic PDF validation (type and size)
- ✅ Detailed error messages with proper HTTP status codes
- ✅ Request/response logging
- ✅ Processing time tracking
- ✅ OpenAPI documentation at `/docs`
- ✅ Alternative JSON-based endpoints available for text input

## 📚 Documentation

Once server is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/api/v1/health
- **Models Status:** http://localhost:8000/api/v1/models/status
