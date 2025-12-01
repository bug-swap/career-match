# Career Match ML Service - FastAPI

A high-performance ML service built with FastAPI for resume parsing, classification, and job matching.

## Features

✅ **FastAPI Framework** - Modern, fast, and async-ready  
✅ **PDF Resume Upload** - Multipart form data support  
✅ **Comprehensive Logging** - Structured JSON or colored text logs  
✅ **Error Handling** - Proper 404, validation, and exception handlers  
✅ **OpenAPI Documentation** - Auto-generated docs at `/docs` and `/redoc`  
✅ **CORS Support** - Configurable cross-origin requests  
✅ **Request Logging** - Automatic logging of all requests/responses  
✅ **ML Pipeline** - Resume section classification, entity extraction, and job matching

## API Endpoints

### Health Check
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/models/status` - ML models loading status

### Resume Processing (PDF Multipart Form)

#### 🎯 Central Route - Complete ML Pipeline
- **`POST /api/v1/resume/parse`** - Run ALL ML models on PDF
  - Accepts: PDF file (multipart/form-data)
  - Returns: Complete analysis (sections + entities + classification)
  - Pipeline: PDF Extract → Preprocess → Sections → Entities → Category

#### 📄 Model 1 - Text Extraction
- **`POST /api/v1/resume/extract-text`** - Extract and preprocess text from PDF
  - Accepts: PDF file (multipart/form-data)
  - Returns: Raw text, cleaned text, statistics
  - Pipeline: PDF Extract → Preprocess

#### 📋 Model 2 - Section Classification
- **`POST /api/v1/resume/classify-sections`** - Classify resume sections from PDF
  - Accepts: PDF file (multipart/form-data)
  - Returns: Classified sections (contact, education, experience, skills, etc.)
  - Pipeline: PDF Extract → Preprocess → Sections

#### 🏷️ Model 3 - Entity Extraction (NER)
- **`POST /api/v1/resume/extract-entities`** - Extract entities using spaCy NER from PDF
  - Accepts: PDF file (multipart/form-data)
  - Returns: Contact info, companies, skills, job titles, education, dates, locations
  - Pipeline: PDF Extract → Preprocess → Sections → Entities

#### 🎓 Model 4 - Category Classification
- **`POST /api/v1/resume/classify-category`** - Classify resume category from PDF
  - Accepts: PDF file (multipart/form-data)
  - Returns: Job category, confidence scores, top 3 predictions
  - Pipeline: PDF Extract → Preprocess → Sections → Entities → Category

### Text-Based Processing (JSON)
- `POST /api/v1/resume/parse/text` - Parse resume from text (legacy)

### Classification (JSON)
- `POST /api/v1/classify/sections` - Classify text into resume sections
- `POST /api/v1/classify/category` - Classify resume category

### Job Matching (JSON)
- `POST /api/v1/match/jobs` - Match resume with job descriptions

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/pratik/Desktop/CU\ Boulder/OOPS/career-match/src
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `app` directory:

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Security
SECRET_KEY=your-secret-key-change-in-production

# CORS
CORS_ORIGINS=*

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=text  # or 'json' for JSON logging
```

### 3. Run the Server

**Development (with auto-reload):**
```bash
cd app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
cd app
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Or using the Python script:**
```bash
cd app
DEBUG=true python app.py
```

## Usage Examples

### 🎯 Central Route - Complete Pipeline

```bash
# Run ALL ML models on a PDF resume
curl -X POST "http://localhost:8000/api/v1/resume/parse" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "contact": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "(555) 123-4567"
    },
    "sections": {
      "contact": "...",
      "education": "...",
      "experience": "...",
      "skills": "..."
    },
    "entities": {
      "skills": ["Python", "FastAPI", "Docker"],
      "companies": ["Tech Corp", "StartupXYZ"],
      "job_titles": ["Senior Software Engineer"]
    },
    "classification": {
      "category": "Software Engineer",
      "confidence": 0.92,
      "top_3": [...]
    },
    "metadata": {
      "processing_time_ms": 487
    }
  }
}
```

### 📄 Model 1 - Extract Text Only

```bash
# Extract and preprocess text from PDF
curl -X POST "http://localhost:8000/api/v1/resume/extract-text" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "raw_text": "Original extracted text...",
  "clean_text": "Cleaned preprocessed text...",
  "metadata": {
    "raw_char_count": 5432,
    "clean_char_count": 5120,
    "word_count": 856,
    "processing_time_ms": 145
  }
}
```

### 📋 Model 2 - Classify Sections Only

```bash
# Classify resume into sections
curl -X POST "http://localhost:8000/api/v1/resume/classify-sections" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "sections": {
    "contact": "John Doe\njohn@example.com...",
    "education": "BS Computer Science, MIT, 2020...",
    "experience": "Senior Engineer at Tech Corp...",
    "skills": "Python, Java, AWS..."
  },
  "metadata": {
    "section_count": 5,
    "processing_time_ms": 234
  }
}
```

### 🏷️ Model 3 - Extract Entities Only

```bash
# Extract entities using NER
curl -X POST "http://localhost:8000/api/v1/resume/extract-entities" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "contact": {
    "name": "John Doe",
    "email": "john@example.com"
  },
  "entities": {
    "companies": ["Tech Corp", "StartupXYZ"],
    "job_titles": ["Senior Software Engineer"],
    "skills": ["Python", "FastAPI", "Docker"],
    "education": ["MIT"],
    "dates": ["2020-2023"],
    "locations": ["San Francisco"]
  },
  "metadata": {
    "total_entities": 15,
    "processing_time_ms": 312
  }
}
```

### 🎓 Model 4 - Classify Category Only

```bash
# Classify resume category
curl -X POST "http://localhost:8000/api/v1/resume/classify-category" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "classification": {
    "category": "Software Engineer",
    "confidence": 0.92,
    "top_3": [
      {"category": "Software Engineer", "confidence": 0.92},
      {"category": "Full Stack Developer", "confidence": 0.85},
      {"category": "Backend Engineer", "confidence": 0.78}
    ]
  },
  "metadata": {
    "processing_time_ms": 398
  }
}
```

### Other Endpoints

#### Parse Text Resume (JSON)

```bash
curl -X POST "http://localhost:8000/api/v1/resume/parse/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John Doe\nSoftware Engineer\nEmail: john@example.com\n..."
  }'
```

#### Match Jobs

```bash
curl -X POST "http://localhost:8000/api/v1/match/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Experienced Python developer...",
    "jobs": [
      {
        "job_id": "J001",
        "title": "Senior Python Developer",
        "description": "Looking for experienced Python engineer...",
        "required_skills": ["Python", "Django", "PostgreSQL"],
        "location": "Remote"
      }
    ],
    "top_k": 5
  }'
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Responses

All errors follow a consistent format:

```json
{
  "success": false,
  "error": "Error message",
  "type": "ErrorType",
  "details": []  // Optional, for validation errors
}
```

### Common Status Codes

- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Route not found
- `413 Request Entity Too Large` - File too large (>16MB)
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

## Logging

The service provides two logging formats:

**Text Format (Development):**
```
[2024-11-30 10:15:23] INFO     app.api.routes.resume - Processing resume: example.pdf
```

**JSON Format (Production):**
```json
{
  "timestamp": "2024-11-30T10:15:23Z",
  "level": "INFO",
  "logger": "app.api.routes.resume",
  "message": "Processing resume: example.pdf",
  "method": "POST",
  "path": "/api/v1/resume/parse"
}
```

Set `LOG_FORMAT=json` in `.env` for JSON logging.

## Architecture

```
app/
├── app.py                 # FastAPI application entrypoint
├── config.py              # Pydantic settings configuration
├── api/
│   ├── routes/            # API route handlers
│   │   ├── resume.py      # Resume parsing endpoints
│   │   ├── classification.py
│   │   ├── matching.py
│   │   └── health.py
│   ├── schemas/           # Pydantic models
│   │   ├── request.py     # Request schemas
│   │   └── response.py    # Response schemas
│   ├── services/          # Business logic
│   │   ├── pdf_service.py
│   │   ├── entity_service.py
│   │   ├── classifier_service.py
│   │   ├── matcher_service.py
│   │   └── section_service.py
│   └── models/
│       └── loader.py      # ML model loader (singleton)
├── utils/
│   ├── error_handlers.py  # Exception handlers
│   └── logging_config.py  # Logging configuration
└── artifacts/             # ML model files
    ├── entity_extractor/
    ├── resume_classifier/
    ├── section_classifier/
    └── job_matcher/
```

## Production Deployment

### Using Gunicorn + Uvicorn Workers

```bash
cd app
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance

- Async support for file uploads and processing
- Singleton model loader (models loaded once at startup)
- Efficient request logging with minimal overhead
- Structured JSON logging for log aggregation tools

## Migration from Flask

This service was migrated from Flask to FastAPI with the following improvements:

1. ✅ Async/await support for better concurrency
2. ✅ Automatic OpenAPI documentation
3. ✅ Pydantic validation for requests/responses
4. ✅ Better error handling and validation
5. ✅ Structured logging with JSON support
6. ✅ Type hints throughout the codebase
7. ✅ Improved multipart form handling

---

**Version**: 2.0.0  
**Framework**: FastAPI 0.115.5  
**Python**: 3.8+
