# CareerMatch - AI-Powered Resume and Job Matching Platform

An intelligent career matching system that uses machine learning to parse resumes, classify job categories, and match candidates with relevant job opportunities. The platform combines modern web technologies with advanced NLP models to provide accurate resume analysis and job recommendations.

## Overview

CareerMatch is a full-stack application that helps job seekers find relevant opportunities by:
- Automatically parsing and analyzing PDF/DOCX resumes
- Extracting key information (skills, experience, education, contact details)
- Classifying resumes into job categories with confidence scores
- Matching resumes with job descriptions using semantic similarity
- Scraping fresh job listings from Indeed and LinkedIn
- Providing a modern, responsive web interface

## Features

- **Resume Parsing** - Extract structured data from PDF/DOCX files using Apache Tika and PDFBox
- **Section Classification** - Automatically identify resume sections (contact, education, experience, skills, etc.)
- **Entity Extraction** - Extract names, emails, phone numbers, companies, job titles, and skills using spaCy NER
- **Category Classification** - Classify resumes into job categories with confidence scores using SBERT embeddings
- **Job Matching** - Semantic similarity matching between resumes and job descriptions
- **Job Scraping** - Automated job data collection from Indeed and LinkedIn
- **Vector Embeddings** - Generate and store job embeddings for fast similarity search
- **REST API** - Comprehensive API for all ML operations
- **Modern UI** - React + TypeScript interface for resume upload and job browsing

## Architecture

The system follows a microservices architecture with the following components:

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Frontend  │────────▶│   Backend    │────────▶│   ML App    │
│ React + Vite│         │  Spring Boot │         │   FastAPI   │
└─────────────┘         └──────────────┘         └─────────────┘
                               │                         │
                               ▼                         ▼
                        ┌──────────────┐         ┌─────────────┐
                        │  PostgreSQL  │         │  ML Models  │
                        │   (Supabase) │         │   (PyTorch) │
                        └──────────────┘         └─────────────┘
                               ▲                         ▲
                               │                         │
                        ┌──────────────┐         ┌─────────────┐
                        │   Scrapper   │         │   Encoder   │
                        │ (Cloud Func) │         │(Cloud Func) │
                        └──────────────┘         └─────────────┘
```

### Components

1. **Backend** (`/backend`) - Spring Boot REST API
   - Orchestrates resume processing and job matching
   - Calls Python ML service via HTTP
   - Manages database operations (PostgreSQL via Supabase)
   - Provides paginated job listings with filters
   - Java 17, Spring Boot 3.3.4, Gradle

2. **ML App** (`/app`) - FastAPI ML Service
   - Handles all machine learning operations
   - PDF text extraction and preprocessing
   - Resume section classification
   - Named Entity Recognition (NER) with spaCy
   - Category classification with SBERT
   - Job matching using semantic similarity
   - Python 3.8+, FastAPI, PyTorch, Transformers, spaCy

3. **Frontend** (`/frontend`) - React Web Interface
   - Resume file upload interface
   - Job category selection (up to 3)
   - Job listing and search
   - Match result visualization
   - React, TypeScript, Vite

4. **Scrapper** (`/scrapper`) - Job Scraping Service
   - Scrapes job data from Indeed and LinkedIn
   - Processes and normalizes job listings
   - Stores jobs in Supabase database
   - Triggers embedding generation
   - Python, JobSpy, Flask (Cloud Functions)

5. **Encoder** (`/encoder`) - Embedding Generation Service
   - Generates vector embeddings for job descriptions
   - Uses sentence-transformers for encoding
   - Updates job records with embeddings
   - Python, PyTorch, sentence-transformers (Cloud Functions)

6. **Notebooks** (`/notebooks`) - Data Science Notebooks
   - Data exploration and analysis
   - Model training and evaluation
   - Resume dataset analysis
   - Jupyter notebooks

## Tech Stack

### Backend
- Java 17
- Spring Boot 3.3.4
- Spring Data JPA
- Spring WebFlux (async HTTP client)
- Apache Tika 2.9.1 (PDF/DOCX parsing)
- Apache POI 5.2.5 (DOCX parsing)
- PostgreSQL (via Supabase)
- Gradle

### Machine Learning
- Python 3.8+
- FastAPI 0.123.0
- PyTorch 2.9.1
- Transformers 4.57.3 (Hugging Face)
- spaCy 3.8.11 + en_core_web_lg
- sentence-transformers 5.1.2
- scikit-learn 1.7.2
- PyMuPDF 1.26.6 (PDF processing)

### Frontend
- React 18
- TypeScript
- Vite
- CSS3

### Database & Infrastructure
- PostgreSQL (Supabase)
- Vector embeddings storage
- Google Cloud Functions
- Docker (optional deployment)

## Getting Started

### Prerequisites

- Java 17 or higher
- Python 3.8 or higher
- Node.js 16 or higher
- PostgreSQL (or Supabase account)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd career-match
```

### 2. Database Setup (Supabase)

1. Create a Supabase project at https://supabase.com
2. Create a `jobs` table with the following schema:
   ```sql
   CREATE TABLE jobs (
     id TEXT PRIMARY KEY,
     title TEXT NOT NULL,
     category TEXT,
     company TEXT,
     location TEXT,
     date_posted TIMESTAMP,
     job_type TEXT,
     is_remote BOOLEAN,
     min_amount FLOAT,
     max_amount FLOAT,
     currency TEXT,
     job_url TEXT,
     description TEXT,
     embedding VECTOR(768)  -- For semantic search
   );
   ```
3. Note your `SUPABASE_URL` and `SUPABASE_API_KEY`

### 3. Backend Setup

```bash
cd backend

# Create .env file
cat > .env << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
ML_SERVICE_URL=http://localhost:8000
EOF

# Build and run
./gradlew bootRun
```

The backend API will be available at `http://localhost:8080`

See [backend/README.md](backend/README.md) for detailed API documentation.

### 4. ML App Setup

```bash
cd app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt

# Download spaCy models
python -m spacy download en_core_web_lg

# Create .env file
cat > .env << EOF
HOST=0.0.0.0
PORT=8000
DEBUG=False
LOG_LEVEL=INFO
LOG_FORMAT=text
CORS_ORIGINS=*
EOF

# Run the service
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The ML API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

See [app/README.md](app/README.md) for detailed usage examples.

### 5. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

See [frontend/README.md](frontend/README.md) for more details.

### 6. Scrapper Setup (Optional)

```bash
cd scrapper

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
EMBEDDING_API_URL=your_encoder_cloud_function_url
EOF

# Run locally (for testing)
python main.py
```

For production, deploy to Google Cloud Functions.

### 7. Encoder Setup (Optional)

```bash
cd encoder

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
EOF

# Run locally (for testing)
functions-framework --target=main --debug
```

For production, deploy to Google Cloud Functions.

## API Endpoints

### Backend API (Port 8080)

Base URL: `http://localhost:8080/api/v1`

#### Resume Analysis
- `POST /resume/sections` - Classify CV sections
- `POST /resume/entities` - Extract contact info + entities
- `POST /resume/category` - Predict resume category

#### Job Matching
- `POST /jobs/match` - Match resume with jobs (returns top K)
- `GET /jobs` - Paginated job listings with filters

See [backend/README.md](backend/README.md) for complete API reference.

### ML API (Port 8000)

Base URL: `http://localhost:8000/api/v1`

#### Central Pipeline
- `POST /resume/parse` - Complete ML pipeline (all models)

#### Individual Models
- `POST /resume/extract-text` - Extract text from PDF
- `POST /resume/classify-sections` - Section classification
- `POST /resume/extract-entities` - Entity extraction (NER)
- `POST /resume/classify-category` - Category classification

#### Job Matching
- `POST /match/jobs` - Match resume with job descriptions

See [app/README.md](app/README.md) for detailed usage examples.

## Usage Examples

### Complete Resume Analysis

```bash
# Using Backend API (calls ML service internally)
curl -X POST "http://localhost:8080/api/v1/resume/category" \
  -F "file=@resume.pdf"
```

Response:
```json
{
  "success": true,
  "data": {
    "category": "Software Engineer",
    "confidence": 0.92,
    "top_3": [
      {"category": "Software Engineer", "confidence": 0.92},
      {"category": "Full Stack Developer", "confidence": 0.85},
      {"category": "Backend Engineer", "confidence": 0.78}
    ]
  },
  "timestamp": 1700000000000
}
```

### Job Matching

```bash
# Match resume with top 10 jobs
curl -X POST "http://localhost:8080/api/v1/jobs/match?topK=10" \
  -F "file=@resume.pdf"
```

### Job Search

```bash
# Search for remote data science jobs
curl "http://localhost:8080/api/v1/jobs?category=Data%20Science&location=Remote&isRemote=true&size=10"
```

## Project Structure

```
career-match/
├── backend/                 # Spring Boot REST API
│   ├── src/main/java/
│   │   └── com/careermatch/backend/
│   │       ├── controller/  # REST controllers
│   │       ├── service/     # Business logic
│   │       ├── model/       # Entity models
│   │       └── config/      # Configuration
│   ├── build.gradle         # Build configuration
│   └── README.md
│
├── app/                     # FastAPI ML Service
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   ├── schemas/         # Pydantic models
│   │   └── services/        # Business logic
│   ├── models/              # ML models
│   │   ├── entity_extractor/
│   │   ├── section_classifier/
│   │   ├── resume_classifier/
│   │   └── job_matcher/
│   ├── utils/               # Utilities
│   ├── app.py               # FastAPI app
│   ├── config.py            # Configuration
│   └── README.md
│
├── frontend/                # React UI
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API clients
│   │   └── App.tsx
│   ├── package.json
│   └── README.md
│
├── scrapper/                # Job scraping service
│   ├── main.py              # Cloud function
│   ├── database.py          # Supabase client
│   └── requirements.txt
│
├── encoder/                 # Embedding generation service
│   ├── main.py              # Cloud function
│   ├── encoder.py           # Encoder model
│   └── requirements.txt
│
├── notebooks/               # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── resumes_exploration.ipynb
│
├── data/                    # Datasets
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
│
├── configs/                 # Configuration files
├── requirements.txt         # Python dependencies (root)
└── README.md               # This file
```

## Machine Learning Models

### 1. Section Classifier
- Classifies resume text into sections (contact, education, experience, skills, etc.)
- Uses rule-based patterns + ML classification
- Models: SBERT embeddings + classifier

### 2. Entity Extractor
- Extracts structured information using spaCy NER
- Entities: names, emails, phones, companies, job titles, skills, dates, locations
- Model: en_core_web_lg + custom patterns

### 3. Resume Classifier
- Classifies resumes into job categories
- Uses SBERT embeddings + neural network classifier
- Returns top-3 predictions with confidence scores
- Categories: Software Engineer, Data Scientist, Product Manager, etc.

### 4. Job Matcher
- Semantic similarity matching between resumes and jobs
- Uses sentence-transformers for encoding
- Stores vector embeddings for fast retrieval
- Returns ranked job matches

## Development

### Running Tests

```bash
# Backend tests
cd backend
./gradlew test

# ML App tests (if available)
cd app
pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production

#### Backend
```bash
cd backend
./gradlew build
java -jar build/libs/backend-0.0.1-SNAPSHOT.jar
```

#### ML App
```bash
cd app
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

#### Frontend
```bash
cd frontend
npm run build
# Serve the dist/ folder with a static file server
```

### Docker Deployment (Optional)

```bash
# Build and run backend
cd backend
docker build -t career-match-backend .
docker run -p 8080:8080 career-match-backend

# Build and run ML app
cd app
docker build -t career-match-ml .
docker run -p 8000:8000 career-match-ml

# Build and run frontend
cd frontend
docker build -t career-match-frontend .
docker run -p 80:80 career-match-frontend
```

## Environment Variables

### Backend (.env)
```
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
ML_SERVICE_URL=http://localhost:8000
```

### ML App (.env)
```
HOST=0.0.0.0
PORT=8000
DEBUG=False
LOG_LEVEL=INFO
LOG_FORMAT=text
CORS_ORIGINS=*
```

### Scrapper (.env)
```
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
EMBEDDING_API_URL=your_encoder_cloud_function_url
```

### Encoder (.env)
```
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_key
```

## Performance

- **Resume Parsing**: ~150-500ms per PDF
- **Section Classification**: ~100-300ms
- **Entity Extraction**: ~200-400ms
- **Category Classification**: ~300-500ms
- **Complete Pipeline**: ~500-1500ms (all models)
- **Job Matching**: ~100-200ms (with pre-computed embeddings)

## Logging

All services provide structured logging:

- Backend: SLF4J with Logback
- ML App: Python logging (JSON or text format)
- Scrapper/Encoder: Python logging

Set `LOG_FORMAT=json` for production log aggregation.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of an academic course at CU Boulder - OOPS (Object-Oriented Programming and Systems).

## Acknowledgments

- **Apache Tika** - PDF/DOCX parsing
- **spaCy** - Named Entity Recognition
- **Hugging Face Transformers** - SBERT models
- **sentence-transformers** - Semantic similarity
- **JobSpy** - Job scraping library
- **Supabase** - Database and vector storage
- **FastAPI** - Modern Python web framework
- **Spring Boot** - Java REST framework

## Contact

For questions or support, please refer to the course materials or contact the development team.

---

**Version**: 1.0.0
**Last Updated**: December 2025
**Course**: OOPS - CU Boulder
