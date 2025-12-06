# Backend API Quick Reference

All endpoints are served from `http://localhost:8080/api/v1` when running `./gradlew bootRun` inside `backend/`.

## Resume Analysis

| Method | Path | Description | Body |
| --- | --- | --- | --- |
| `POST` | `/resume/sections` | Classify CV sections via Python ML service | `multipart/form-data` with `file=<resume.pdf>` |
| `POST` | `/resume/entities` | Extract contact info + skills/entities | `multipart/form-data` with `file=<resume.pdf>` |
| `POST` | `/resume/category` | Predict resume category with confidence and top-3 classes | `multipart/form-data` with `file=<resume.pdf>` |

### cURL Example
```bash
curl -X POST "http://localhost:8080/api/v1/resume/entities" \
  -H "Accept: application/json" \
  -F "file=@/path/to/resume.pdf"
```

## Job Matching & Listings

| Method | Path | Description | Body/Params |
| --- | --- | --- | --- |
| `POST` | `/jobs/match` | Upload a resume and receive ranked job matches | `multipart/form-data`: `file=<resume.pdf>`, optional `topK` (default `10`) |
| `GET` | `/jobs` | Paginated job list with filters | Query params (all optional unless noted):
- `category`
- `location`
- `jobType`
- `isRemote`
- `searchQuery`
- `page` (default `0`)
- `size` (default `20`)
- `sortBy` (default `datePosted`)
- `sortOrder` (`ASC`/`DESC`, default `DESC`)

### GET example
```bash
curl "http://localhost:8080/api/v1/jobs?category=Data%20Science&location=Remote&isRemote=true&size=5"
```

### Match example
```bash
curl -X POST "http://localhost:8080/api/v1/jobs/match?topK=15" \
  -H "Accept: application/json" \
  -F "file=@/path/to/resume.pdf"
```

## Response Wrapping
All successful endpoints return the standard envelope:
```json
{
  "success": true,
  "data": { ...endpoint specific payload... },
  "timestamp": 1700000000000
}
```
Errors respond with `success=false` and a message describing the failure.
