from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from config import settings
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup: Load ML models
    logger.info("🚀 Starting application...")
    from api.core.loader import ModelLoader
    try:
        ModelLoader.get_instance()
        logger.info("✅ All models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Career Match ML Service",
        description="ML service for resume parsing, classification, and job matching",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests and responses"""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"➡️  {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
                "query_params": dict(request.query_params)
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"⬅️  {request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration, 2)
                }
            )
            
            return response
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"❌ {request.method} {request.url.path} - Error - {duration:.2f}ms: {str(e)}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration, 2)
                },
                exc_info=True
            )
            raise
    
    # Register error handlers
    from utils.error_handlers import (
        validation_exception_handler,
        http_exception_handler,
        general_exception_handler,
        not_found_handler
    )
    
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Register routers
    from api.routers import classification, health, matching, resume
    
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(resume.router, prefix="/api/v1/resume", tags=["Resume"])
    app.include_router(classification.router, prefix="/api/v1/classify", tags=["Classification"])
    app.include_router(matching.router, prefix="/api/v1/match", tags=["Matching"])
    
    # Custom 404 handler for undefined routes
    @app.get("/{full_path:path}", include_in_schema=False)
    async def catch_all(request: Request, full_path: str):
        """Catch-all route for undefined paths"""
        return await not_found_handler(request, None)
    
    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('DEBUG', 'False').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
