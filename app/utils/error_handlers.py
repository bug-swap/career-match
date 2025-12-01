import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors (422 Unprocessable Entity)
    
    Args:
        request: FastAPI request object
        exc: Validation error exception
        
    Returns:
        JSON response with validation error details
    """
    errors = []
    for error in exc.errors():
        errors.append({
            'field': '.'.join(str(x) for x in error['loc']),
            'message': error['msg'],
            'type': error['type']
        })
    
    logger.warning(
        f"Validation error on {request.method} {request.url.path}",
        extra={
            'method': request.method,
            'path': request.url.path,
            'errors': errors
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            'success': False,
            'error': 'Validation error',
            'type': 'ValidationError',
            'details': errors
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions (4xx, 5xx)
    
    Args:
        request: FastAPI request object
        exc: HTTP exception
        
    Returns:
        JSON response with error details
    """
    logger.warning(
        f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}",
        extra={
            'method': request.method,
            'path': request.url.path,
            'status_code': exc.status_code,
            'error': str(exc.detail)
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'success': False,
            'error': str(exc.detail),
            'type': exc.__class__.__name__
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle all unhandled exceptions (500 Internal Server Error)
    
    Args:
        request: FastAPI request object
        exc: Any exception
        
    Returns:
        JSON response with error details
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        extra={
            'method': request.method,
            'path': request.url.path,
            'error': str(exc)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            'success': False,
            'error': 'Internal server error',
            'type': exc.__class__.__name__,
            'message': str(exc)
        }
    )


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle 404 Not Found errors for undefined routes
    
    Args:
        request: FastAPI request object
        exc: Exception (can be None for catch-all)
        
    Returns:
        JSON response with 404 error
    """
    logger.warning(
        f"Route not found: {request.method} {request.url.path}",
        extra={
            'method': request.method,
            'path': request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            'success': False,
            'error': f'Route not found: {request.method} {request.url.path}',
            'type': 'NotFound',
            'message': 'The requested endpoint does not exist. Check the API documentation at /docs'
        }
    )