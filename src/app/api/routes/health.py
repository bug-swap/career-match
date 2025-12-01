from fastapi import APIRouter
from api.models.loader import ModelLoader
from api.schemas.response import HealthResponse, ModelsStatusResponse

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint
    
    Returns service status and version information.
    """
    return {
        'status': 'healthy',
        'version': '2.0.0'
    }


@router.get('/models/status', response_model=ModelsStatusResponse)
async def models_status():
    """
    Check ML models loading status
    
    Returns the loading status of all ML models and whether all are loaded.
    """
    loader = ModelLoader.get_instance()
    status = loader.get_status()
    
    return status