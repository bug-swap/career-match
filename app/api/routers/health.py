from fastapi import APIRouter

from api.core.loader import ModelLoader
from api.schemas.response import HealthResponse, ModelsStatusResponse

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
async def health_check() -> dict:
    return {'status': 'healthy', 'version': '2.0.0'}


@router.get('/models/status', response_model=ModelsStatusResponse)
async def models_status() -> dict:
    return ModelLoader.get_instance().get_status()
