"""Endpoints de uso: documentos más consultados e historial (administración)."""
import logging

from fastapi import APIRouter, Depends

from src.app.api.v1.deps import get_analytics_service
from src.app.core.security import CurrentUser, require_admin
from src.app.services.analytics_service import AnalyticsService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats/documents")
async def documentos_mas_consultados(
    limite: int = 25,
    user: CurrentUser = Depends(require_admin),
    service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Qué documentos sostienen las respuestas.

    Solo cuenta desde que se activó el registro: los documentos indexados antes
    aparecen en cero hasta que alguien pregunte algo que los use.
    """
    return service.documentos_mas_consultados(limite=min(max(limite, 1), 100))


@router.get("/history")
async def historial(
    dias: int = 7,
    limite: int = 200,
    user: CurrentUser = Depends(require_admin),
    service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Consultas de los últimos días, de la más reciente a la más antigua.

    Se guardan la pregunta, la respuesta y los documentos usados. No se guarda
    nada que identifique a quien preguntó.
    """
    return service.historial(dias=min(max(dias, 1), 7), limite=min(max(limite, 1), 500))
