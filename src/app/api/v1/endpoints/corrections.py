"""Endpoints de las correcciones aprendidas (administración)."""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.app.api.v1.deps import get_corrections_service
from src.app.core.security import CurrentUser, require_admin
from src.app.services.corrections_service import CorrectionsService

router = APIRouter()
logger = logging.getLogger(__name__)


class CorrectionUpdateRequest(BaseModel):
    pregunta: str = None
    correccion: str = None


@router.get("/corrections")
async def listar_correcciones(
    user: CurrentUser = Depends(require_admin),
    service: CorrectionsService = Depends(get_corrections_service),
):
    """Todas las correcciones que el asistente aplica antes que los documentos."""
    return service.listar()


@router.patch("/corrections/{correccion_id}")
async def editar_correccion(
    correccion_id: str,
    request: CorrectionUpdateRequest,
    user: CurrentUser = Depends(require_admin),
    service: CorrectionsService = Depends(get_corrections_service),
):
    return service.actualizar(
        correccion_id, actor=user, pregunta=request.pregunta, correccion=request.correccion
    )


@router.delete("/corrections/{correccion_id}")
async def eliminar_correccion(
    correccion_id: str,
    user: CurrentUser = Depends(require_admin),
    service: CorrectionsService = Depends(get_corrections_service),
):
    return service.eliminar(correccion_id, user)
