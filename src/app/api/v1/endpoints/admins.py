"""Endpoints de gestión de personas administradoras."""
import logging

from fastapi import APIRouter, Depends

from src.app.api.v1.deps import get_admin_service
from src.app.core.security import CurrentUser, require_admin
from src.app.schemas.admin import AdminCreateRequest, AdminUpdateRequest
from src.app.services.admin_service import AdminService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/admins/me")
async def whoami(user: CurrentUser = Depends(require_admin)):
    """Identidad de quien está autenticado. Lo usa el panel para saludar y auditar."""
    return {
        "uid": user.uid,
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }


@router.get("/admins")
async def list_admins(
    user: CurrentUser = Depends(require_admin),
    service: AdminService = Depends(get_admin_service),
):
    return {"admins": service.list_admins()}


@router.post("/admins", status_code=201)
async def create_admin(
    request: AdminCreateRequest,
    user: CurrentUser = Depends(require_admin),
    service: AdminService = Depends(get_admin_service),
):
    """
    Da acceso administrativo a una persona.

    Si no se envía contraseña, se devuelve un enlace de Firebase para que la
    persona defina la suya; así nadie más conoce su clave.
    """
    return service.create_admin(
        email=request.email,
        name=request.name,
        password=request.password,
        actor=user,
    )


@router.patch("/admins/{uid}")
async def update_admin(
    uid: str,
    request: AdminUpdateRequest,
    user: CurrentUser = Depends(require_admin),
    service: AdminService = Depends(get_admin_service),
):
    return service.update_admin(
        uid=uid,
        actor=user,
        disabled=request.disabled,
        name=request.name,
    )


@router.delete("/admins/{uid}")
async def remove_admin(
    uid: str,
    user: CurrentUser = Depends(require_admin),
    service: AdminService = Depends(get_admin_service),
):
    """Revoca el acceso: quita el rol, deshabilita la cuenta y corta las sesiones activas."""
    return service.remove_admin(uid, user)
