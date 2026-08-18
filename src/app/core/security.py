"""
Autenticación y autorización de administradores.

Hay un único rol: 'admin'. Quien lo tiene puede entrenar el asistente, gestionar
los documentos indexados y dar o quitar acceso a otras personas.

El rol vive en los custom claims de Firebase Auth. Como respaldo, los correos
listados en ADMIN_EMAILS siempre se consideran administradores; eso permite
arrancar el sistema y evita quedarse sin acceso si un claim se pierde.
"""
import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel

from src.app.config import settings

logger = logging.getLogger(__name__)

ROLE_ADMIN = "admin"
VALID_ROLES = (ROLE_ADMIN,)
# Rol de una versión anterior que tenía dos niveles. Se acepta como 'admin'
# para que nadie quede fuera tras la unificación.
LEGACY_ROLES = ("superadmin",)


class CurrentUser(BaseModel):
    uid: str
    email: str = ""
    name: str = ""
    role: str = ROLE_ADMIN

    @property
    def display_name(self) -> str:
        return self.name or self.email or self.uid


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="No autorizado: falta el token de sesión de Firebase.",
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="No autorizado: token vacío.")
    return token


def get_current_user(authorization: Optional[str] = Header(None)) -> CurrentUser:
    """Verifica el ID token de Firebase y confirma que la cuenta es administradora."""
    from firebase_admin import auth  # import diferido: Firebase se inicia en el startup

    token = _extract_bearer(authorization)

    try:
        decoded = auth.verify_id_token(token, check_revoked=True)
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="La sesión fue revocada. Inicie sesión de nuevo.")
    except auth.UserDisabledError:
        raise HTTPException(status_code=403, detail="La cuenta está deshabilitada.")
    except Exception as e:
        logger.error(f"❌ Token de Firebase inválido o expirado: {e}")
        raise HTTPException(status_code=401, detail="Token de sesión inválido o expirado.")

    email = (decoded.get("email") or "").lower()
    role = decoded.get("role")

    if role in LEGACY_ROLES:
        role = ROLE_ADMIN

    # Respaldo de arranque: allowlist por correo
    if email and email in settings.admin_emails:
        role = ROLE_ADMIN

    if role not in VALID_ROLES:
        logger.warning(f"🚫 Acceso denegado para {email or decoded.get('uid')}: sin rol de administración.")
        raise HTTPException(
            status_code=403,
            detail="Su cuenta no tiene permisos de administración en Chat FJ.",
        )

    return CurrentUser(
        uid=decoded.get("uid", ""),
        email=email,
        name=decoded.get("name", "") or decoded.get("displayName", "") or "",
        role=ROLE_ADMIN,
    )


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Única puerta de autorización del sistema."""
    return user
