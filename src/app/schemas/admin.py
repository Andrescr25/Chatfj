import re
from typing import Optional

from pydantic import BaseModel, field_validator

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AdminCreateRequest(BaseModel):
    email: str
    name: str = ""
    # Si no se envía contraseña, se genera un enlace para que la persona defina la suya.
    password: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validar_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Formato de correo electrónico inválido.")
        return v

    @field_validator("password")
    @classmethod
    def validar_password(cls, v: Optional[str]) -> Optional[str]:
        # Cadena vacía = "sin contraseña": se enviará enlace para definirla.
        if v is not None and v.strip() == "":
            return None
        if v is not None and len(v) < 8:
            raise ValueError("La contraseña debe tener al menos 8 caracteres.")
        return v


class AdminUpdateRequest(BaseModel):
    disabled: Optional[bool] = None
    name: Optional[str] = None


class AdminResponse(BaseModel):
    uid: str
    email: str = ""
    name: str = ""
    role: str = "admin"
    disabled: bool = False
    created_at: str = ""
    last_sign_in: str = ""
    invited_by: str = ""
    protected: bool = False  # administrador por configuración del servidor: no se revoca desde el panel
