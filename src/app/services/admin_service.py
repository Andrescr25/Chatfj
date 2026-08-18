"""
Gestión de personas administradoras sobre Firebase Auth.

Hay un único rol ('admin'), guardado como custom claim, que es lo que revisa
src/app/core/security.py en cada petición. Firestore guarda solo datos
complementarios (quién invitó a quién).
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from src.app.config import settings
from src.app.core import audit
from src.app.core.security import LEGACY_ROLES, ROLE_ADMIN, CurrentUser

logger = logging.getLogger(__name__)

ADMINS_COLLECTION = "admins"


def _ms_to_iso(ms: Optional[int]) -> str:
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return ""


class AdminService:
    def __init__(self):
        self._db = self._init_firestore()

    def _init_firestore(self):
        try:
            import firebase_admin
            from firebase_admin import firestore

            if firebase_admin._apps:
                return firestore.client()
        except Exception as e:
            logger.warning(f"⚠️ Firestore no disponible para datos de administradores: {e}")
        return None

    # ---------- Lectura ----------

    def _metadata(self, uid: str) -> Dict[str, Any]:
        if not self._db:
            return {}
        try:
            snap = self._db.collection(ADMINS_COLLECTION).document(uid).get()
            return snap.to_dict() or {} if snap.exists else {}
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer metadata de {uid}: {e}")
            return {}

    def _to_record(self, user) -> Dict[str, Any]:
        claims = user.custom_claims or {}
        email = (user.email or "").lower()
        protected = email in settings.admin_emails
        claim_role = claims.get("role", "")
        es_admin = protected or claim_role == ROLE_ADMIN or claim_role in LEGACY_ROLES
        meta = self._metadata(user.uid)
        return {
            "uid": user.uid,
            "email": email,
            "name": user.display_name or meta.get("name", "") or "",
            "role": ROLE_ADMIN if es_admin else "",
            "disabled": bool(user.disabled),
            "created_at": _ms_to_iso(getattr(user.user_metadata, "creation_timestamp", None)),
            "last_sign_in": _ms_to_iso(getattr(user.user_metadata, "last_sign_in_timestamp", None)),
            "invited_by": meta.get("invited_by", ""),
            "protected": protected,
        }

    def list_admins(self) -> List[Dict[str, Any]]:
        """Devuelve solo las cuentas con rol de administración."""
        from firebase_admin import auth

        records: List[Dict[str, Any]] = []
        try:
            for user in auth.list_users().iterate_all():
                record = self._to_record(user)
                if record["role"]:
                    records.append(record)
        except Exception as e:
            logger.error(f"❌ Error listando usuarios de Firebase: {e}")
            raise HTTPException(status_code=502, detail="No se pudo consultar Firebase Auth.")

        records.sort(key=lambda r: r["email"])
        return records

    def count_active_admins(self, exclude_uid: str = "") -> int:
        return len([a for a in self.list_admins() if not a["disabled"] and a["uid"] != exclude_uid])

    # ---------- Escritura ----------

    def create_admin(
        self,
        email: str,
        name: str,
        password: Optional[str],
        actor: CurrentUser,
    ) -> Dict[str, Any]:
        from firebase_admin import auth

        try:
            existing = auth.get_user_by_email(email)
        except auth.UserNotFoundError:
            existing = None
        except Exception as e:
            logger.error(f"❌ Error consultando usuario por correo: {e}")
            raise HTTPException(status_code=502, detail="No se pudo consultar Firebase Auth.")

        if existing:
            if self._to_record(existing)["role"]:
                raise HTTPException(
                    status_code=409,
                    detail="Esa persona ya tiene acceso administrativo.",
                )
            user = existing
            if name and not user.display_name:
                auth.update_user(user.uid, display_name=name)
            if user.disabled:
                auth.update_user(user.uid, disabled=False)
        else:
            try:
                user = auth.create_user(
                    email=email,
                    display_name=name or None,
                    password=password or None,
                    email_verified=False,
                )
            except Exception as e:
                logger.error(f"❌ Error creando usuario: {e}")
                raise HTTPException(status_code=400, detail=f"No se pudo crear la cuenta: {e}")

        auth.set_custom_user_claims(user.uid, {"role": ROLE_ADMIN})

        reset_link = ""
        if not password:
            try:
                reset_link = auth.generate_password_reset_link(email)
            except Exception as e:
                logger.warning(f"⚠️ No se pudo generar el enlace de contraseña: {e}")

        if self._db:
            try:
                self._db.collection(ADMINS_COLLECTION).document(user.uid).set({
                    "uid": user.uid,
                    "email": email,
                    "name": name,
                    "role": ROLE_ADMIN,
                    "invited_by": actor.email or actor.uid,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(f"⚠️ No se pudo guardar metadata del administrador: {e}")

        audit.log_action("admin.create", actor.uid, actor.email, target=email)

        record = self._to_record(auth.get_user(user.uid))
        record["password_reset_link"] = reset_link
        return record

    def update_admin(
        self,
        uid: str,
        actor: CurrentUser,
        disabled: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        from firebase_admin import auth

        if uid == actor.uid and disabled is not None:
            raise HTTPException(
                status_code=400,
                detail="No puede deshabilitar su propia cuenta.",
            )

        try:
            user = auth.get_user(uid)
        except auth.UserNotFoundError:
            raise HTTPException(status_code=404, detail="La cuenta no existe.")

        current = self._to_record(user)

        if disabled is True:
            if current["protected"]:
                raise HTTPException(
                    status_code=400,
                    detail="Esta cuenta está protegida por configuración del servidor "
                           "(ADMIN_EMAILS). Quítela de esa lista antes de deshabilitarla.",
                )
            if self.count_active_admins(exclude_uid=uid) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Debe quedar al menos una persona administradora activa.",
                )

        if disabled is not None:
            auth.update_user(uid, disabled=disabled)
            if disabled:
                auth.revoke_refresh_tokens(uid)
        if name is not None:
            auth.update_user(uid, display_name=name)

        audit.log_action(
            "admin.update", actor.uid, actor.email, target=current["email"],
            details={"disabled": disabled, "name": name},
        )
        return self._to_record(auth.get_user(uid))

    def remove_admin(self, uid: str, actor: CurrentUser) -> Dict[str, Any]:
        """
        Revoca el acceso: quita el rol, deshabilita la cuenta y corta las sesiones
        activas. No borra el usuario de Firebase para no perder el rastro.
        """
        from firebase_admin import auth

        if uid == actor.uid:
            raise HTTPException(status_code=400, detail="No puede eliminar su propia cuenta.")

        try:
            user = auth.get_user(uid)
        except auth.UserNotFoundError:
            raise HTTPException(status_code=404, detail="La cuenta no existe.")

        current = self._to_record(user)

        if current["protected"]:
            raise HTTPException(
                status_code=400,
                detail="Esta cuenta está protegida por configuración del servidor "
                       "(ADMIN_EMAILS). Quítela de esa lista antes de revocarla.",
            )

        if self.count_active_admins(exclude_uid=uid) == 0:
            raise HTTPException(
                status_code=400,
                detail="Debe quedar al menos una persona administradora activa.",
            )

        auth.set_custom_user_claims(uid, {})
        auth.update_user(uid, disabled=True)
        auth.revoke_refresh_tokens(uid)

        if self._db:
            try:
                self._db.collection(ADMINS_COLLECTION).document(uid).update({
                    "role": "",
                    "revoked_by": actor.email or actor.uid,
                    "revoked_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning(f"⚠️ No se pudo actualizar metadata al revocar: {e}")

        audit.log_action("admin.revoke", actor.uid, actor.email, target=current["email"])
        return {"status": "success", "uid": uid, "email": current["email"]}
