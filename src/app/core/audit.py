"""
Bitácora de acciones administrativas.

Toda acción sensible (alta o baja de administradores, subida o eliminación de
documentos) queda registrada con quién la hizo y cuándo. Es el respaldo del
compromiso de transparencia: se puede reconstruir qué información entró o salió
del índice y por decisión de quién.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

COLLECTION = "audit_log"
LOCAL_PATH = "logs/audit_log.jsonl"
_lock = threading.Lock()


def log_action(
    action: str,
    actor_uid: str,
    actor_email: str = "",
    target: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Registra una acción administrativa. Nunca interrumpe la operación principal."""
    entry = {
        "action": action,
        "actor_uid": actor_uid,
        "actor_email": actor_email,
        "target": target,
        "details": details or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"📝 AUDITORÍA: {action} | por {actor_email or actor_uid} | sobre {target}")

    try:
        import firebase_admin
        from firebase_admin import firestore

        if firebase_admin._apps:
            firestore.client().collection(COLLECTION).add(entry)
            return
    except Exception as e:
        logger.warning(f"⚠️ No se pudo escribir auditoría en Firestore: {e}")

    try:
        with _lock:
            os.makedirs(os.path.dirname(LOCAL_PATH) or ".", exist_ok=True)
            with open(LOCAL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo escribir auditoría local: {e}")
