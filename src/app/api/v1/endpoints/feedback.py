from typing import Optional

import logging
from fastapi import APIRouter, Depends, Header, HTTPException

from src.app.api.v1.endpoints.chat import get_chat_service
from src.app.core import audit
from src.app.core.security import get_current_user
from src.app.schemas.feedback import FeedbackRequest
from src.app.services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback")
async def receive_feedback(
    request: FeedbackRequest,
    authorization: Optional[str] = Header(None),
    service: ChatService = Depends(get_chat_service),
):
    """
    Recibe retroalimentación de las personas usuarias.

    El feedback simple es anónimo. Enseñarle una corrección al asistente es una
    acción administrativa: exige sesión válida y rol de administración.
    """
    has_admin_actions = any(item.intent in ["correction", "expansion"] for item in request.items)

    user = None
    if has_admin_actions:
        # Lanza 401/403 si el token falta, expiró o la cuenta no tiene rol.
        user = get_current_user(authorization)
        logger.info(f"🔑 Solicitud de entrenamiento autorizada para: {user.email} ({user.role})")

    try:
        count = 0
        for item in request.items:
            service.training_service.log_feedback(
                item.original_question,
                item.feedback,
                1 if item.intent != "negative" else -1,
            )

            if item.intent in ["correction", "expansion"]:
                # El nombre del entrenador queda atado a la cuenta autenticada.
                item.trainer_name = user.display_name
                service.training_service.learn_correction(item)
                count += 1

        if count and user:
            audit.log_action(
                "entrenamiento.correccion", user.uid, user.email,
                target=request.items[0].original_question[:80],
                details={"correcciones": count},
            )

        return {"status": "success", "learned_items": count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error saving feedback")
