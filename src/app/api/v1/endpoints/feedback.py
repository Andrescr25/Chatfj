from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from src.app.schemas.feedback import FeedbackRequest
from src.app.services.chat_service import ChatService
from src.app.api.v1.endpoints.chat import get_chat_service
from firebase_admin import auth
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/feedback")
async def receive_feedback(
    request: FeedbackRequest,
    authorization: Optional[str] = Header(None),
    service: ChatService = Depends(get_chat_service)
):
    """
    Recibe feedback del usuario.
    Si el feedback incluye corrección explícita, se aprende (requiere autenticación de Firebase).
    """
    # Validar si contiene acciones administrativas de entrenamiento
    has_admin_actions = any(item.intent in ["correction", "expansion"] for item in request.items)
    
    if has_admin_actions:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="No autorizado: falta el token de sesión de Firebase."
            )
        
        token = authorization.split("Bearer ")[1]
        try:
            # Validar el ID token con Firebase
            decoded_token = auth.verify_id_token(token)
            email = decoded_token.get("email")
            logger.info(f"🔑 Solicitud de entrenamiento autorizada para: {email}")
        except Exception as e:
            logger.error(f"❌ Token de Firebase inválido o expirado: {e}")
            raise HTTPException(
                status_code=401,
                detail="Token de sesión de Firebase inválido o expirado."
            )

    try:
        count = 0
        for item in request.items:
            # Log basic feedback
            service.training_service.log_feedback(
                item.original_question, 
                item.feedback, 
                1 if item.intent != "negative" else -1
            )
            
            # Learn correction if intent suggests it
            if item.intent in ["correction", "expansion"]:
                service.training_service.learn_correction(item)
                count += 1
                
        return {"status": "success", "learned_items": count}
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error procesando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error saving feedback")
