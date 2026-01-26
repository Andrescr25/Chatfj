from fastapi import APIRouter, HTTPException, Depends
from src.app.schemas.feedback import FeedbackRequest
from src.app.services.chat_service import ChatService
from src.app.api.v1.endpoints.chat import get_chat_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/feedback")
async def receive_feedback(
    request: FeedbackRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Recibe feedback del usuario.
    Si el feedback incluye corrección explícita, se aprende.
    """
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
    except Exception as e:
        logger.error(f"Error procesando feedback: {e}")
        raise HTTPException(status_code=500, detail="Error saving feedback")
