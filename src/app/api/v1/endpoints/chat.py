from fastapi import APIRouter, HTTPException, Depends
from src.app.schemas.chat import QueryRequest, QueryResponse
from src.app.services.chat_service import ChatService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency to get Chat Service (Singleton-ish pattern or plain init)
# For simplicity in this app, we can instantiate or use a global.
# Ideally use lru_cache for dependency injection.
_chat_service = None

def get_chat_service():
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service

@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Endpoint principal de consulta.
    Procesa la pregunta, busca contexto y genera respuesta.
    """
    try:
        response = await service.get_response(request.question, request.history)
        return response
    except Exception as e:
        logger.error(f"Error procesando pregunta: {e}")
        # Return graceful error to frontend
        return QueryResponse(
            answer="Lo siento, ocurrió un error técnico momentáneo. Por favor intentá preguntar de nuevo en unos segundos.",
            sources=[],
            processing_time=0.0
        )
