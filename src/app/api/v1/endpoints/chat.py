import logging

from fastapi import APIRouter, Depends

from src.app.api.v1.deps import get_chat_service
from src.app.schemas.chat import QueryRequest, QueryResponse
from src.app.services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)

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
