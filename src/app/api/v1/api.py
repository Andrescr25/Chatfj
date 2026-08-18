from fastapi import APIRouter
from src.app.api.v1.endpoints import admins, chat, documents, feedback

api_router = APIRouter()

api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(feedback.router, tags=["feedback"])
api_router.include_router(documents.router, tags=["documentos"])
api_router.include_router(admins.router, tags=["administradores"])
