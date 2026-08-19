from fastapi import APIRouter

from src.app.api.v1.endpoints import (
    admins,
    analytics,
    chat,
    corrections,
    diagnostics,
    documents,
    feedback,
)

api_router = APIRouter()

api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(feedback.router, tags=["feedback"])
api_router.include_router(documents.router, tags=["documentos"])
api_router.include_router(admins.router, tags=["administradores"])
api_router.include_router(corrections.router, tags=["correcciones"])
api_router.include_router(analytics.router, tags=["analítica"])
api_router.include_router(diagnostics.router, tags=["diagnóstico"])
