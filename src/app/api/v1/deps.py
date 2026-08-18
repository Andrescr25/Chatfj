"""
Dependencias compartidas por los endpoints.

Los servicios son costosos de construir (abren conexión a Pinecone y prueban la
API de embeddings), así que se crean una sola vez y se reutilizan. Antes cada
endpoint tenía su propio singleton global, y documents.py importaba el de
chat.py para reaprovechar la conexión.
"""
from functools import lru_cache

from src.app.services.chat_service import ChatService
from src.app.services.document_service import DocumentService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService()


@lru_cache(maxsize=1)
def get_document_service() -> DocumentService:
    # Comparte el almacén vectorial y el cliente de embeddings con el chat:
    # duplicarlos significaría abrir dos conexiones y probar la API dos veces.
    chat = get_chat_service()
    return DocumentService(chat.vector_store, chat.embedding_service)


@lru_cache(maxsize=1)
def get_admin_service():
    from src.app.services.admin_service import AdminService

    return AdminService()
