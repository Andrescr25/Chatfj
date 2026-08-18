"""Endpoints de gestión de los documentos indexados (administradores)."""
import io
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from src.app.api.v1.endpoints.chat import get_chat_service
from src.app.core.security import CurrentUser, require_admin
from src.app.services.document_service import DocumentService

router = APIRouter()
logger = logging.getLogger(__name__)

_document_service = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        chat_service = get_chat_service()
        _document_service = DocumentService(
            chat_service.vector_store, chat_service.embedding_service
        )
    return _document_service


@router.get("/documents")
async def list_documents(
    include_deleted: bool = False,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    return {
        "documents": service.list_documents(include_deleted=include_deleted),
        "stats": service.stats(),
    }


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    """Estado de un documento. El panel consulta este endpoint para el progreso."""
    return service.get_document(doc_id)


@router.get("/documents/{doc_id}/content")
async def get_document_content(
    doc_id: str,
    offset: int = 0,
    limit: int = 20,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    """
    Texto indexado del documento, por fragmentos.

    Muestra lo que el asistente lee realmente al responder, que no siempre
    coincide con lo que se ve en el PDF (un PDF escaneado, por ejemplo, puede
    haber quedado sin texto útil).
    """
    return service.get_document_content(doc_id, offset=offset, limit=limit)


@router.post("/documents", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    title: str = Form(""),
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    """
    Sube un documento y lo indexa en segundo plano.

    Responde de inmediato con el registro en estado 'pendiente'; el progreso se
    consulta en GET /documents/{doc_id}.
    """
    content = await file.read()
    record = service.create_document(
        filename=file.filename,
        content=content,
        actor=user,
        category=category,
        title=title,
    )
    background_tasks.add_task(service.index_document, record["doc_id"])
    return record


@router.post("/documents/{doc_id}/reindex", status_code=202)
async def reindex_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    """Borra los vectores del documento y lo vuelve a indexar desde el archivo original."""
    record = service.reindex_document(doc_id, user)
    background_tasks.add_task(service.index_document, doc_id)
    return record


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    """Elimina del índice todos los fragmentos del documento y su archivo original."""
    return service.delete_document(doc_id, user)


@router.get("/documents/{doc_id}/download")
async def download_document(
    doc_id: str,
    user: CurrentUser = Depends(require_admin),
    service: DocumentService = Depends(get_document_service),
):
    content, filename = service.download_document(doc_id)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
