"""
Gestión de los documentos indexados en Pinecone.

Cada documento tiene un prefijo propio de IDs de vectores
('doc::{doc_id}::{n}'), lo que permite eliminarlo completo sin tocar el resto
del índice. Pinecone serverless no admite borrar por filtro de metadatos: por
eso el catálogo (src/app/core/registry.py) es indispensable.

Los documentos indexados antes de este panel conservan sus prefijos originales
('{archivo}_chunk_' y 'docs_new__{archivo}_chunk_'); el script
scripts/backfill_documents_registry.py los registra para que también se puedan
administrar desde aquí.
"""
import hashlib
import logging
import re
import tempfile
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from src.app.config import settings
from src.app.core import audit
from src.app.core.rag.loaders import SUPPORTED_EXTENSIONS, chunk_text, read_file
from src.app.core.registry import (
    STATUS_DELETED,
    STATUS_ERROR,
    STATUS_INDEXED,
    STATUS_INDEXING,
    STATUS_PENDING,
    DocumentRegistry,
    utcnow_iso,
)
from src.app.core.security import CurrentUser
from src.app.core.storage import DocumentStorage

logger = logging.getLogger(__name__)

DOC_ID_PREFIX = "doc::"


def slugify(name: str) -> str:
    base = Path(name).stem
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    return (base or "documento")[:40]


class DocumentService:
    def __init__(self, vector_store, embedding_service=None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service or vector_store.embedding_service
        self.registry = DocumentRegistry()
        self.storage = DocumentStorage()

    # ---------- Consulta ----------

    def list_documents(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        return self.registry.list(include_deleted=include_deleted)

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        record = self.registry.get(doc_id)
        if not record:
            raise HTTPException(status_code=404, detail="El documento no existe en el catálogo.")
        return record

    def stats(self) -> Dict[str, Any]:
        documentos = self.list_documents()
        index_stats = self.vector_store.index_stats()
        namespaces = index_stats.get("namespaces", {}) or {}
        default_ns = namespaces.get("", {}) or namespaces.get("__default__", {}) or {}
        corrections_ns = namespaces.get("corrections", {}) or {}
        return {
            "documentos": len(documentos),
            "fragmentos_catalogados": sum(d.get("chunks", 0) or 0 for d in documentos),
            "vectores_en_indice": default_ns.get("vector_count", 0),
            "correcciones_aprendidas": corrections_ns.get("vector_count", 0),
            "catalogo": self.registry.backend,
            "almacenamiento": "firebase" if self.storage.bucket else "local",
        }

    # ---------- Alta ----------

    def create_document(
        self,
        filename: str,
        content: bytes,
        actor: CurrentUser,
        category: str = "general",
        title: str = "",
    ) -> Dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no admitido ({ext}). Use: {', '.join(SUPPORTED_EXTENSIONS)}",
            )

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"El archivo supera el límite de {settings.MAX_UPLOAD_MB} MB.",
            )
        if not content:
            raise HTTPException(status_code=400, detail="El archivo está vacío.")

        file_hash = hashlib.sha256(content).hexdigest()
        duplicado = self.registry.find_by_hash(file_hash)
        if duplicado:
            raise HTTPException(
                status_code=409,
                detail=f"Ese mismo archivo ya está indexado como '{duplicado.get('filename')}'.",
            )

        doc_id = f"{slugify(filename)}-{uuid.uuid4().hex[:8]}"
        stored = self.storage.save(doc_id, filename, content)

        record = {
            "doc_id": doc_id,
            "filename": filename,
            "title": title or Path(filename).stem,
            "category": category or "general",
            "extension": ext,
            "size_bytes": len(content),
            "file_hash": file_hash,
            "vector_prefix": f"{DOC_ID_PREFIX}{doc_id}::",
            "chunks": 0,
            "chunks_total": 0,
            "status": STATUS_PENDING,
            "error": "",
            "uploaded_by": actor.email or actor.uid,
            "uploaded_at": utcnow_iso(),
            "updated_at": utcnow_iso(),
            "legacy": False,
            **stored,
        }
        self.registry.save(record)

        audit.log_action(
            "documento.subir", actor.uid, actor.email, target=filename,
            details={"doc_id": doc_id, "size_bytes": len(content), "category": category},
        )
        return record

    # ---------- Indexación ----------

    def index_document(self, doc_id: str) -> Dict[str, Any]:
        """Indexa un documento del catálogo. Pensado para correr en segundo plano."""
        record = self.registry.get(doc_id)
        if not record:
            logger.error(f"❌ Documento {doc_id} no está en el catálogo.")
            return {"status": STATUS_ERROR, "error": "documento no encontrado"}

        self.registry.update(doc_id, status=STATUS_INDEXING, error="", chunks=0)

        try:
            content = self.storage.load(record)
            if not content:
                raise RuntimeError("No se encontró el archivo original.")

            with tempfile.NamedTemporaryFile(suffix=record.get("extension", ""), delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                text = read_file(Path(tmp.name))

            if not text.strip():
                raise RuntimeError(
                    "No se pudo extraer texto del documento. "
                    "Si es un PDF escaneado, necesita reconocimiento de texto (OCR) previo."
                )

            chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            if not chunks:
                raise RuntimeError("El documento no produjo fragmentos indexables.")

            self.registry.update(doc_id, chunks_total=len(chunks))
            prefix = record["vector_prefix"]
            base_metadata = {
                "source": record["filename"],
                "filename": record["filename"],
                "doc_id": doc_id,
                "categoria": record.get("category", "general"),
                "uploaded_by": record.get("uploaded_by", ""),
                "uploaded_at": record.get("uploaded_at", ""),
            }

            batch_size = settings.EMBED_BATCH_SIZE
            subidos = 0
            for i in range(0, len(chunks), batch_size):
                lote = chunks[i:i + batch_size]
                vectores = self._embed_with_retry(lote)
                if not vectores:
                    raise RuntimeError(
                        "El servicio de embeddings (HuggingFace) no respondió. Intente de nuevo en unos minutos."
                    )

                payload = [
                    (f"{prefix}{i + j}", vec, {**base_metadata, "text": lote[j]})
                    for j, vec in enumerate(vectores)
                ]
                if not self.vector_store.upsert_vectors(payload, namespace=""):
                    raise RuntimeError("Pinecone rechazó la subida de vectores.")

                subidos += len(payload)
                self.registry.update(doc_id, chunks=subidos)
                time.sleep(0.5)  # respiro entre lotes para no chocar con el límite de la API

            result = self.registry.update(
                doc_id, status=STATUS_INDEXED, chunks=subidos, chunks_total=len(chunks),
                indexed_at=utcnow_iso(), error="",
            )
            logger.info(f"✅ Documento '{record['filename']}' indexado: {subidos} fragmentos.")
            return result or {}

        except Exception as e:
            logger.error(f"❌ Error indexando {doc_id}: {e}")
            self.registry.update(doc_id, status=STATUS_ERROR, error=str(e))
            return {"status": STATUS_ERROR, "error": str(e)}

    def _embed_with_retry(self, textos: List[str], intentos: int = 3) -> List[List[float]]:
        for intento in range(intentos):
            try:
                vectores = self.embedding_service.embed_documents(textos)
                if vectores and len(vectores) == len(textos):
                    return vectores
                logger.warning(
                    f"⚠️ Embeddings incompletos (intento {intento + 1}/{intentos}): "
                    f"{len(vectores) if vectores else 0}/{len(textos)}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error de embeddings (intento {intento + 1}/{intentos}): {e}")
            time.sleep(5 * (intento + 1))
        return []

    def reindex_document(self, doc_id: str, actor: CurrentUser) -> Dict[str, Any]:
        record = self.get_document(doc_id)
        if record.get("status") == STATUS_DELETED:
            raise HTTPException(status_code=400, detail="El documento fue eliminado.")
        # Sin el archivo original, reindexar borraría los vectores sin poder
        # reconstruirlos. Es el caso de los documentos de la ingesta inicial.
        if not record.get("storage_path"):
            raise HTTPException(
                status_code=400,
                detail="Este documento no tiene archivo original guardado, así que no se "
                       "puede reindexar. Para actualizarlo, elimínelo y vuelva a subirlo.",
            )
        self._delete_vectors(record)
        self.registry.update(doc_id, status=STATUS_PENDING, chunks=0, error="")
        audit.log_action(
            "documento.reindexar", actor.uid, actor.email,
            target=record.get("filename", doc_id), details={"doc_id": doc_id},
        )
        return self.get_document(doc_id)

    # ---------- Baja ----------

    def _resolve_vector_ids(self, record: Dict[str, Any]) -> List[str]:
        """
        Obtiene los IDs de vectores del documento.

        Primero se consulta Pinecone por prefijo. Si el índice no admite el
        listado (pod-based), se reconstruyen a partir del conteo guardado en el
        catálogo, ya que los IDs son deterministas ('{prefijo}{n}').
        """
        prefix = record.get("vector_prefix")
        if not prefix:
            return []

        ids = self.vector_store.list_vector_ids(prefix, namespace="")
        if ids is not None:
            return ids

        total = record.get("chunks") or record.get("chunks_total") or 0
        if not total:
            raise HTTPException(
                status_code=500,
                detail="No se pudieron determinar los fragmentos del documento. "
                       "Ejecute el script de reconciliación del catálogo.",
            )
        logger.warning(f"⚠️ Listado por prefijo no disponible: se reconstruyen {total} IDs.")
        return [f"{prefix}{i}" for i in range(total)]

    def _delete_vectors(self, record: Dict[str, Any]) -> int:
        ids = self._resolve_vector_ids(record)
        if not ids:
            return 0
        try:
            return self.vector_store.delete_vectors(ids, namespace="")
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"No se pudieron eliminar los vectores en Pinecone: {e}",
            )

    def delete_document(self, doc_id: str, actor: CurrentUser) -> Dict[str, Any]:
        record = self.get_document(doc_id)
        if record.get("status") == STATUS_DELETED:
            raise HTTPException(status_code=400, detail="El documento ya fue eliminado.")

        eliminados = self._delete_vectors(record)
        self.storage.delete(record)

        self.registry.update(
            doc_id,
            status=STATUS_DELETED,
            deleted_by=actor.email or actor.uid,
            deleted_at=utcnow_iso(),
            vectors_deleted=eliminados,
            storage_path="",
        )

        audit.log_action(
            "documento.eliminar", actor.uid, actor.email,
            target=record.get("filename", doc_id),
            details={"doc_id": doc_id, "fragmentos_eliminados": eliminados},
        )
        logger.info(
            f"🗑️ Documento '{record.get('filename')}' eliminado por {actor.email}: "
            f"{eliminados} fragmentos."
        )
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": record.get("filename", ""),
            "fragmentos_eliminados": eliminados,
        }

    # ---------- Lectura del contenido indexado ----------

    def _sorted_vector_ids(self, record: Dict[str, Any]) -> List[str]:
        """IDs del documento ordenados por número de fragmento."""
        ids = self._resolve_vector_ids(record)

        def orden(vid: str) -> int:
            cola = vid.rsplit("::", 1)[-1] if "::" in vid else vid.rsplit("_", 1)[-1]
            try:
                return int(cola)
            except ValueError:
                return 0

        return sorted(ids, key=orden)

    def get_document_content(self, doc_id: str, offset: int = 0, limit: int = 20) -> Dict[str, Any]:
        """
        Devuelve el texto indexado del documento, fragmento por fragmento.

        La fuente es el propio índice, no el archivo original: así el panel
        muestra exactamente lo que el asistente lee cuando responde. Por eso
        también funciona con los documentos heredados, que no tienen original.
        """
        record = self.get_document(doc_id)
        if record.get("status") == STATUS_DELETED:
            raise HTTPException(status_code=400, detail="El documento fue eliminado.")

        limit = max(1, min(limit, 50))
        offset = max(0, offset)

        ids = self._sorted_vector_ids(record)
        total = len(ids)
        seleccion = ids[offset:offset + limit]

        metadatos = self.vector_store.fetch_vector_metadata(seleccion, namespace="")

        fragmentos = []
        for posicion, vid in enumerate(seleccion, start=offset):
            md = metadatos.get(vid, {})
            fragmentos.append({
                "numero": posicion + 1,
                "id": vid,
                "texto": (md.get("text") or "").strip(),
            })

        return {
            "doc_id": doc_id,
            "filename": record.get("filename", ""),
            "title": record.get("title", ""),
            "category": record.get("category", ""),
            "total_fragmentos": total,
            "offset": offset,
            "limit": limit,
            "fragmentos": fragmentos,
        }

    def download_document(self, doc_id: str) -> tuple:
        record = self.get_document(doc_id)
        content = self.storage.load(record)
        if not content:
            raise HTTPException(status_code=404, detail="El archivo original ya no está disponible.")
        return content, record.get("filename", f"{doc_id}.bin")
