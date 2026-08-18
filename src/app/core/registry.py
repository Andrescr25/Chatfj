"""
Catálogo de documentos indexados.

Pinecone guarda vectores, no documentos: no hay forma de preguntarle "¿qué
archivos tengo?". Este catálogo es la pieza que permite listar, reindexar y
eliminar documentos desde el panel de administración.

Fuente primaria: colección 'documents' de Firestore.
Respaldo local (desarrollo o Firestore no disponible): data/documents_registry.json
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COLLECTION = "documents"

STATUS_PENDING = "pendiente"
STATUS_INDEXING = "indexando"
STATUS_INDEXED = "indexado"
STATUS_ERROR = "error"
STATUS_DELETED = "eliminado"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    def __init__(self, json_path: str = "data/documents_registry.json"):
        self.json_path = json_path
        self._lock = threading.Lock()
        self._db = self._init_firestore()

    def _init_firestore(self):
        try:
            import firebase_admin
            from firebase_admin import firestore

            if firebase_admin._apps:
                db = firestore.client()
                logger.info("✅ Catálogo de documentos usando Firestore")
                return db
            logger.warning("⚠️ Firebase no inicializado: catálogo de documentos en archivo local.")
        except Exception as e:
            logger.warning(f"⚠️ Firestore no disponible para el catálogo ({e}). Usando archivo local.")
        return None

    @property
    def backend(self) -> str:
        return "firestore" if self._db else "local"

    # ---------- Respaldo en archivo ----------

    def _read_json(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.json_path) or os.path.getsize(self.json_path) == 0:
            return {}
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ No se pudo leer el catálogo local: {e}")
            return {}

    def _write_json(self, data: Dict[str, Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
        tmp = f"{self.json_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.json_path)

    # ---------- API pública ----------

    def save(self, record: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = record["doc_id"]
        if self._db:
            self._db.collection(COLLECTION).document(doc_id).set(record)
            return record
        with self._lock:
            data = self._read_json()
            data[doc_id] = record
            self._write_json(data)
        return record

    def update(self, doc_id: str, **fields) -> Optional[Dict[str, Any]]:
        fields["updated_at"] = utcnow_iso()
        if self._db:
            try:
                self._db.collection(COLLECTION).document(doc_id).update(fields)
            except Exception as e:
                logger.error(f"❌ Error actualizando catálogo ({doc_id}): {e}")
                return None
            return self.get(doc_id)
        with self._lock:
            data = self._read_json()
            if doc_id not in data:
                return None
            data[doc_id].update(fields)
            self._write_json(data)
            return data[doc_id]

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if self._db:
            snap = self._db.collection(COLLECTION).document(doc_id).get()
            return snap.to_dict() if snap.exists else None
        return self._read_json().get(doc_id)

    def list(self, include_deleted: bool = False) -> List[Dict[str, Any]]:
        if self._db:
            records = [d.to_dict() for d in self._db.collection(COLLECTION).stream()]
        else:
            records = list(self._read_json().values())

        if not include_deleted:
            records = [r for r in records if r.get("status") != STATUS_DELETED]
        records.sort(key=lambda r: r.get("uploaded_at") or "", reverse=True)
        return records

    def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Detecta duplicados: mismo contenido ya indexado."""
        for record in self.list():
            if record.get("file_hash") and record["file_hash"] == file_hash:
                return record
        return None

    def prefix_exists(self, vector_prefix: str) -> bool:
        return any(r.get("vector_prefix") == vector_prefix for r in self.list(include_deleted=True))
