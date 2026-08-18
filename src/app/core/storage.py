"""
Guarda el archivo original de cada documento indexado.

Sin el original no se puede reindexar ni auditar qué se subió. Se intenta
Firebase Storage; si no está configurado o falla, se usa el disco local
(suficiente en desarrollo, pero recordar que el disco de Render es efímero).
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from src.app.config import settings

logger = logging.getLogger(__name__)

BACKEND_FIREBASE = "firebase"
BACKEND_LOCAL = "local"


class DocumentStorage:
    def __init__(self):
        self.bucket = self._init_bucket()

    def _init_bucket(self):
        if not settings.FIREBASE_STORAGE_BUCKET:
            logger.info("ℹ️ FIREBASE_STORAGE_BUCKET no configurado: los originales se guardan en disco.")
            return None
        try:
            import firebase_admin
            from firebase_admin import storage

            if not firebase_admin._apps:
                return None
            bucket = storage.bucket(settings.FIREBASE_STORAGE_BUCKET)
            logger.info(f"✅ Firebase Storage conectado ({settings.FIREBASE_STORAGE_BUCKET})")
            return bucket
        except Exception as e:
            logger.warning(f"⚠️ Firebase Storage no disponible ({e}). Se usará disco local.")
            return None

    def _local_path(self, doc_id: str, filename: str) -> Path:
        ext = Path(filename).suffix.lower()
        return Path(settings.UPLOAD_DIR) / f"{doc_id}{ext}"

    def save(self, doc_id: str, filename: str, content: bytes) -> Dict[str, str]:
        if self.bucket:
            try:
                blob_path = f"documentos/{doc_id}/{filename}"
                blob = self.bucket.blob(blob_path)
                blob.upload_from_string(content)
                return {"storage_backend": BACKEND_FIREBASE, "storage_path": blob_path}
            except Exception as e:
                logger.error(f"❌ Error subiendo a Firebase Storage, se usa disco local: {e}")

        path = self._local_path(doc_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"storage_backend": BACKEND_LOCAL, "storage_path": str(path)}

    def load(self, record: Dict) -> Optional[bytes]:
        backend = record.get("storage_backend")
        path = record.get("storage_path")
        if not path:
            return None
        if backend == BACKEND_FIREBASE and self.bucket:
            try:
                return self.bucket.blob(path).download_as_bytes()
            except Exception as e:
                logger.error(f"❌ Error descargando de Firebase Storage: {e}")
                return None
        try:
            return Path(path).read_bytes()
        except Exception as e:
            logger.error(f"❌ Error leyendo el original en disco: {e}")
            return None

    def delete(self, record: Dict) -> bool:
        backend = record.get("storage_backend")
        path = record.get("storage_path")
        if not path:
            return False
        try:
            if backend == BACKEND_FIREBASE and self.bucket:
                self.bucket.blob(path).delete()
                return True
            if os.path.exists(path):
                os.remove(path)
                return True
        except Exception as e:
            logger.warning(f"⚠️ No se pudo eliminar el archivo original: {e}")
        return False
