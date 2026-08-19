"""
Gestión de las correcciones aprendidas.

Las correcciones viven en Pinecone, en el namespace 'corrections', y tienen
prioridad sobre los documentos oficiales al responder. Hasta ahora solo se
podían crear: no había forma de revisarlas, arreglar una mal escrita ni quitar
una equivocada sin entrar por línea de comandos.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import HTTPException

logger = logging.getLogger(__name__)

NAMESPACE = "corrections"


class CorrectionsService:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def _indice(self):
        indice = getattr(self.vector_store, "pinecone_index", None)
        if not indice:
            raise HTTPException(status_code=503, detail="La base vectorial no está disponible.")
        return indice

    def listar(self, limite: int = 500) -> Dict[str, Any]:
        """Devuelve todas las correcciones con su metadata, de la más nueva a la más vieja."""
        self._indice()  # falla temprano y con mensaje claro si Pinecone no responde

        ids = self.vector_store.list_vector_ids("", namespace=NAMESPACE)
        if ids is None:
            raise HTTPException(
                status_code=503,
                detail="El índice no permite listar las correcciones.",
            )

        correcciones = []
        for i in range(0, min(len(ids), limite), 100):
            lote = ids[i:i + 100]
            metadatos = self.vector_store.fetch_vector_metadata(lote, namespace=NAMESPACE)
            for vid, md in metadatos.items():
                correcciones.append({
                    "id": vid,
                    "pregunta": md.get("original_question", ""),
                    "correccion": md.get("text", ""),
                    "entrenador": md.get("trainer", ""),
                    "intencion": md.get("intent", ""),
                    "fecha": md.get("timestamp", ""),
                    "editada_por": md.get("edited_by", ""),
                    "editada_en": md.get("edited_at", ""),
                })

        correcciones.sort(key=lambda c: c.get("fecha", ""), reverse=True)
        return {"correcciones": correcciones, "total": len(ids)}

    def obtener(self, correccion_id: str) -> Dict[str, Any]:
        metadatos = self.vector_store.fetch_vector_metadata([correccion_id], namespace=NAMESPACE)
        md = metadatos.get(correccion_id)
        if not md:
            raise HTTPException(status_code=404, detail="La corrección no existe.")
        return {
            "id": correccion_id,
            "pregunta": md.get("original_question", ""),
            "correccion": md.get("text", ""),
            "entrenador": md.get("trainer", ""),
            "intencion": md.get("intent", ""),
            "fecha": md.get("timestamp", ""),
        }

    def actualizar(
        self,
        correccion_id: str,
        actor,
        pregunta: str = None,
        correccion: str = None,
    ) -> Dict[str, Any]:
        """
        Edita una corrección conservando su identificador.

        Si cambia la pregunta hay que recalcular el vector: es la pregunta la
        que decide cuándo se aplica esta corrección, no el texto de la respuesta.
        """
        actual = self.obtener(correccion_id)
        nueva_pregunta = (pregunta if pregunta is not None else actual["pregunta"]).strip()
        nuevo_texto = (correccion if correccion is not None else actual["correccion"]).strip()

        if not nueva_pregunta or not nuevo_texto:
            raise HTTPException(
                status_code=400,
                detail="La pregunta y el texto de la corrección no pueden quedar vacíos.",
            )

        indice = self._indice()
        metadata = {
            "text": nuevo_texto,
            "original_question": nueva_pregunta,
            "intent": actual.get("intencion") or "correction",
            "trainer": actual.get("entrenador") or "anon",
            "timestamp": actual.get("fecha") or datetime.now(timezone.utc).isoformat(),
            "type": "correction",
            "edited_by": actor.email or actor.uid,
            "edited_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            vector = self.vector_store.embedding_service.embed_query_sync(nueva_pregunta)
            if not vector:
                raise RuntimeError("no se pudo generar el vector de la pregunta")
            indice.upsert(vectors=[(correccion_id, vector, metadata)], namespace=NAMESPACE)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Error actualizando la corrección {correccion_id}: {e}")
            raise HTTPException(status_code=502, detail=f"No se pudo guardar la corrección: {e}")

        from src.app.core import audit

        audit.log_action(
            "correccion.editar", actor.uid, actor.email, target=correccion_id,
            details={"pregunta": nueva_pregunta[:80]},
        )
        return self.obtener(correccion_id)

    def eliminar(self, correccion_id: str, actor) -> Dict[str, Any]:
        actual = self.obtener(correccion_id)
        indice = self._indice()
        try:
            indice.delete(ids=[correccion_id], namespace=NAMESPACE)
        except Exception as e:
            logger.error(f"❌ Error eliminando la corrección {correccion_id}: {e}")
            raise HTTPException(status_code=502, detail=f"No se pudo eliminar: {e}")

        from src.app.core import audit

        audit.log_action(
            "correccion.eliminar", actor.uid, actor.email, target=correccion_id,
            details={"pregunta": actual["pregunta"][:80]},
        )
        return {"status": "success", "id": correccion_id, "pregunta": actual["pregunta"]}
