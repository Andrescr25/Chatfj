import asyncio
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone as PineconeClient
except ImportError:
    PineconeClient = None
    PineconeVectorStore = None


from src.app.config import settings
from src.app.core.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class Fragmento:
    """
    Fragmento recuperado del índice.

    Reemplaza al Document de LangChain en la búsqueda: expone lo único que el
    servicio de chat necesita y evita depender del envoltorio para elegir el
    espacio del índice.
    """

    __slots__ = ("page_content", "metadata")

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vectorstore = None
        self.pinecone_index = None  # Direct Pinecone index for namespace operations
        self.ultimo_espacio = ""
        self._initialize()

    def _initialize(self):
        if settings.PINECONE_API_KEY and settings.PINECONE_ENV:
            logger.info(f"🌲 Conectando a Pinecone: {settings.PINECONE_INDEX_NAME}")
            try:
                pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
                self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
                
                # LangChain wrapper for doc search (default namespace)
                self.vectorstore = PineconeVectorStore(
                    index=self.pinecone_index,
                    embedding=self.embedding_service.client,
                    text_key="text"
                )
                logger.info("✅ Pinecone Vector Store conectado")
            except Exception as e:
                logger.error(f"❌ Error conectando Pinecone: {e}")
                self.vectorstore = None
                self.pinecone_index = None
        else:
            logger.warning("⚠️ Credenciales de Pinecone no encontradas.")

    async def search_async(self, query: str, k: int = 4) -> List[Any]:
        """
        Busca en el espacio que corresponde al modelo que generó el vector.

        Los documentos están indexados en dos espacios, uno por modelo de
        embeddings. Consultar el espacio equivocado no da error: da resultados
        sin sentido. Por eso el espacio lo decide quien genera el vector, no
        esta función.
        """
        if not self.pinecone_index:
            return []

        loop = asyncio.get_running_loop()

        def _buscar():
            vector, espacio = self.embedding_service.embed_query_con_espacio(query)
            if not vector:
                return []

            respuesta = self.pinecone_index.query(
                vector=vector,
                top_k=k,
                namespace=espacio,
                include_metadata=True,
            )
            encontrados = []
            for match in respuesta.get("matches", []):
                metadata = dict(match.get("metadata") or {})
                texto = metadata.pop("text", "")
                encontrados.append((Fragmento(texto, metadata), match.get("score", 0.0)))

            self.ultimo_espacio = espacio
            if espacio:
                logger.info(f"🔎 Búsqueda resuelta en el espacio '{espacio}' ({len(encontrados)} resultados)")
            return encontrados

        try:
            return await loop.run_in_executor(None, _buscar)
        except Exception as e:
            logger.error(f"❌ Error buscando en Pinecone: {e}\n{traceback.format_exc()}")
            return []

    # === CORRECTIONS NAMESPACE METHODS ===

    def upsert_correction(self, correction_id: str, question: str, correction_text: str, 
                          intent: str = "correction", trainer: str = "anon") -> bool:
        """Upsert a correction into the 'corrections' namespace."""
        if not self.pinecone_index:
            logger.error("❌ Pinecone index no disponible para guardar corrección.")
            return False
        
        try:
            vector = self.embedding_service.embed_query_sync(question)
            if not vector:
                logger.error("❌ No se pudo generar embedding para la corrección.")
                return False
            
            metadata = {
                "text": correction_text,
                "original_question": question,
                "intent": intent,
                "trainer": trainer,
                "timestamp": datetime.now().isoformat(),
                "type": "correction"
            }
            
            self.pinecone_index.upsert(
                vectors=[(correction_id, vector, metadata)],
                namespace="corrections"
            )
            logger.info(f"✅ Corrección guardada en Pinecone (namespace=corrections): {correction_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando corrección en Pinecone: {e}")
            return False

    async def search_corrections_async(self, query: str, k: int = 3, threshold: float = 0.80) -> List[Dict]:
        """Search for relevant corrections in the 'corrections' namespace."""
        if not self.pinecone_index:
            return []
        
        loop = asyncio.get_running_loop()
        
        def _search():
            try:
                vector = self.embedding_service.embed_query_sync(query)
                if not vector:
                    return []
                
                results = self.pinecone_index.query(
                    vector=vector,
                    top_k=k,
                    namespace="corrections",
                    include_metadata=True
                )
                
                corrections = []
                for match in results.get("matches", []):
                    score = match.get("score", 0)
                    if score >= threshold:
                        metadata = match.get("metadata", {})
                        corrections.append({
                            "id": match.get("id"),
                            "score": score,
                            "correction": metadata.get("text", ""),
                            "original_question": metadata.get("original_question", ""),
                            "trainer": metadata.get("trainer", "anon"),
                            "intent": metadata.get("intent", "correction"),
                            "timestamp": metadata.get("timestamp", "")
                        })
                
                return corrections
            except Exception as e:
                logger.error(f"❌ Error buscando correcciones en Pinecone: {e}")
                return []
        
        return await loop.run_in_executor(None, _search)

    # === DOCUMENTS NAMESPACE METHODS (gestión desde el panel) ===

    def upsert_vectors(self, vectors: List[Any], namespace: str = "") -> bool:
        """Sube un lote de vectores (id, values, metadata) al namespace indicado."""
        if not self.pinecone_index:
            logger.error("❌ Pinecone index no disponible para subir vectores.")
            return False
        try:
            self.pinecone_index.upsert(vectors=vectors, namespace=namespace)
            return True
        except Exception as e:
            logger.error(f"❌ Error subiendo vectores a Pinecone: {e}")
            return False

    def list_vector_ids(self, prefix: str, namespace: str = "") -> Optional[List[str]]:
        """
        Lista los IDs de vectores que empiezan con un prefijo.

        Devuelve None si el índice no soporta el listado por prefijo (índices
        pod-based); en ese caso el llamador debe reconstruir los IDs a partir
        del catálogo.
        """
        if not self.pinecone_index:
            return None
        try:
            ids: List[str] = []
            for page in self.pinecone_index.list(prefix=prefix, namespace=namespace):
                if isinstance(page, str):
                    ids.append(page)
                else:
                    ids.extend(page)
            return ids
        except Exception as e:
            logger.warning(f"⚠️ El índice no permite listar por prefijo ('{prefix}'): {e}")
            return None

    def delete_vectors(self, ids: List[str], namespace: str = "", batch_size: int = 500) -> int:
        """Elimina vectores por ID en lotes. Devuelve cuántos se solicitaron eliminar."""
        if not self.pinecone_index or not ids:
            return 0
        deleted = 0
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            try:
                self.pinecone_index.delete(ids=batch, namespace=namespace)
                deleted += len(batch)
            except Exception as e:
                logger.error(f"❌ Error eliminando vectores de Pinecone: {e}")
                raise
        return deleted

    def fetch_vector_metadata(self, ids: List[str], namespace: str = "") -> Dict[str, Dict]:
        """Trae la metadata de vectores concretos (para vista previa e inventario)."""
        if not self.pinecone_index or not ids:
            return {}
        try:
            result = self.pinecone_index.fetch(ids=ids, namespace=namespace)
            vectors = getattr(result, "vectors", None)
            if vectors is None and isinstance(result, dict):
                vectors = result.get("vectors", {})
            return {
                vid: (getattr(v, "metadata", None) or (v.get("metadata") if isinstance(v, dict) else {}) or {})
                for vid, v in (vectors or {}).items()
            }
        except Exception as e:
            logger.error(f"❌ Error consultando metadata en Pinecone: {e}")
            return {}

    def index_stats(self) -> Dict[str, Any]:
        """Estadísticas del índice (total de vectores por namespace)."""
        if not self.pinecone_index:
            return {}
        try:
            stats = self.pinecone_index.describe_index_stats()
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as e:
            logger.error(f"❌ Error consultando estadísticas de Pinecone: {e}")
            return {}

    def fetch_vectors_full(self, ids: List[str], namespace: str = "") -> List[tuple]:
        """
        Trae vectores completos (valores + metadata).

        Necesario para renombrar un documento: Pinecone no permite actualizar
        metadata en lote, pero sí volver a subir los vectores. Traer de a 100 y
        resubir de a 100 convierte 1.000 llamadas en 20.
        """
        if not self.pinecone_index or not ids:
            return []
        try:
            resultado = self.pinecone_index.fetch(ids=ids, namespace=namespace)
            vectores = getattr(resultado, "vectors", None)
            if vectores is None and isinstance(resultado, dict):
                vectores = resultado.get("vectors", {})

            completos = []
            for vid, v in (vectores or {}).items():
                valores = getattr(v, "values", None)
                if valores is None and isinstance(v, dict):
                    valores = v.get("values")
                metadata = getattr(v, "metadata", None)
                if metadata is None and isinstance(v, dict):
                    metadata = v.get("metadata")
                if valores:
                    completos.append((vid, list(valores), dict(metadata or {})))
            return completos
        except Exception as e:
            logger.error(f"❌ Error trayendo vectores completos: {e}")
            raise

    @property
    def umbral_actual(self) -> float:
        """Umbral de relevancia del espacio en el que se resolvió la última búsqueda."""
        from src.app.config import settings
        from src.app.core.rag.embeddings import ESPACIO_GEMINI

        if self.ultimo_espacio == ESPACIO_GEMINI:
            return settings.SEARCH_THRESHOLD_GEMINI
        return settings.SEARCH_THRESHOLD_E5
