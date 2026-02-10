import logging
import asyncio
import traceback
from typing import List, Any, Optional, Dict
from datetime import datetime
try:
    from pinecone import Pinecone as PineconeClient
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeClient = None
    PineconeVectorStore = None


from src.app.config import settings
from src.app.core.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vectorstore = None
        self.pinecone_index = None  # Direct Pinecone index for namespace operations
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
        """Realiza búsqueda vectorial asíncrona en docs (namespace default)."""
        if not self.vectorstore:
            return []
            
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, 
                lambda: self.vectorstore.similarity_search_with_score(query, k=k)
            )
            return results
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"❌ Error buscando en Pinecone: {e}\nTraceback COMPLETO:\n{error_details}")
            if str(e).strip() == "0":
                logger.critical("🚨 ERROR CRÍTICO '0' CONSTANTE: Posible fallo de memoria en EmbeddingService o librería C++ subyacente.")
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
