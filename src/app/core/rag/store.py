import logging
import asyncio
import traceback
from typing import List, Any
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
        self._initialize()

    def _initialize(self):
        if settings.PINECONE_API_KEY and settings.PINECONE_ENV:
            logger.info(f"🌲 Conectando a Pinecone: {settings.PINECONE_INDEX_NAME}")
            try:
                pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
                index = pc.Index(settings.PINECONE_INDEX_NAME)
                
                self.vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embedding_service.client,
                    text_key="text"
                )
                logger.info("✅ Pinecone Vector Store conectado")
            except Exception as e:
                logger.error(f"❌ Error conectando Pinecone: {e}")
                self.vectorstore = None
        else:
            logger.warning("⚠️ Credenciales de Pinecone no encontradas.")

    async def search_async(self, query: str, k: int = 4) -> List[Any]:
        """Realiza búsqueda vectorial asíncrona."""
        if not self.vectorstore:
            return []
            
        loop = asyncio.get_running_loop()
        try:
            # search with score
            results = await loop.run_in_executor(
                None, 
                lambda: self.vectorstore.similarity_search_with_score(query, k=k)
            )
            return results
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"❌ Error buscando en Pinecone: {e}\nTraceback COMPLETO:\n{error_details}")
            # Si el error es "0", es muy probable que sea un problema de subproceso o memoria
            if str(e).strip() == "0":
                logger.critical("🚨 ERROR CRÍTICO '0' CONSTANTE: Posible fallo de memoria en EmbeddingService o librería C++ subyacente.")
            return []
