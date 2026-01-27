import logging
import asyncio
import numpy as np
from typing import List, Optional, Any
try:
    from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from src.app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = None
        self._initialize()

    def _initialize(self):
        if settings.HUGGINGFACEHUB_API_TOKEN:
            logger.info("☁️ Usando HuggingFace Inference API para embeddings (Zero-RAM)")
            self.client = HuggingFaceInferenceAPIEmbeddings(
                api_key=settings.HUGGINGFACEHUB_API_TOKEN,
                model_name=settings.EMBEDDING_MODEL_NAME
            )
            # Test connection immediately
            self._test_api()
        else:
            logger.warning("⚠️ HUGGINGFACEHUB_API_TOKEN no encontrado. Usando embeddings locales (Alto consumo de RAM)")
            try:
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                self.client = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
            except ImportError:
                logger.error("❌ sentence-transformers no instalado y no hay API Token. El sistema fallará.")

    async def embed_query(self, text: str) -> List[float]:
        """Genera embedding para un texto (async wrapper)."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self.client.embed_query, text)
            return result
        except KeyError as e:
            # Catch specific KeyError: 0 from langchain if API returns dict
            logger.error(f"❌ Error crítico en HuggingFace API. Posible respuesta de error: {e}")
            # Try to debug by making a raw request or catching the internal return?
            # Since we can't easily see the internal return of the library call without patching,
            # we rely on the critical logs we added in __init__ or just fail gracefully.
            raise e
        except Exception as e:
             logger.error(f"Error generando embedding: {e}")
             raise e

    def _test_api(self):
        """Prueba inicial de conexión con HF."""
        if not self.client: return
        try:
            res = self.client.embed_query("test")
            if isinstance(res, dict):
                logger.critical(f"🚨 HuggingFace API retornó un ERROR: {res}")
            else:
                logger.info("✅ HuggingFace API funcionando correctamente.")
        except Exception as e:
            logger.error(f"❌ Error probando HuggingFace API: {e}")


    def embed_query_sync(self, text: str) -> List[float]:
        """Genera embedding síncrono (para uso en inicialización)."""
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)
