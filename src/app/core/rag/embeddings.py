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

class SafeHuggingFaceEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    """Wrapper seguro para HF API que maneja errores de carga y respuestas inesperadas."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return super().embed_documents(texts)
        except KeyError:
            # Captura el error común cuando la API devuelve un dict de error
            logger.error("❌ HuggingFace API devolvió un error (KeyError: 0). Probablemente 'Model Loading' o 'Invalid Token'.")
            return []
        except Exception as e:
            logger.error(f"❌ Error en HuggingFace Embeddings: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        try:
            result = self.embed_documents([text])
            if result and len(result) > 0:
                return result[0]
            return []
        except Exception as e:
             logger.error(f"❌ Error en embed_query: {e}")
             return []

class EmbeddingService:
    def __init__(self):
        self.client = None
        self._initialize()

    def _initialize(self):
        if settings.HUGGINGFACEHUB_API_TOKEN:
            logger.info("☁️ Usando HuggingFace Inference API para embeddings (Zero-RAM)")
            self.client = SafeHuggingFaceEmbeddings(
                api_key=settings.HUGGINGFACEHUB_API_TOKEN,
                model_name=settings.EMBEDDING_MODEL_NAME
            )
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
            if not result:
                logger.warning(f"⚠️ Embedding vacío para texto: {text[:20]}...")
            return result
        except Exception as e:
             logger.error(f"Error generando embedding async: {e}")
             return []

    def _test_api(self):
        """Prueba inicial de conexión con HF."""
        if not self.client: return
        try:
            res = self.client.embed_query("test")
            if not res:
                logger.critical("🚨 HuggingFace API falló en la prueba inicial (retornó vacío/error).")
            else:
                logger.info("✅ HuggingFace API funcionando correctamente.")
        except Exception as e:
            logger.error(f"❌ Error probando HuggingFace API: {e}")

    def embed_query_sync(self, text: str) -> List[float]:
        """Genera embedding síncrono."""
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)
