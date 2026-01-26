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
        return await loop.run_in_executor(None, self.client.embed_query, text)

    def embed_query_sync(self, text: str) -> List[float]:
        """Genera embedding síncrono (para uso en inicialización)."""
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)
