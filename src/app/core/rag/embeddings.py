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

import time
import requests

class SafeHuggingFaceEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    """
    Cliente robusto para HF Inference API usando requests directo.
    Maneja el estado 'Model Loading' y errores 503 automáticamente.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # URL oficial de la Inference API (New Router endpoint - 2025)
        # El endpoint viejo api-inference.huggingface.co fue deprecado (410 Gone)
        api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        # Extract raw string from SecretStr (Pydantic v2 auto-converts api_key)
        raw_key = self.api_key.get_secret_value() if hasattr(self.api_key, 'get_secret_value') else str(self.api_key)
        headers = {"Authorization": f"Bearer {raw_key}"}
        
        # Payload con opción wait_for_model
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }

        retries = 3
        for attempt in range(retries):
            try:
                # Debug logging
                masked_key = f"{raw_key[:4]}...{raw_key[-4:]}" if raw_key and len(raw_key) > 8 else "NO_KEY"
                logger.info(f"🔗 Requesting: {api_url} (Key: {masked_key})")
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=20)
                
                # Check 503 (Loading) explicitly even if wait_for_model is True
                if response.status_code == 503:
                    estimated_time = response.json().get("estimated_time", 5.0)
                    logger.warning(f"⏳ Modelo cargando... Esperando {estimated_time}s (Intento {attempt+1}/{retries})")
                    time.sleep(estimated_time + 1)
                    continue
                
                if response.status_code != 200:
                    logger.error(f"❌ Error API HF ({response.status_code}): {response.text[:500]}") # Log first 500 chars
                    logger.warning(f"🔍 Headers de respuesta: {response.headers}")
                    return []

                result = response.json()
                
                # Validación de formato (debe ser lista de listas)
                if isinstance(result, list) and len(result) > 0:
                     # A veces devuelve [ [[...]] ] (nested) o directamente [[...]]
                    if isinstance(result[0], list):
                        if isinstance(result[0][0], list): # Extra nest: [ [[...]] ]
                             logger.warning(f"⚠️ Estructura anidada extra detectada: tipo {type(result[0][0])}")
                             return result[0]
                        return result
                    # Si es lista plana (un solo doc), encapsular
                    if isinstance(result[0], float):
                        return [result]
                    
                logger.error(f"❌ Formato inesperado de API. Tipo de respuesta: {type(result)}")
                logger.error(f"🔍 Contenido crudo (truncado): {str(result)[:500]}")
                return []
                
            except Exception as e:
                logger.error(f"❌ Error de conexión HF: {e}")
                time.sleep(2)
        
        logger.error("❌ Fallaron todos los reintentos con HuggingFace API.")
        return []

    def embed_query(self, text: str) -> List[float]:
        try:
            result = self.embed_documents([text])
            if result and len(result) > 0:
                vector = result[0]
                if vector: return vector # Ensure vector is not empty
            
            # Si llegamos aquí, falló.
            # LANZAR ERROR para que store.py lo capture y no llame a Pinecone con basura
            raise ValueError(f"No se pudo generar embedding para: {text[:15]}...")
            
        except Exception as e:
             logger.error(f"❌ Error en embed_query: {e}")
             raise e # Re-raise to let store.py handle it gracefully

class EmbeddingService:
    def __init__(self):
        self.client = None
        self._initialize()

    def _initialize(self):
        if settings.HUGGINGFACEHUB_API_TOKEN:
            token = settings.HUGGINGFACEHUB_API_TOKEN
            masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
            logger.info(f"☁️ Usando HuggingFace Inference API para embeddings (Token: {masked})")
            self.client = SafeHuggingFaceEmbeddings(
                api_key=token,
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
