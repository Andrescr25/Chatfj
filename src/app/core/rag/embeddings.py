import asyncio
import logging
import time
from typing import List

import requests

try:
    from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from src.app.config import settings

logger = logging.getLogger(__name__)


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
                if vector:
                    return vector  # Ensure vector is not empty
            
            # Si llegamos aquí, falló.
            # LANZAR ERROR para que store.py lo capture y no llame a Pinecone con basura
            raise ValueError(f"No se pudo generar embedding para: {text[:15]}...")
            
        except Exception as e:
             logger.error(f"❌ Error en embed_query: {e}")
             raise e # Re-raise to let store.py handle it gracefully

# Cada modelo de embeddings vive en su propio espacio del índice. Comparar
# vectores de modelos distintos no falla: devuelve resultados sin sentido, en
# silencio. Por eso el espacio se decide por el modelo que generó el vector.
ESPACIO_E5 = ""
ESPACIO_GEMINI = "gemini"


class ProveedorConNombre:
    """
    Envoltura para identificar a cada proveedor en los registros.

    El cliente de HuggingFace es un modelo de Pydantic y no acepta atributos
    nuevos, así que el nombre y el espacio del índice viven aquí.
    """

    def __init__(self, cliente, nombre: str, espacio: str):
        self._cliente = cliente
        self.nombre = nombre
        self.espacio = espacio

    def embed_query(self, text: str) -> List[float]:
        return self._cliente.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._cliente.embed_documents(texts)


class EmbeddingsGemini:
    """
    Embeddings de Google Gemini, ajustados a 1024 dimensiones para que quepan
    en el mismo índice que los de e5.
    """

    def __init__(self, api_key: str, model: str, dimensiones: int = 1024):
        if not api_key:
            raise ValueError("Falta la llave de API de Gemini")
        self.api_key = api_key
        self.model = model
        self.dimensiones = dimensiones
        self.espacio = ESPACIO_GEMINI
        self.nombre = f"gemini:{model}"
        self._base = "https://generativelanguage.googleapis.com/v1beta"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        peticiones = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.dimensiones,
            }
            for t in texts
        ]
        r = requests.post(
            f"{self._base}/models/{self.model}:batchEmbedContents",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"requests": peticiones},
            timeout=90,
        )
        if r.status_code != 200:
            error = RuntimeError(f"{self.nombre} respondió {r.status_code}: {r.text[:160]}")
            error.status_code = r.status_code
            raise error
        return [e["values"] for e in r.json().get("embeddings", [])]

    def embed_query(self, text: str) -> List[float]:
        vectores = self.embed_documents([text])
        if not vectores or not vectores[0]:
            raise ValueError(f"{self.nombre} devolvió un embedding vacío")
        return vectores[0]


class EmbeddingsCompatiblesOpenAI:
    """
    Embeddings desde cualquier API compatible con OpenAI (DeepInfra, Nebius...).

    Sirve de respaldo cuando HuggingFace se queda sin crédito. Es indispensable
    que el modelo sea el mismo con el que se construyó el índice: vectores de
    otro modelo no son comparables y las búsquedas devolverían cualquier cosa.
    """

    def __init__(self, api_key: str, model: str, base_url: str, nombre: str):
        if not api_key:
            raise ValueError(f"Falta la llave de API de {nombre}")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.nombre = nombre
        self.espacio = ESPACIO_E5

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        r = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        if r.status_code != 200:
            error = RuntimeError(f"{self.nombre} respondió {r.status_code}: {r.text[:160]}")
            error.status_code = r.status_code
            raise error

        datos = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in datos]

    def embed_query(self, text: str) -> List[float]:
        vectores = self.embed_documents([text])
        if not vectores or not vectores[0]:
            raise ValueError(f"{self.nombre} devolvió un embedding vacío")
        return vectores[0]


class EmbeddingService:
    """
    Embeddings con respaldo.

    Los embeddings son el punto más crítico del sistema: sin ellos no se puede
    indexar ni buscar, y el asistente responde sin documentos, en silencio. Por
    eso admiten cascada igual que los modelos de lenguaje.
    """

    def __init__(self):
        self.client = None
        self.clientes_hf = []
        self.respaldo = None
        self.gemini = None
        self._initialize()
        self._inicializar_respaldo()

    def _inicializar_respaldo(self):
        """
        Dos respaldos de distinta naturaleza:

        - EMBEDDINGS_FALLBACK: sirve el MISMO modelo (e5) y escribe en el mismo
          espacio del índice. Es un respaldo del proveedor.
        - Gemini: es OTRO modelo, así que vive en su propio espacio. Es un
          respaldo del modelo, y por eso los documentos se indexan en ambos.
        """
        if settings.EMBEDDINGS_FALLBACK_API_KEY:
            try:
                self.respaldo = EmbeddingsCompatiblesOpenAI(
                    api_key=settings.EMBEDDINGS_FALLBACK_API_KEY,
                    model=settings.EMBEDDING_MODEL_NAME,
                    base_url=settings.EMBEDDINGS_FALLBACK_BASE_URL,
                    nombre="respaldo de embeddings",
                )
                logger.info(f"🧭 Respaldo del mismo modelo: {settings.EMBEDDINGS_FALLBACK_BASE_URL}")
            except Exception as e:
                logger.warning(f"⚠️ Respaldo de embeddings no disponible: {e}")

        if settings.GEMINI_API_KEY and settings.EMBEDDINGS_GEMINI_ENABLED:
            try:
                self.gemini = EmbeddingsGemini(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.EMBEDDINGS_GEMINI_MODEL,
                )
                logger.info(f"🧭 Segundo modelo de embeddings: {self.gemini.nombre} (espacio '{ESPACIO_GEMINI}')")
            except Exception as e:
                logger.warning(f"⚠️ Embeddings de Gemini no disponibles: {e}")

    @property
    def proveedores_e5(self) -> list:
        """
        Todos los que producen vectores del mismo modelo, en orden de uso:
        las llaves de HuggingFace y, al final, el respaldo por otra pasarela.
        """
        proveedores = list(getattr(self, "clientes_hf", None) or ([self.client] if self.client else []))
        if self.respaldo is not None:
            proveedores.append(self.respaldo)
        return proveedores

    def _intentar(self, proveedores, operacion, *args):
        ultimo_error = None
        for proveedor in proveedores:
            try:
                resultado = getattr(proveedor, operacion)(*args)
                if resultado:
                    return resultado
                raise RuntimeError("devolvió vacío")
            except Exception as e:
                ultimo_error = e
                logger.warning(
                    f"↪️ Embeddings: {getattr(proveedor, 'nombre', 'principal')} falló "
                    f"({str(e)[:80]})"
                )
        raise ultimo_error or RuntimeError("no hay proveedores de embeddings")

    def _con_respaldo(self, operacion, *args):
        """Cascada dentro del mismo modelo: el vector siempre es comparable con e5."""
        return self._intentar(self.proveedores_e5, operacion, *args)

    def embed_query_con_espacio(self, text: str) -> tuple:
        """
        Devuelve (vector, espacio del índice donde buscar).

        El espacio lo decide el modelo que generó el vector: buscar en el
        espacio equivocado no da error, da resultados sin sentido.
        """
        try:
            return self._con_respaldo("embed_query", text), ESPACIO_E5
        except Exception as e:
            if not self.gemini:
                raise
            logger.warning(
                f"↪️ Embeddings: ningún proveedor de e5 respondió ({str(e)[:70]}). "
                f"Se consulta el espacio '{ESPACIO_GEMINI}'."
            )
            return self.gemini.embed_query(text), ESPACIO_GEMINI

    def embed_documents_por_espacio(self, texts: List[str]) -> dict:
        """
        Vectores para cada espacio del índice, de cara a la indexación.

        Se indexa en los dos para que exista respaldo real: si mañana falla un
        modelo, el otro ya tiene los mismos documentos disponibles.
        """
        resultado = {}
        try:
            resultado[ESPACIO_E5] = self._con_respaldo("embed_documents", texts)
        except Exception as e:
            logger.error(f"❌ Sin vectores de e5 para este lote: {str(e)[:110]}")

        if self.gemini:
            try:
                resultado[ESPACIO_GEMINI] = self.gemini.embed_documents(texts)
            except Exception as e:
                logger.warning(f"⚠️ Sin vectores de Gemini para este lote: {str(e)[:110]}")

        if not resultado:
            raise RuntimeError("Ningún proveedor de embeddings respondió")
        return resultado

    def _initialize(self):
        tokens = settings.huggingface_tokens
        if tokens:
            # Una cliente por llave: el crédito de HuggingFace es por cuenta, así
            # que varias llaves dan varios cupos del MISMO modelo. Sus vectores
            # son comparables entre sí, por eso comparten espacio del índice.
            self.clientes_hf = []
            primero = None
            for i, token in enumerate(tokens):
                enmascarada = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
                bruto = SafeHuggingFaceEmbeddings(
                    api_key=token,
                    model_name=settings.EMBEDDING_MODEL_NAME,
                )
                primero = primero or bruto
                self.clientes_hf.append(
                    ProveedorConNombre(bruto, f"huggingface[{i + 1}]", ESPACIO_E5)
                )
                logger.info(f"☁️ Embeddings por HuggingFace, llave {i + 1} ({enmascarada})")

            # self.client queda como el cliente sin envolver: LangChain lo exige
            self.client = primero
            if len(self.clientes_hf) > 1:
                logger.info(f"🧭 {len(self.clientes_hf)} llaves de HuggingFace en cascada")
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
            result = await loop.run_in_executor(None, self.embed_query_sync, text)
            if not result:
                logger.warning(f"⚠️ Embedding vacío para texto: {text[:20]}...")
            return result
        except Exception as e:
             logger.error(f"Error generando embedding async: {e}")
             return []

    def _test_api(self):
        """Prueba inicial de conexión con HF."""
        if not self.client:
            return
        try:
            res = self.client.embed_query("test")
            if not res:
                logger.critical("🚨 HuggingFace API falló en la prueba inicial (retornó vacío/error).")
            else:
                logger.info("✅ HuggingFace API funcionando correctamente.")
        except Exception as e:
            logger.error(f"❌ Error probando HuggingFace API: {e}")

    def embed_query_sync(self, text: str) -> List[float]:
        """Genera embedding síncrono, con respaldo si el principal falla."""
        return self._con_respaldo("embed_query", text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._con_respaldo("embed_documents", texts)
