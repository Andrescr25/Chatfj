"""
Construcción de la cascada de modelos de lenguaje.

El sistema tenía un solo proveedor fijo: cuando Gemini se quedó sin créditos, el
chat entero dejó de responder aunque las otras llaves funcionaban. Ahora se
declara un orden de intento y el primero que responda gana.
"""
import logging

from src.app.config import settings
from src.app.core.llm.client import (
    BaseLLM,
    FallbackLLM,
    GeminiLLM,
    GroqLLM,
    OpenAICompatibleLLM,
)

logger = logging.getLogger(__name__)


def _partir(entrada: str) -> tuple:
    """'huggingface:zai-org/GLM-5.2' -> ('huggingface', 'zai-org/GLM-5.2')"""
    proveedor, _, modelo = entrada.partition(":")
    return proveedor.strip().lower(), (modelo.strip() or None)


def _crear(entrada: str) -> BaseLLM:
    nombre, modelo = _partir(entrada)
    if nombre == "groq":
        return GroqLLM(api_key=settings.GROQ_API_KEY, model=modelo or settings.GROQ_MODEL)

    if nombre == "gemini":
        return GeminiLLM(api_key=settings.GEMINI_API_KEY, model=modelo or settings.GEMINI_MODEL)

    if nombre == "huggingface":
        return OpenAICompatibleLLM(
            api_key=settings.HUGGINGFACEHUB_API_TOKEN,
            model=modelo or settings.HUGGINGFACE_CHAT_MODEL,
            base_url=settings.HUGGINGFACE_CHAT_BASE_URL,
            nombre="huggingface",
            timeout=90,  # su router puede tardar en despertar el modelo
        )

    if nombre == "cerebras":
        return OpenAICompatibleLLM(
            api_key=settings.CEREBRAS_API_KEY,
            model=modelo or settings.CEREBRAS_MODEL,
            base_url=settings.CEREBRAS_BASE_URL,
            nombre="cerebras",
        )

    if nombre == "omniroute":
        return OpenAICompatibleLLM(
            api_key=settings.OMNIROUTE_API_KEY,
            model=modelo or settings.OMNIROUTE_MODEL,
            base_url=settings.OMNIROUTE_BASE_URL,
            nombre="omniroute",
        )

    if nombre == "openrouter":
        return OpenAICompatibleLLM(
            api_key=settings.OPENROUTER_API_KEY,
            model=modelo or settings.OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            nombre="openrouter",
            extra_headers={
                "HTTP-Referer": "https://chatfj-26458.web.app",
                "X-Title": "Chat FJ - Facilitadores Judiciales",
            },
        )

    raise ValueError(f"Proveedor desconocido: '{nombre}'")


def construir_cascada() -> FallbackLLM:
    """
    Arma la cascada según LLM_CHAIN (o LLM_PROVIDER si aquella está vacía).

    Un proveedor sin llave configurada se omite con un aviso, en vez de tumbar
    el arranque: el objetivo es que el sistema responda con lo que tenga.
    """
    proveedores = []
    for entrada in settings.llm_chain:
        try:
            proveedores.append(_crear(entrada))
        except Exception as e:
            logger.warning(f"⚠️ Proveedor '{entrada}' no disponible, se omite de la cascada: {e}")

    if not proveedores:
        raise RuntimeError(
            "Ningún proveedor de IA quedó configurado. Revise LLM_CHAIN y las llaves de API."
        )

    cascada = FallbackLLM(proveedores)
    if len(proveedores) == 1:
        logger.info(f"🤖 Modelo: {cascada.nombre} (sin respaldo configurado)")
    else:
        logger.info(f"🤖 Cascada de modelos: {cascada.nombre}")
    return cascada
