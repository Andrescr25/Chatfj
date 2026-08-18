import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

import google.generativeai as genai
import requests
from groq import Groq

logger = logging.getLogger(__name__)

TEMPERATURA = 0.3
MAX_TOKENS = 2000
TOP_P = 0.9

# Señales de que el proveedor falló por saturación o falta de cupo, no porque la
# petición esté mal. Son los casos en los que vale la pena pasar al siguiente.
SEÑALES_TRANSITORIAS = (
    "429", "rate limit", "rate_limit", "quota", "credits", "billing",
    "overloaded", "capacity", "timeout", "timed out", "unavailable",
    "resource_exhausted", "resource exhausted", "too many requests",
    "500", "502", "503", "504",
)


def es_error_transitorio(error: Exception) -> bool:
    """
    ¿Conviene intentar con otro proveedor?

    El caso que motivó esto: Gemini devolvió 429 "prepayment credits are
    depleted" y el chat quedó caído, aunque la llave de Groq funcionaba.
    """
    codigo = getattr(error, "status_code", None) or getattr(error, "code", None)
    if isinstance(codigo, int) and (codigo == 429 or 500 <= codigo < 600):
        return True

    texto = f"{type(error).__name__} {error}".lower()
    return any(señal in texto for señal in SEÑALES_TRANSITORIAS)


class BaseLLM(ABC):
    nombre: str = "desconocido"

    @abstractmethod
    async def generate_async(self, prompt: str, system_message: str = None) -> str:
        pass


class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY no configurada")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.nombre = f"groq:{model}"

    async def generate_async(self, prompt, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURA,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                stream=False,
            )
            return completion.choices[0].message.content.strip()

        return await loop.run_in_executor(None, _run)


class OpenAICompatibleLLM(BaseLLM):
    """
    Cliente para cualquier pasarela con API compatible con OpenAI.

    Lo usan OpenRouter y OmniRoute: ambos exponen POST {base_url}/chat/completions
    con autenticación Bearer, así que no hace falta un cliente por cada uno.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        nombre: str,
        extra_headers: Optional[dict] = None,
        timeout: int = 60,
    ):
        if not api_key:
            raise ValueError(f"Falta la llave de API de {nombre}")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.nombre = f"{nombre}:{model}"
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

    async def generate_async(self, prompt, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.extra_headers,
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": TEMPERATURA,
                    "max_tokens": MAX_TOKENS,
                    "top_p": TOP_P,
                    "stream": False,
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                # Se conserva el código para que la cascada distinga un 429 de
                # un error de petición mal formada.
                error = RuntimeError(
                    f"{self.nombre} respondió {response.status_code}: {response.text[:200]}"
                )
                error.status_code = response.status_code
                raise error

            datos = response.json()
            return datos["choices"][0]["message"]["content"].strip()

        return await loop.run_in_executor(None, _run)


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada")
        genai.configure(api_key=api_key)
        model_name = model if "gemini" in model else "gemini-2.5-flash"
        self.model = genai.GenerativeModel(model_name)
        self.nombre = f"gemini:{model_name}"

    async def generate_async(self, prompt: str, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()

        def _run() -> str:
            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURA,
                    max_output_tokens=MAX_TOKENS,
                    top_p=TOP_P,
                ),
            )
            return response.text.strip()

        return await loop.run_in_executor(None, _run)


class FallbackLLM(BaseLLM):
    """
    Cascada de proveedores: si el primero falla por cupo o saturación, sigue el
    siguiente sin que la persona usuaria se entere.

    Un proveedor que responde mal por su propia culpa (petición inválida) también
    cede el turno, porque desde el punto de vista del usuario da igual el motivo:
    lo que importa es obtener respuesta.
    """

    def __init__(self, proveedores: list):
        if not proveedores:
            raise ValueError("La cascada de modelos está vacía")
        self.proveedores = proveedores
        self.nombre = " → ".join(p.nombre for p in proveedores)
        self.ultimo_usado = ""

    async def generate_async(self, prompt: str, system_message: str = None) -> str:
        ultimo_error: Optional[Exception] = None

        for posicion, proveedor in enumerate(self.proveedores, start=1):
            try:
                respuesta = await proveedor.generate_async(prompt, system_message)
                if not respuesta:
                    raise RuntimeError("respuesta vacía")

                self.ultimo_usado = proveedor.nombre
                if posicion > 1:
                    logger.warning(
                        f"↪️ Respondió {proveedor.nombre} tras fallar "
                        f"{posicion - 1} proveedor(es) antes"
                    )
                return respuesta

            except Exception as e:
                ultimo_error = e
                motivo = "sin cupo o saturado" if es_error_transitorio(e) else "error"
                quedan = len(self.proveedores) - posicion
                logger.error(
                    f"❌ {proveedor.nombre} falló ({motivo}): {str(e)[:180]}"
                    + (f" — se intenta con el siguiente ({quedan} por probar)" if quedan else "")
                )

        logger.critical("🚨 Todos los proveedores de la cascada fallaron.")
        raise ultimo_error
