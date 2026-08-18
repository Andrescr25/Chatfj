import asyncio
import logging
from abc import ABC, abstractmethod

import google.generativeai as genai
import requests
from groq import Groq

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    @abstractmethod
    async def generate_async(self, prompt: str, system_message: str = None) -> str:
        pass

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY no configurada")
        self.client = Groq(api_key=api_key)
        self.model = model

    async def generate_async(self, prompt, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()
        
        def _run() -> str:
            try:
                # Build messages with role separation if system_message provided
                if system_message:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    top_p=0.9,
                    stream=False
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error en Groq API: {e}")
                raise e

        # Using run_in_executor to avoid blocking the event loop
        return await loop.run_in_executor(None, _run)

class OpenRouterLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY no configurada")
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    async def generate_async(self, prompt, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()
        
        def _run() -> str:
            try:
                # Build messages with role separation if system_message provided
                if system_message:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://facilitadores-judiciales.cr",
                        "X-Title": "Sistema Facilitadores Judiciales CR",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 2000,
                        "top_p": 0.9,
                        "stream": False
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenRouter API error: {response.text}")
                    return "Error en servicio de IA."
                    
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                logger.error(f"Error en OpenRouter API: {e}")
                raise e

        return await loop.run_in_executor(None, _run)

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY no configurada")
        genai.configure(api_key=api_key)
        # Asegurarnos de usar un modelo válido de Gemini, por defecto 2.5-flash
        model_name = model if "gemini" in model else "gemini-2.5-flash"
        self.model = genai.GenerativeModel(model_name)

    async def generate_async(self, prompt: str, system_message: str = None) -> str:
        loop = asyncio.get_running_loop()
        
        def _run() -> str:
            try:
                # Gemini maneja el system message diferente (usando instrucciones del sistema)
                # O simplemente podemos concatenarlo
                full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
                response = self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2000,
                        top_p=0.9
                    )
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error en Gemini API: {e}")
                raise e

        return await loop.run_in_executor(None, _run)
