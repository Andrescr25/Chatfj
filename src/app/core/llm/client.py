import logging
import asyncio
from typing import Optional
from abc import ABC, abstractmethod
from groq import Groq
import requests

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        pass

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY no configurada")
        self.client = Groq(api_key=api_key)
        self.model = model

    async def generate_async(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        
        def _run() -> str:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=2000,
                    top_p=0.95,
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

    async def generate_async(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        
        def _run() -> str:
            try:
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
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "max_tokens": 2000,
                        "top_p": 0.95
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
