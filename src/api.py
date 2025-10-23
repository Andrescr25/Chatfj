#!/usr/bin/env python3
"""
Chat FJ - Servicio Nacional de Facilitadoras y Facilitadores Judiciales
API optimizada con sistema híbrido (MockLLM + Groq API)
"""

import os
import sys
import logging
import re
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, AsyncGenerator

# Cargar variables de entorno desde config/config.env
from dotenv import load_dotenv
load_dotenv("config/config.env")
load_dotenv()  # .env si existe (prioridad)
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, OrderedDict
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importaciones de FastAPI
try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    from contextlib import asynccontextmanager
    import uvicorn
except ImportError as e:
    logger.error(f"Error importando FastAPI: {e}")
    sys.exit(1)

# Importaciones de LangChain
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Error importando LangChain: {e}")
    sys.exit(1)

# Importar Groq
try:
    from groq import Groq  # type: ignore
except Exception as e:
    logger.error(f"Error importando Groq: {e}")
    logger.error("Instala Groq: pip install groq")
    sys.exit(1)

# Configuración
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY no está configurada")
    logger.error("Configura tu API Key en config/config.env")
    sys.exit(1)


# Modelos de Pydantic para la API
class Message(BaseModel):
    """Mensaje en el historial de conversación."""
    role: str  # 'user' o 'assistant'
    content: str


class QueryRequest(BaseModel):
    """Modelo para peticiones de consulta."""
    question: str
    history: List[Message] = []  # Historial de conversación


class QueryResponse(BaseModel):
    """Modelo para respuestas de consulta."""
    answer: str
    sources: List[Any] = []  # Puede ser string o dict con metadata
    processing_time: float = 0.0
    cached: bool = False


class SmartCache:
    """Cache inteligente con TTL y límite de tamaño."""
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtener valor del cache si existe y no ha expirado."""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    # Mover al final (más reciente)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Expiró, eliminar
                    del self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Guardar valor en cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, time.time())
            # Limitar tamaño del cache
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Limpiar cache."""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Estadísticas del cache."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


class GroqLLM:
    """LLM usando Groq API en la nube - 1-2 segundos por respuesta."""
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        if not api_key:
            raise ValueError("GROQ_API_KEY no está configurada. Obtén una gratis en: https://console.groq.com")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.name = f"Groq {model}"
    
    async def generate_async(self, prompt: str) -> str:
        """Generación asíncrona ultra-rápida con Groq."""
        loop = asyncio.get_event_loop()
        
        def _run() -> str:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                    top_p=0.9,
                    stream=False
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error en Groq API: {e}")
                return f"Lo siento, hubo un error al procesar tu pregunta. Por favor intenta de nuevo."
        
        return await loop.run_in_executor(None, _run)


class JudicialBot:
    """Bot judicial usando Groq API y RAG con ChromaDB."""
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.vectordb = None
        self.llm = None
        self.cache = SmartCache()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.embedder = None
        
    async def initialize(self):
        """Inicialización asíncrona."""
        try:
            logger.info("🚀 Inicializando sistema...")
            
            # Cargar embeddings
            loop = asyncio.get_event_loop()
            self.embedder = await loop.run_in_executor(
                self.executor, 
                lambda: SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            )
            
            # Inicializar Groq
            logger.info(f"🚀 Usando Groq API: {GROQ_MODEL}")
            self.llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)

            # Cargar base de datos vectorial
            if os.path.exists(self.persist_dir):
                self.vectordb = await loop.run_in_executor(
                    self.executor,
                    lambda: Chroma(
                        persist_directory=self.persist_dir,
                        embedding_function=self.embedder
                    )
                )
                
                doc_count = await loop.run_in_executor(
                    self.executor,
                    lambda: self.vectordb._collection.count()
                )
                
                logger.info(f"✅ Sistema inicializado con {doc_count} documentos")
            else:
                logger.warning("⚠️ Base de datos vectorial no encontrada")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización: {e}")
            return False
    
    async def search_documents_async(self, query: str, k: int = 2) -> List[Document]:
        """Búsqueda asíncrona de documentos."""
        if not self.vectordb:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.vectordb.similarity_search(query, k=k)
            )
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def clean_answer(self, raw_text: str) -> str:
        """Limpia metainstrucciones de la respuesta."""
        if not raw_text:
            return ""
        
        lines = raw_text.splitlines()
        cleaned_lines = []
        
        # Construir patrones prohibidos, respetando ALLOW_CONTACTS
        redact_contacts = os.getenv("ALLOW_CONTACTS", "false").lower() != "true"
        forbidden_patterns = [
            r"^\s*fuente\s*:",
            r"^\s*fuentes\s*:",
            r"^\s*tiempo\s*:",
            r"estructura\s+sugerida",
            r"si no hay provincia",
            r"^\s*contexto\s*:",
            r"^\s*pregunta\s*:",
            r"^\s*respuesta\s*:",
            r"ahora responde",
            r"respuesta estructurada"
        ]
        if redact_contacts:
            forbidden_patterns.append(r"^\s*tel")
        
        forbidden_regexes = [re.compile(pat, re.IGNORECASE) for pat in forbidden_patterns]
        
        for line in lines:
            if any(rx.search(line) for rx in forbidden_regexes):
                continue
            # Evitar placeholders obvios
            if "XXXX" in line:
                continue
            cleaned_lines.append(line)
        
        cleaned = "\n".join(cleaned_lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        # Filtrar números telefónicos (según variable de entorno)
        if redact_contacts:
            phone_like = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
            cleaned = phone_like.sub("[consultar directorio oficial]", cleaned)
        
        return cleaned
    
    async def ask_async(self, question: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesamiento asíncrono ultra-rápido de preguntas con contexto conversacional."""
        start_time = time.time()
        if history is None:
            history = []
        
        try:
            # 1. Detectar saludos y consultas simples (ANTES de buscar documentos)
            question_lower = question.lower().strip()
            
            # Saludos simples
            if question_lower in ["hola", "buenos días", "buenas tardes", "buenas noches", "hey", "holi", "ola"]:
                response = {
                    "answer": """¡Hola! 👋 Soy Chat FJ, del Servicio Nacional de Facilitadoras y Facilitadores Judiciales de Costa Rica.

Estoy aquí para ayudarte con:
• Pensiones alimentarias
• Conciliaciones
• Problemas laborales
• Consultas legales
• Trámites judiciales
• Y mucho más

¿En qué te puedo ayudar hoy? Contame tu situación.""",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "cached": False
                }
                self.cache.set(question, response)
                return response
            
            # Despedidas
            if any(word in question_lower for word in ["adiós", "adios", "chao", "hasta luego", "gracias", "bye"]):
                response = {
                    "answer": """¡Con mucho gusto! 😊 

Si necesitás más ayuda en el futuro, no dudes en volver. Estamos aquí para ayudarte.

¡Que tengas un excelente día! 🌟""",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "cached": False
                }
                self.cache.set(question, response)
                return response
            
            # Preguntas sobre el bot
            if any(phrase in question_lower for phrase in ["quién sos", "quien sos", "qué sos", "que sos", "qué haces", "que haces", "para qué sirves", "para que sirves"]):
                response = {
                    "answer": """Soy Chat FJ, un asistente virtual del Servicio Nacional de Facilitadoras y Facilitadores Judiciales de Costa Rica. 🇨🇷

Mi función es:
✅ Orientarte en temas legales y judiciales
✅ Ayudarte a resolver problemas de forma práctica
✅ Darte información sobre:
   • Pensiones alimentarias
   • Conciliaciones
   • Derechos laborales
   • Trámites judiciales
   • Defensa Pública
   • Y mucho más

💡 **Importante:** Te doy orientación, pero siempre verifica la información con fuentes oficiales.

¿En qué te puedo ayudar específicamente?""",
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "cached": False
                }
                self.cache.set(question, response)
                return response
            
            # 2. Verificar cache (más rápido)
            cached_response = self.cache.get(question)
            if cached_response:
                cached_response['processing_time'] = time.time() - start_time
                return cached_response
            
            # 3. Procesamiento con RAG
            # Intensificar retrieval para respuestas más ricas
            relevant_docs = await self.search_documents_async(question, k=4)
            
            # Crear contexto limitado
            context = ""
            sources = []
            
            for doc in relevant_docs[:2]:
                filename = doc.metadata.get('filename', 'Documento')
                context += f"\n--- {filename} ---\n"
                context += (doc.page_content[:400] if doc.page_content else "") + "\n"
                
                sources.append({
                    "filename": filename,
                    "content": doc.page_content[:150] + "...",
                    "source": doc.metadata.get("source", "Desconocido")
                })
            
            # Detectar ubicación simple en la pregunta para orientar mejor
            detected_location = None
            for loc in [
                "san josé", "cartago", "alajuela", "heredia", "puntarenas",
                "guanacaste", "limón", "liberia", "pérez zeledón", "desamparados",
                "escazú", "goicoechea"
            ]:
                if loc in question.lower():
                    detected_location = loc.title()
                    break

            location_hint = f"Ubicación detectada: {detected_location}. Adapta la guía a esa localidad, menciona oficinas locales y teléfonos oficiales si se permiten." if detected_location else ""

            # Agregar historial de conversación si existe
            conversation_context = ""
            if history and len(history) > 0:
                conversation_context = "\n\n**CONVERSACIÓN PREVIA:**\n"
                for msg in history[-4:]:  # Solo las últimas 4 interacciones
                    role_label = "Usuario" if msg.get("role") == "user" else "Tú (Facilitador)"
                    conversation_context += f"{role_label}: {msg.get('content', '')}\n"
                conversation_context += "\nConsidera este contexto para dar una respuesta más personalizada y coherente.\n"

            # Crear prompt simplificado para respuestas más rápidas
            prompt = f"""Sos un asistente virtual del SNFJ. Tu objetivo es que el usuario pueda resolver su problema POR SÍ MISMO.

CRÍTICO - DOMINIO DE ESPECIALIZACIÓN:
SOLO respondes temas legales/judiciales de Costa Rica. Si preguntan matemáticas, recetas, consejos generales, etc:
Responde: "Disculpá, me especializo solo en temas legales y judiciales. ¿Tenés alguna consulta sobre pensiones, trámites o derechos?"
NO respondas temas fuera de tu área.

CRÍTICO - CONTINUIDAD CONVERSACIONAL:
Lee TODO el historial de conversación. Si el usuario hace una pregunta de seguimiento, mantené el tema original.
Ejemplo: Si habló de problemas laborales en Alajuela y pregunta por Heredia, sigue con el MISMO tema laboral pero en Heredia.

PREGUNTAS DE SEGUIMIENTO:
Si el usuario ya tiene contexto (pregunta de seguimiento), SÉ MÁS DIRECTO Y CONCISO. No repitas info ya dada.

DIRECCIONES (MUY IMPORTANTE):
Da referencias REALES que la gente conoce:
✅ "Frente al Parque Central de Alajuela"
✅ "100 metros norte del McDonald's"
✅ "Al lado del Banco Nacional"
✅ "Diagonal a la Catedral"
❌ NO solo: "Calle 2, Avenida 4"

Respondé de forma natural. NO uses etiquetas. 

Da una respuesta:
1. Reconoce el contexto previo si existe (breve)
2. Info nueva: teléfonos, direcciones CON REFERENCIAS, horarios
3. Solo documentos si es necesario
4. Al final: "¿Necesitás que te aclare algo más sobre [tema específico]?"

NO ofrezcas contactar facilitadores. VOS sos la solución completa.

TELÉFONOS REALES:
- Ministerio de Trabajo: 800-8722256
- Defensa Pública: 2287-3700
- PANI: 1147

{conversation_context}

Contexto legal:
{context}

Pregunta: {question}

Respuesta (clara, con pasos si aplica, y al final ofrecé ayuda adicional):"""
            
            # Generar respuesta asíncrona
            answer_raw = await self.llm.generate_async(prompt)
            answer = self.clean_answer(answer_raw)
            
            response = {
                "answer": answer,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "cached": False
            }
            
            # Guardar en cache
            self.cache.set(question, response)
            
            logger.info(f"✅ Respuesta generada en {response['processing_time']:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error procesando pregunta: {e}")
            return {
                "answer": "Disculpa, hubo un error técnico. Por favor intenta de nuevo en un momento.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "cached": False
            }

# Instancia global del bot
bot = JudicialBot(PERSIST_DIR)

# Configuración de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Iniciando API...")
    success = await bot.initialize()
    if not success:
        logger.error("❌ Error en inicialización")
    yield
    # Shutdown
    logger.info("👋 Cerrando API...")

app = FastAPI(
    title="Bot de Facilitadores Judiciales",
    description="API optimizada para consultas judiciales con respuestas rápidas y amables",
    version="2.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Verificación de salud del sistema."""
    cache_stats = bot.cache.stats()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "cache_stats": cache_stats,
        "features": [
            "Cache inteligente",
            "Respuestas precomputadas", 
            "Procesamiento asíncrono",
            "Operaciones paralelas",
            "Limpieza de respuestas"
        ]
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Endpoint principal para preguntas con respuestas optimizadas."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    # Convertir history de Message a dict si es necesario
    history_dicts = [msg.dict() if hasattr(msg, 'dict') else msg for msg in request.history]
    response = await bot.ask_async(request.question, history=history_dicts)
    return QueryResponse(**response)

@app.post("/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """Endpoint con respuesta streaming para percepción de velocidad."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    async def generate_stream():
        # Obtener respuesta completa con historial
        history_dicts = [msg.dict() if hasattr(msg, 'dict') else msg for msg in request.history]
        response = await bot.ask_async(request.question, history=history_dicts)
        answer = response["answer"]
        
        # Simular streaming por palabras para percepción de velocidad
        words = answer.split()
        for i, word in enumerate(words):
            chunk = {
                "word": word,
                "is_final": i == len(words) - 1,
                "processing_time": response["processing_time"],
                "cached": response["cached"]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)  # 50ms entre palabras
        
        # Enviar fuentes al final
        if response["sources"]:
            sources_chunk = {
                "sources": response["sources"],
                "is_sources": True
            }
            yield f"data: {json.dumps(sources_chunk, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/stats")
async def get_stats():
    """Estadísticas del sistema."""
    return {
        "cache_stats": bot.cache.stats(),
        "precomputed_responses": len(bot.precomputed.responses),
        "system_status": "optimal"
    }

@app.get("/documents")
async def get_documents():
    """Obtiene información sobre los documentos cargados."""
    total_docs = 0
    sample_docs = []
    
    try:
        if bot.vectordb:
            # Obtener conteo total de documentos
            collection = bot.vectordb._collection
            total_docs = collection.count()
            
            # Obtener muestra de documentos (primeros 5)
            if total_docs > 0:
                results = collection.get(limit=5)
                if results and 'documents' in results:
                    sample_docs = [
                        {
                            "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            "id": results['ids'][i] if 'ids' in results else str(i)
                        }
                        for i, doc in enumerate(results['documents'])
                    ]
        
        return {
            "total_documents": total_docs,
            "sample_documents": sample_docs,
            "vector_db_status": "active" if bot.vectordb else "inactive"
        }
    except Exception as e:
        logger.error(f"Error obteniendo documentos: {e}")
        return {
            "total_documents": 0,
            "sample_documents": [],
            "vector_db_status": "error",
            "error": str(e)
        }

@app.post("/clear-cache")
async def clear_cache():
    """Limpia el cache del sistema."""
    bot.cache.clear()
    return {"message": "Cache limpiado exitosamente"}

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop="asyncio",
        log_level="info"
    )
