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
import requests
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple
from duckduckgo_search import DDGS

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
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
print(f"DEBUG: PERSIST_DIR = {PERSIST_DIR}")
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
    """Cache inteligente con TTL, límite de tamaño y persistencia SQLite (Optimización 2025-10-24)."""
    def __init__(self, max_size: int = 1000, ttl: int = 3600, db_path: str = "data/cache.db"):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Inicializa base de datos SQLite para persistencia."""
        try:
            import sqlite3
            import os
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.info(f"✅ Cache persistente inicializado: {self.db_path}")

            # Cargar cache en memoria
            self._load_from_db()
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar cache persistente: {e}")

    def _load_from_db(self):
        """Carga cache desde DB a memoria."""
        try:
            import sqlite3
            import json

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value, timestamp FROM cache ORDER BY timestamp DESC LIMIT ?", (self.max_size,))

            current_time = time.time()
            loaded = 0
            for key, value_json, timestamp in cursor.fetchall():
                # Solo cargar si no ha expirado
                if current_time - timestamp < self.ttl:
                    self.cache[key] = (json.loads(value_json), timestamp)
                    loaded += 1

            conn.close()
            if loaded > 0:
                logger.info(f"📦 Cache cargado desde DB: {loaded} entradas")
        except Exception as e:
            logger.warning(f"⚠️ Error cargando cache desde DB: {e}")

    def _save_to_db(self, key: str, value: Dict[str, Any], timestamp: float):
        """Guarda una entrada en la DB."""
        try:
            import sqlite3
            import json

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), timestamp)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"⚠️ Error guardando en cache DB: {e}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtener valor del cache si existe y no ha expirado."""
        with self.lock:
            # Primero buscar en memoria
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    self.cache.move_to_end(key)
                    return value
                else:
                    del self.cache[key]

            # Si no está en memoria, buscar en DB
            try:
                import sqlite3
                import json

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT value, timestamp FROM cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    value_json, timestamp = row
                    if time.time() - timestamp < self.ttl:
                        # Restaurar a memoria
                        value = json.loads(value_json)
                        self.cache[key] = (value, timestamp)
                        self.hits += 1
                        return value
            except Exception as e:
                logger.warning(f"⚠️ Error leyendo cache DB: {e}")

            self.misses += 1
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Guardar valor en cache (memoria y DB)."""
        with self.lock:
            timestamp = time.time()

            # Guardar en memoria
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, timestamp)

            # Limitar tamaño del cache en memoria
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            # Guardar en DB de forma asíncrona (no bloquear)
            threading.Thread(target=self._save_to_db, args=(key, value, timestamp), daemon=True).start()

    def clear(self) -> None:
        """Limpiar cache."""
        with self.lock:
            self.cache.clear()
            try:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache")
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando cache DB: {e}")

    def stats(self) -> Dict[str, Any]:
        """Estadísticas del cache."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        # Contar entradas en DB
        db_size = 0
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cache")
            db_size = cursor.fetchone()[0]
            conn.close()
        except:
            pass

        return {
            "size": len(self.cache),
            "db_size": db_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


class LegalVerificationHelper:
    """Ayudante para verificar leyes vigentes y validar referencias legales."""

    _leyes_db = None

    @classmethod
    def load_leyes_db(cls):
        """Carga la base de datos de leyes vigentes."""
        if cls._leyes_db is None:
            try:
                import json
                leyes_path = "data/leyes_vigentes.json"
                with open(leyes_path, 'r', encoding='utf-8') as f:
                    cls._leyes_db = json.load(f)
                logger.info(f"✅ Base de leyes vigentes cargada: {len(cls._leyes_db.get('leyes', {}))} leyes")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo cargar base de leyes: {e}")
                cls._leyes_db = {"leyes": {}, "instituciones": {}}
        return cls._leyes_db

    @classmethod
    def verify_ley(cls, ley_id: str) -> dict:
        """Verifica si una ley está vigente y retorna su info."""
        db = cls.load_leyes_db()
        ley_id_clean = ley_id.lower().replace("ley", "").replace("n.°", "").replace("°", "").strip()

        if ley_id_clean in db.get("leyes", {}):
            return db["leyes"][ley_id_clean]
        return None

    @classmethod
    def get_institucion_info(cls, institucion_id: str) -> dict:
        """Obtiene información sobre una institución."""
        db = cls.load_leyes_db()
        return db.get("instituciones", {}).get(institucion_id, None)

    @classmethod
    def validate_citation(cls, text: str) -> dict:
        """Valida citas legales en el texto y retorna info."""
        import re
        db = cls.load_leyes_db()

        # Buscar menciones de leyes
        ley_pattern = r'Ley\s+(?:N\.°\s*)?(\d+)'
        matches = re.findall(ley_pattern, text, re.IGNORECASE)

        validated = {
            "leyes_mencionadas": [],
            "leyes_validas": [],
            "leyes_invalidas": []
        }

        for ley_num in matches:
            validated["leyes_mencionadas"].append(ley_num)
            if ley_num in db.get("leyes", {}):
                validated["leyes_validas"].append(ley_num)
            else:
                validated["leyes_invalidas"].append(ley_num)

        return validated

    @classmethod
    def validate_category_citation_match(cls, category: str, text: str) -> dict:
        """
        Valida que las leyes citadas en el texto coincidan con la categoría detectada.
        Retorna: {"is_valid": bool, "errors": [], "corrections": []}
        """
        import re
        db = cls.load_leyes_db()

        # Mapeo de categorías a leyes esperadas
        category_to_laws = {
            "pension_alimentaria": ["7654", "codigo_familia"],
            "laboral": ["codigo_trabajo"],
            "pension_vejez": ["7935", "ccss"],
            "violencia": ["violencia_domestica", "7586"],
            "penal": ["codigo_penal"],
            "civil": ["codigo_civil", "arrendamientos"],
            "menores": ["pani", "7739"],
            "discapacidad": ["7600"],
            "adulto_mayor": ["7935"],
            "general": []
        }

        expected_laws = category_to_laws.get(category, [])

        # Buscar menciones de leyes en el texto
        ley_pattern = r'Ley\s+(?:N\.°\s*)?(\d+)'
        codigo_pattern = r'Código\s+(?:Procesal\s+)?(?:de\s+)?([A-Za-záéíóúñ]+)'

        ley_matches = re.findall(ley_pattern, text, re.IGNORECASE)
        codigo_matches = re.findall(codigo_pattern, text, re.IGNORECASE)

        result = {
            "is_valid": True,
            "errors": [],
            "corrections": [],
            "cited_laws": ley_matches + codigo_matches
        }

        # Si no hay leyes esperadas (categoría general), aceptar cualquier cita
        if not expected_laws:
            return result

        # Verificar cada ley citada
        for cited_law in ley_matches:
            if cited_law not in expected_laws:
                result["is_valid"] = False
                # Buscar nombre de la ley incorrecta
                ley_info = db.get("leyes", {}).get(cited_law)
                ley_name = ley_info.get("nombre", f"Ley {cited_law}") if ley_info else f"Ley {cited_law}"

                # Buscar nombre de la ley correcta
                correct_law_id = expected_laws[0] if expected_laws else None
                if correct_law_id:
                    correct_info = db.get("leyes", {}).get(correct_law_id)
                    correct_name = correct_info.get("nombre", "") if correct_info else ""

                    result["errors"].append(
                        f"Cita incorrecta: {ley_name} no corresponde a categoría {category}"
                    )
                    result["corrections"].append(
                        f"Debería citar: {correct_name} (relacionada con {category})"
                    )

        return result


class WebSearchHelper:
    """Ayudante para buscar información actualizada en web usando DuckDuckGo."""

    @staticmethod
    def _get_generic_contact_info(query_lower: str) -> str:
        """Retorna información de contacto genérica para instituciones conocidas."""
        if any(word in query_lower for word in ["pensión", "pension", "alimentaria", "cuota"]):
            return "Ministerio de Trabajo: 800-MTSS (800-6877). Poder Judicial: 800-PODER-J"
        if any(word in query_lower for word in ["trabajo", "laboral", "despido", "horas"]):
            return "Ministerio de Trabajo y Seguridad Social: Tel 800-MTSS (800-6877)"
        if any(word in query_lower for word in ["pani", "niños", "menores"]):
            return "PANI: Tel 1147 (línea gratuita 24/7)"
        if any(word in query_lower for word in ["violencia", "denuncia", "oij"]):
            return "OIJ: Tel 800-8000-645. Emergencias: 911"
        return "Poder Judicial de Costa Rica: Tel 2295-3774"

    @staticmethod
    async def search_web_info(query: str, location: str = None) -> Tuple[str, List[Dict[str, str]]]:
        """
        Busca información actualizada en web (teléfonos, direcciones, horarios).
        Retorna: (texto_resumen, [{"title": "", "url": "", "snippet": ""}])
        """
        try:
            # Detectar qué tipo de información necesitamos
            query_lower = query.lower()

            # Construir búsqueda específica para Costa Rica
            search_queries = []

            if any(word in query_lower for word in ["pensión", "pension", "alimentaria"]):
                if location:
                    search_queries.append(f"pensiones alimentarias teléfono dirección {location} Costa Rica")
                else:
                    search_queries.append("pensiones alimentarias Costa Rica teléfono contacto")

            if any(word in query_lower for word in ["defensa", "pública", "abogado"]):
                if location:
                    search_queries.append(f"Defensa Pública teléfono {location} Costa Rica")
                else:
                    search_queries.append("Defensa Pública Costa Rica teléfono contacto")

            if any(word in query_lower for word in ["pani", "niños", "menores"]):
                if location:
                    search_queries.append(f"PANI oficina teléfono {location} Costa Rica")
                else:
                    search_queries.append("PANI Costa Rica teléfono 1147")

            if any(word in query_lower for word in ["trabajo", "laboral", "despido", "ministerio"]):
                if location:
                    search_queries.append(f"Ministerio de Trabajo oficina {location} Costa Rica teléfono")
                else:
                    search_queries.append("Ministerio de Trabajo Costa Rica teléfono")

            if any(word in query_lower for word in ["violencia", "oij", "denuncia"]):
                if location:
                    search_queries.append(f"OIJ oficina {location} Costa Rica teléfono")
                else:
                    search_queries.append("OIJ Costa Rica teléfono 800-8000-645")

            # Si no hay queries específicas, búsqueda genérica
            if not search_queries:
                if location:
                    search_queries.append(f"{query} {location} Costa Rica teléfono dirección")
                else:
                    search_queries.append(f"{query} Costa Rica contacto")

            # Realizar búsqueda web
            loop = asyncio.get_event_loop()
            results = []

            for search_query in search_queries[:1]:  # Solo la primera búsqueda
                def _search():
                    try:
                        with DDGS() as ddgs:
                            return list(ddgs.text(search_query, max_results=3, region='cr-es'))
                    except Exception as e:
                        logger.warning(f"Error en búsqueda web: {e}")
                        return []

                search_results = await loop.run_in_executor(None, _search)
                if search_results:
                    results.extend(search_results)
                    break

            if not results:
                return "", []

            # Filtrar solo resultados de Costa Rica o dominios oficiales conocidos
            # Ser más flexible: aceptar sitios .cr O sitios que mencionen "costa rica" en title/body
            valid_domains = [
                '.cr', '.go.cr', 'poderjudicial.go.cr', 'pani.go.cr',
                'mtss.go.cr', 'oij.go.cr', 'defensapublica.cr',
                'ccss.sa.cr', 'tse.go.cr', 'asamblea.go.cr',
                'poder-judicial.go.cr', 'ministeriopublico.go.cr'
            ]

            filtered_results = []
            for result in results:
                href = result.get('href', '').lower()
                title = result.get('title', '').lower()
                body = result.get('body', '').lower()

                # Verificar que sea de Costa Rica (dominio O contenido)
                is_cr_domain = any(domain in href for domain in valid_domains)
                is_cr_content = 'costa rica' in title or 'costa rica' in body

                if is_cr_domain or is_cr_content:
                    filtered_results.append(result)

            # Si no hay resultados filtrados, retornar información genérica
            if not filtered_results:
                logger.warning(f"No se encontraron resultados de sitios costarricenses. Resultados originales: {[r.get('href') for r in results[:3]]}")
                # En lugar de no devolver nada, devolver info genérica de instituciones conocidas
                return WebSearchHelper._get_generic_contact_info(query_lower), []

            # Formatear resultados
            web_info = []
            sources = []

            for result in filtered_results[:2]:  # Top 2 resultados válidos
                title = result.get('title', '')
                body = result.get('body', '')
                href = result.get('href', '')

                web_info.append(f"{title}: {body[:200]}")
                sources.append({
                    "title": title,
                    "url": href,
                    "snippet": body[:250]
                })

            return "\n".join(web_info), sources

        except Exception as e:
            logger.error(f"Error en WebSearchHelper: {e}")
            return "", []


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
            logger.info(f"🔍 Buscando ChromaDB en: {self.persist_dir}")
            logger.info(f"🔍 Path exists: {os.path.exists(self.persist_dir)}")
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
    
    def detect_ambiguity(self, question: str) -> dict:
        """
        Detecta si una pregunta es ambigua y genera preguntas aclaratorias.

        Returns:
            dict con:
            - is_ambiguous: bool
            - ambiguity_type: str (keyword, context, multiple_scenarios)
            - clarifying_questions: list[str]
            - possible_categories: list[str]
        """
        question_lower = question.lower()

        # Palabras clave ambiguas y sus contextos posibles
        AMBIGUOUS_TERMS = {
            "acoso": {
                "contexts": ["laboral", "sexual", "escolar", "cibernético", "callejero", "violencia doméstica"],
                "questions": [
                    "¿El acoso es en el trabajo, en la escuela, en la calle, o en su hogar?",
                    "¿Quién lo está acosando? (jefe, compañero, pareja, extraño)",
                    "¿Es acoso físico, verbal, sexual, o por redes sociales?"
                ],
                "categories": ["laboral", "violencia", "penal", "menores"]
            },
            "denuncia": {
                "contexts": ["violencia", "robo", "estafa", "maltrato", "corrupción"],
                "questions": [
                    "¿Qué tipo de situación quiere denunciar?",
                    "¿Es un delito (robo, agresión), violencia doméstica, o un problema laboral/civil?"
                ],
                "categories": ["penal", "violencia", "laboral", "civil"]
            },
            "pensión": {
                "contexts": ["alimentaria", "vejez", "invalidez", "viudez"],
                "questions": [
                    "¿Es pensión alimentaria (para hijos) o pensión de la CCSS (vejez/invalidez)?",
                    "¿Para quién es la pensión?"
                ],
                "categories": ["pension_alimentaria", "pension_vejez"]
            },
            "demanda": {
                "contexts": ["laboral", "civil", "familia", "pensión"],
                "questions": [
                    "¿Qué tipo de demanda quiere interponer?",
                    "¿Es por despido, divorcio, desalojo, pensión, o deuda?"
                ],
                "categories": ["laboral", "civil", "pension_alimentaria"]
            },
            "hijo": {
                "contexts": ["pensión", "custodia", "maltrato", "registro"],
                "questions": [
                    "¿Su consulta es sobre pensión alimentaria, custodia, protección del PANI, o registro civil?",
                    "¿Qué necesita resolver con respecto a su hijo/a?"
                ],
                "categories": ["pension_alimentaria", "menores", "civil"]
            },
            "despido": {
                "contexts": ["con causa", "sin causa", "embarazo", "discriminación"],
                "questions": [
                    "¿Lo despidieron con preaviso o sin preaviso?",
                    "¿Le dieron razones? ¿Considera que fue injusto o discriminatorio?"
                ],
                "categories": ["laboral"]
            },
            "desalojo": {
                "contexts": ["inquilino", "propietario", "falta de pago", "vencimiento"],
                "questions": [
                    "¿Usted es el inquilino o el propietario?",
                    "¿Cuál es la razón del desalojo? (falta de pago, fin de contrato, otra)"
                ],
                "categories": ["civil"]
            }
        }

        # Detectar términos ambiguos en la pregunta
        detected_ambiguities = []
        possible_categories = set()

        for term, config in AMBIGUOUS_TERMS.items():
            if term in question_lower:
                detected_ambiguities.append({
                    "term": term,
                    "questions": config["questions"],
                    "categories": config["categories"]
                })
                possible_categories.update(config["categories"])

        # Detectar preguntas muy cortas (< 8 palabras = probable ambigüedad)
        word_count = len(question.split())
        is_too_short = word_count < 8

        # Detectar falta de detalles específicos o contexto claro
        has_details = any(keyword in question_lower for keyword in [
            "trabajo", "jefe", "esposo", "esposa", "pareja", "hijo", "hija",
            "contrato", "pago", "dinero", "violencia", "golpes", "amenazas",
            "ccss", "pani", "juzgado", "meses", "años", "documento"
        ])

        # Detectar si ya se especificó el tipo de acoso/denuncia/pensión
        has_type_specified = any(combo in question_lower for combo in [
            "acoso laboral", "acoso sexual", "acoso escolar", "acoso callejero",
            "violencia doméstica", "violencia intrafamiliar",
            "pensión alimentaria", "pensión vejez", "pensión invalidez",
            "demanda laboral", "demanda civil", "demanda pensión"
        ])

        is_ambiguous = (len(detected_ambiguities) > 0 and not has_details and not has_type_specified) or (is_too_short and len(detected_ambiguities) > 0 and not has_type_specified)

        if is_ambiguous:
            # Consolidar preguntas aclaratorias
            all_questions = []
            for ambiguity in detected_ambiguities:
                all_questions.extend(ambiguity["questions"][:2])  # Máximo 2 por término

            return {
                "is_ambiguous": True,
                "ambiguity_type": "keyword" if detected_ambiguities else "too_short",
                "clarifying_questions": all_questions[:3],  # Máximo 3 preguntas totales
                "possible_categories": list(possible_categories),
                "detected_terms": [a["term"] for a in detected_ambiguities]
            }

        return {
            "is_ambiguous": False,
            "clarifying_questions": [],
            "possible_categories": [],
            "detected_terms": []
        }

    def expand_query(self, query: str) -> str:
        """Expande la consulta con sinónimos y términos relacionados para mejor retrieval."""
        query_lower = query.lower()

        # Detectar contexto específico primero
        # Si menciona "ex" + "no paga" + "pensión" → es pensión ALIMENTARIA
        is_child_support = any(word in query_lower for word in ["ex", "hijos", "hijo", "niños"]) and \
                          any(word in query_lower for word in ["pensión", "pension", "paga"]) and \
                          any(word in query_lower for word in ["no paga", "incumpl", "debe"])

        # Si menciona "jubil" o "vejez" o "CCSS" → es pensión de VEJEZ
        is_retirement = any(word in query_lower for word in ["jubil", "vejez", "ccss", "edad", "retiro", "años"])

        # Diccionario de expansiones para términos legales comunes
        expansions = {
            "pensión": "pensión alimentaria manutención obligación alimentaria cuota familia" if is_child_support else "pensión",
            "pension": "pensión alimentaria manutención obligación alimentaria cuota familia" if is_child_support else "pensión",
            "divorcio": "divorcio separación disolución matrimonio familia",
            "despido": "despido cesantía terminación laboral trabajo",
            "trabajo": "trabajo laboral empleador trabajador relación laboral",
            "violencia": "violencia doméstica agresión familiar maltrato familia",
            "niños": "niños menores niñez infancia hijos familia",
            "hijos": "hijos menores niñez infancia descendientes familia",
            "ex": "ex pareja cónyuge progenitor familia",
            "paga": "paga cumple cancela deposita cuota obligación"
        }

        expanded_terms = [query]
        for key, expansion in expansions.items():
            if key in query_lower:
                expanded_terms.append(expansion)

        # Si es pensión alimentaria, agregar términos específicos
        if is_child_support:
            expanded_terms.append("familia procesal incumplimiento")

        return " ".join(expanded_terms)

    async def search_documents_async(self, query: str, k: int = 6) -> List[Document]:
        """Búsqueda asíncrona con query expansion y mayor recuperación."""
        if not self.vectordb:
            return []

        try:
            # Expandir query para mejor recall
            expanded_query = self.expand_query(query)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.vectordb.similarity_search(expanded_query, k=k)
            )
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []

    def categorize_document(self, filename: str, content: str) -> str:
        """Categoriza documentos por tipo de caso legal."""
        filename_lower = filename.lower()
        content_lower = content.lower()

        # Categorías principales - OPTIMIZADO 2025-10-24
        categories = {
            "pension_alimentaria": ["pensión alimentaria", "pension", "familia", "procesal", "cuota", "alimentaria", "ex pareja", "hijo", "manutención"],
            "laboral": ["trabajo", "laboral", "empleador", "despido", "salario", "jornada", "finiquito", "preaviso", "horas extra"],
            "pension_vejez": ["vejez", "ccss", "invalidez", "jubilación", "seguro social", "adulto mayor", "retiro"],
            "violencia": ["violencia", "maltrato", "agresión", "domestica", "golpea", "violencia doméstica", "protección", "abuso"],
            "penal": ["penal", "delito", "contravención", "criminal", "denuncia", "bienestar animal"],
            "civil": ["civil", "contrato", "obligaciones", "propiedad", "desalojo", "arrendamiento", "alquiler", "inquilino", "casa alquilada"],
            "menores": ["pani", "menor", "niñez", "infancia", "juvenil", "niño", "hijo", "hija", "niña", "peligro", "abuso infantil"],
            "migracion": ["migrante", "refugiado", "apatridia", "migración"],
            "conciliacion": ["conciliación", "mediación", "facilitador"],
            "constitucional": ["constitución", "derechos humanos", "sala constitucional"]
        }

        # Buscar categoría
        for category, keywords in categories.items():
            if any(kw in filename_lower or kw in content_lower[:500] for kw in keywords):
                return category

        return "general"

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 3, return_scores: bool = False) -> List[Document]:
        """Reordena documentos por relevancia semántica real a la pregunta con contexto inteligente."""
        if not documents:
            return []

        query_lower = query.lower()

        # Detectar contexto de la consulta con categorización mejorada - OPTIMIZADO 2025-10-24
        query_category = None

        # MENORES (prioridad alta - nueva)
        is_menores = any(word in query_lower for word in ["pani", "niño", "niña", "hijo", "hija", "menor", "infancia"]) and \
                     any(word in query_lower for word in ["peligro", "abuso", "maltrato", "ayuda", "protección"])
        if is_menores:
            query_category = "menores"

        # CIVIL - Desalojos y arrendamientos (prioridad alta - nueva)
        is_civil = any(word in query_lower for word in ["desalojo", "desalojar", "arrendamiento", "alquiler", "inquilino", "casa alquilada", "renta"])
        if is_civil and not is_menores:
            query_category = "civil"

        # Pensión alimentaria (hijos) - MEJORADA
        is_child_support = (any(word in query_lower for word in ["ex", "hijos", "hijo", "niños", "niñas", "alimentaria"]) and \
                           any(word in query_lower for word in ["pensión", "pension", "paga", "cuota", "manutención"])) or \
                          ("aumento" in query_lower and "pensión" in query_lower and any(word in query_lower for word in ["hijo", "ex", "alimentaria"]))
        if is_child_support and not is_menores:
            query_category = "pension_alimentaria"

        # Pensión de vejez/invalidez - MEJORADA
        is_retirement = (any(word in query_lower for word in ["jubil", "vejez", "ccss", "edad", "retiro", "años", "invalidez", "adulto mayor"]) or \
                        ("aumento" in query_lower and "pensión" in query_lower and any(word in query_lower for word in ["vejez", "ccss", "jubilación"]))) and \
                       not is_child_support
        if is_retirement and not is_menores:
            query_category = "pension_vejez"

        # Laboral - MEJORADA
        is_work = any(word in query_lower for word in ["despido", "despidieron", "trabajo", "laboral", "empleador", "salario", "horas extra", "jefe", "jornada", "pago", "finiquito", "cesantía", "preaviso"])
        if is_work and not is_menores:
            query_category = "laboral"

        # Violencia - MEJORADA
        is_violence = any(word in query_lower for word in ["violencia", "maltrato", "agresión", "golpe", "golpea", "abuso", "protección"]) and \
                     not is_menores  # Evitar confusión con casos de menores
        if is_violence:
            query_category = "violencia"

        # Términos clave para scoring
        query_terms = set(query_lower.split())

        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            filename = doc.metadata.get('filename', '')
            filename_lower = filename.lower()

            # Categorizar documento
            doc_category = self.categorize_document(filename, doc.page_content)

            # Score base por coincidencia de términos
            content_score = sum(1 for term in query_terms if term in content_lower)
            filename_score = sum(3 for term in query_terms if term in filename_lower)

            # Bonificaciones por CATEGORIZACIÓN
            category_bonus = 0

            # CATEGORÍA EXACTA: Gran bonus
            if query_category and query_category == doc_category:
                category_bonus += 50
                logger.info(f"   🎯 Categoría exacta: {doc_category} - {filename}")

            # CATEGORÍA INCOMPATIBLE: Fuerte penalización
            incompatible_pairs = {
                "pension_alimentaria": ["laboral", "pension_vejez"],
                "laboral": ["pension_alimentaria", "pension_vejez", "menores"],
                "pension_vejez": ["pension_alimentaria", "laboral"]
            }

            if query_category and query_category in incompatible_pairs:
                if doc_category in incompatible_pairs[query_category]:
                    category_bonus -= 60
                    logger.info(f"   ❌ Categoría incompatible: query={query_category}, doc={doc_category} - {filename}")

            # Bonificaciones adicionales por contexto
            context_bonus = 0

            # Pensión alimentaria
            if is_child_support:
                if "familia" in filename_lower or "procesal" in filename_lower or "7654" in content_lower:
                    context_bonus += 15
                if "alimentaria" in content_lower or "alimentaria" in filename_lower:
                    context_bonus += 10

            # Pensión vejez
            if is_retirement:
                if "ccss" in filename_lower or "vejez" in filename_lower or "reglamento" in filename_lower:
                    context_bonus += 15

            # Laboral
            if is_work:
                if "trabajo" in filename_lower or "laboral" in filename_lower or "código" in filename_lower:
                    context_bonus += 20

            # Violencia
            if is_violence:
                if "violencia" in filename_lower or "familia" in filename_lower:
                    context_bonus += 15

            # Score total
            total_score = (content_score * 0.5) + (filename_score * 0.5) + category_bonus + context_bonus
            scored_docs.append((total_score, doc, doc_category))

        # Ordenar por score descendente
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Log para debugging (solo top 3)
        logger.info(f"🔍 Reranking results for: {query[:50]}...")
        logger.info(f"   📂 Query category: {query_category or 'general'}")
        for i, (score, doc, category) in enumerate(scored_docs[:3], 1):
            logger.info(f"   {i}. [{score:.1f}] {category} - {doc.metadata.get('filename', 'Unknown')}")

        # Retornar top_k documentos (con o sin scores)
        if return_scores:
            return scored_docs[:top_k]  # Retorna tuplas (score, doc, category)
        else:
            return [doc for score, doc, category in scored_docs[:top_k]]
    
    def clean_answer(self, raw_text: str) -> str:
        """Limpia metainstrucciones y formato markdown excesivo, preservando citas."""
        if not raw_text:
            return ""

        lines = raw_text.splitlines()
        cleaned_lines = []

        # Construir patrones prohibidos
        redact_contacts = os.getenv("ALLOW_CONTACTS", "false").lower() != "true"
        forbidden_patterns = [
            r"^\s*tiempo\s*:",
            r"estructura\s+sugerida",
            r"si no hay provincia",
            r"^\s*contexto\s*:",
            r"^\s*pregunta del usuario\s*:",
            r"^\s*tu respuesta\s*:",
            r"^\s*\*\*pregunta del usuario\*\*",
            r"^\s*\*\*tu respuesta\*\*",
            r"^\s*respuesta\s*\(siguiendo",
            r"ahora responde",
            r"respuesta estructurada",
            r"^\s*\*\*reglas fundamentales\*\*",
            r"^\s*\*\*fuentes legales disponibles\*\*",
            r"^\s*importante\s*:",
            r"lenguaje simple",
            r"evita jerga"
        ]

        if redact_contacts:
            forbidden_patterns.append(r"^\s*tel(?!éfono)")

        forbidden_regexes = [re.compile(pat, re.IGNORECASE) for pat in forbidden_patterns]

        for line in lines:
            # Permitir líneas con citaciones legales
            if re.search(r"según\s+(?:el|la|los)", line, re.IGNORECASE):
                cleaned_lines.append(line)
                continue

            if any(rx.search(line) for rx in forbidden_regexes):
                continue
            if "XXXX" in line:
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Eliminar markdown excesivo pero mantener referencias [1], [2]
        # Eliminar asteriscos de negritas ** pero NO tocar [1], [2], [3]
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **texto** → texto
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)      # *texto* → texto

        # Eliminar bullets markdown pero mantener números de pasos
        cleaned = re.sub(r'^\s*[\*\-]\s+', '', cleaned, flags=re.MULTILINE)

        # Filtrar teléfonos si es necesario
        if redact_contacts:
            phone_like = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
            cleaned = phone_like.sub("[consultar directorio oficial]", cleaned)

        return cleaned
    
    def extract_keywords(self, question: str) -> List[str]:
        """Extrae palabras clave relevantes de la pregunta del usuario."""
        question_lower = question.lower()

        # Palabras a ignorar (stopwords en español)
        stopwords = {
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "de", "del", "en", "a", "por", "para", "con", "sin",
            "que", "mi", "tu", "su", "me", "te", "se", "le",
            "y", "o", "pero", "si", "no", "como", "cuando", "donde",
            "es", "está", "son", "están", "ser", "estar", "hay",
            "tengo", "tiene", "quiero", "necesito", "puedo", "debo"
        }

        # Extraer palabras (sin stopwords)
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        return keywords[:5]  # Top 5 keywords

    async def ask_async(self, question: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        FLUJO HÍBRIDO REDISEÑADO:
        1) Buscar en internet con keywords
        2) Comparar con base vectorial
        3) Generar respuesta híbrida breve
        4) Adjuntar referencias con links
        """
        start_time = time.time()
        if history is None:
            history = []

        try:
            # 1. Detectar saludos y consultas simples
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

            # 2. Verificar cache
            cached_response = self.cache.get(question)
            if cached_response:
                cached_response['processing_time'] = time.time() - start_time
                return cached_response

            # ============================================
            # PASO 0: DETECTAR AMBIGÜEDAD
            # ============================================
            ambiguity_check = self.detect_ambiguity(question)

            if ambiguity_check["is_ambiguous"]:
                logger.info(f"⚠️ Pregunta ambigua detectada: {ambiguity_check['detected_terms']}")

                # Generar respuesta con preguntas aclaratorias
                clarifying_text = "Para poder ayudarte mejor, necesito que me des más detalles:\n\n"
                for i, q in enumerate(ambiguity_check["clarifying_questions"], 1):
                    clarifying_text += f"{i}. {q}\n"

                clarifying_text += "\n💡 **Tip:** Mientras más detalles me des sobre tu situación, mejor podré orientarte con la información legal correcta."

                response = {
                    "answer": clarifying_text,
                    "sources": [],
                    "processing_time": time.time() - start_time,
                    "cached": False,
                    "is_clarification": True,
                    "ambiguity_info": ambiguity_check
                }
                # NO cachear preguntas ambiguas
                return response

            # ============================================
            # NUEVO FLUJO HÍBRIDO
            # ============================================

            logger.info(f"🔍 Iniciando flujo híbrido para: {question[:50]}...")

            # PASO 1: Extraer keywords y buscar en internet
            keywords = self.extract_keywords(question)
            logger.info(f"📌 Keywords extraídas: {keywords}")

            # Construir query de búsqueda web
            web_query = " ".join(keywords) + " Costa Rica ley derecho"

            # Detectar ubicación
            detected_location = None
            for loc in [
                "san josé", "cartago", "alajuela", "heredia", "puntarenas",
                "guanacaste", "limón", "liberia", "pérez zeledón", "desamparados",
                "escazú", "goicoechea", "san carlos", "limón centro", "nicoya",
                "turrialba", "grecia", "palmares"
            ]:
                if loc in question.lower():
                    detected_location = loc.title()
                    break

            # PASO 2: Buscar en base vectorial con scoring
            # Optimización: Reducido k de 4 a 3 para mejorar velocidad (2025-10-24)
            relevant_docs = await self.search_documents_async(question, k=3)
            reranked_docs_with_scores = self.rerank_documents(question, relevant_docs, top_k=2, return_scores=True)

            # Extraer documentos, scores y categoría detectada
            detected_category = "general"
            if reranked_docs_with_scores and isinstance(reranked_docs_with_scores[0], tuple):
                reranked_docs = [doc for score, doc, category in reranked_docs_with_scores]
                best_score = reranked_docs_with_scores[0][0] if reranked_docs_with_scores else 0
                # Capturar categoría detectada del primer documento (mejor match)
                detected_category = reranked_docs_with_scores[0][2] if len(reranked_docs_with_scores[0]) > 2 else "general"
            else:
                reranked_docs = reranked_docs_with_scores
                best_score = 0

            # THRESHOLD: Solo buscar en web si confianza es baja (<65 puntos)
            # Optimización: Reducido de 70.0 a 65.0 para capturar más casos válidos
            confidence_threshold = 65.0
            should_search_web = best_score < confidence_threshold

            logger.info(f"📊 Best doc score: {best_score:.1f} - Threshold: {confidence_threshold}")

            # PASO 2.5: Buscar en web SOLO si confianza es baja O usuario menciona ubicación
            web_info = ""
            web_sources = []

            if should_search_web or detected_location:
                if should_search_web:
                    logger.info("⚠️ Confianza baja - Activando búsqueda web complementaria")
                else:
                    logger.info(f"📍 Ubicación detectada ({detected_location}) - Buscando info local")

                web_info, web_sources = await WebSearchHelper.search_web_info(question, detected_location)

            # PASO 3: Comparar y crear contexto híbrido
            hybrid_context = ""
            sources = []

            # Primero: Información web (si existe)
            if web_info and web_sources:
                hybrid_context += "--- INFORMACIÓN WEB ACTUALIZADA ---\n"
                hybrid_context += web_info + "\n\n"

                for idx, web_src in enumerate(web_sources, 1):
                    sources.append({
                        "reference_number": idx,
                        "filename": web_src.get("title", "Fuente Web"),
                        "content": web_src.get("snippet", ""),
                        "snippet": web_src.get("snippet", "")[:150] + "...",
                        "source": web_src.get("url", ""),
                        "title": web_src.get("title", "Información web"),
                        "type": "web",
                        "url": web_src.get("url", "")
                    })

            # Segundo: Documentos legales (agrupados)
            docs_by_file = {}
            for doc in reranked_docs:
                filename = doc.metadata.get('filename', 'Documento')
                if filename not in docs_by_file:
                    docs_by_file[filename] = {
                        "docs": [],
                        "metadata": doc.metadata
                    }
                docs_by_file[filename]["docs"].append(doc)

            web_count = len(sources)  # Cuántas fuentes web ya tenemos

            for filename, file_data in docs_by_file.items():
                # Combinar fragmentos
                combined_content = "\n\n---\n\n".join([
                    doc.page_content[:800].strip()
                    for doc in file_data["docs"]
                ])

                hybrid_context += f"--- DOCUMENTO LEGAL: {filename} ---\n"
                hybrid_context += combined_content[:500] + "...\n\n"  # Más breve

                sources.append({
                    "reference_number": web_count + len(sources) - web_count + 1,
                    "filename": filename,
                    "content": combined_content,
                    "snippet": file_data["docs"][0].page_content[:150] + "...",
                    "source": file_data["metadata"].get("source", "Base de datos legal"),
                    "type": "document"
                })

            # PASO 4: Generar respuesta BREVE con contexto híbrido
            is_follow_up = history and len(history) > 0

            # Crear lista de referencias
            refs_list = ", ".join([
                f"[{i}]" for i in range(1, len(sources) + 1)
            ])

            prompt = f"""🧠 ROL:
Eres un asistente jurídico especializado en Facilitadores Judiciales de Costa Rica.
Explicas temas legales en lenguaje simple, correcto y empático, orientado al ciudadano común.

🔴 REGLAS CRÍTICAS DE CITAS LEGALES (NO VIOLAR):

1. VIOLENCIA DOMÉSTICA → Ley N.° 7586 (NUNCA Ley 7654)
2. MENORES / PANI → Ley N.° 7739 Código de Niñez (NUNCA Ley 7654)
3. PENSIONES ALIMENTARIAS → Ley N.° 7654 (SOLO para pensiones)
4. CIVIL / DESALOJOS → Ley de Arrendamientos + Código Civil
5. LABORAL → Código de Trabajo (NUNCA Ley 7654)

⚠️ SI VES "VIOLENCIA" O "AGRESIÓN" → Ley 7586 (no 7654)
⚠️ SI VES "PANI" O "HIJO EN PELIGRO" → Ley 7739 (no 7654)
⚠️ NO INVENTAR artículos que no estén en el contexto

🧩 COMPORTAMIENTO:
• Usa tono empático, claro y educativo
• Cita SIEMPRE la ley correcta según la categoría
• NUNCA confundas funciones: los juzgados dictan medidas, la Dirección General de Adaptación Social SOLO ejecuta apremios corporales
• Si la consulta no tiene relación con Costa Rica, aclara amablemente que tu ámbito es el sistema jurídico costarricense
• Devuelve siempre una respuesta breve, precisa y con orientación práctica
• Usa SOLO estas referencias disponibles: {refs_list}

⚖️ INSTANCIAS Y SUS FUNCIONES (NO CONFUNDIR):

VIOLENCIA DOMÉSTICA:
• Juzgado de Violencia Doméstica (adscrito al Juzgado de Familia)
• OIJ (Organismo de Investigación Judicial)
• Fiscalía
• Facilitadores Judiciales
❌ NO: Juzgado de Pensiones Alimentarias

MENORES / PANI:
• PANI: Tel 1147 (línea gratuita 24/7)
• Juzgado de Familia
• Facilitadores Judiciales
❌ NO: Juzgado de Pensiones (solo si hay pensión involucrada)

PENSIONES ALIMENTARIAS:
• Juzgado de Pensiones Alimentarias
• Juzgado de Familia
• Facilitadores Judiciales
❌ NO: Dirección General de Adaptación Social (solo ejecuta, no recibe)

LABORAL:
• Ministerio de Trabajo: Tel 800-MTSS (800-6877)
• Juzgado de Trabajo
• Facilitadores Judiciales
❌ NO: Juzgado de Pensiones

📚 FUENTES LEGALES DISPONIBLES:
{hybrid_context}

❓ PREGUNTA DEL USUARIO: {question}

✍️ FORMATO DE RESPUESTA ESPERADO:

1. **Explicación breve del procedimiento** (2-3 líneas)
   - Qué puede hacer el usuario
   - Cuál es el proceso legal

2. **Dónde acudir** (1-2 líneas)
   - Institución específica y correcta según las reglas arriba
   - Departamento o juzgado correcto

3. **Cita legal** (1 línea)
   - Ley específica correcta según la categoría
   - Solo citar artículos si están en el contexto
   - Usar referencias: [1], [2], etc.

4. **Recomendación práctica** (1 línea)
   - Facilitador Judicial local
   - Orientación gratuita disponible

RESPUESTA:"""

            # Generar respuesta
            answer_raw = await self.llm.generate_async(prompt)
            answer = self.clean_answer(answer_raw)

            # PASO 4.5: Validar citas legales (opcional - para logging)
            validation = LegalVerificationHelper.validate_citation(answer)
            if validation["leyes_invalidas"]:
                logger.warning(f"⚠️ Leyes no verificadas en respuesta: {validation['leyes_invalidas']}")
            if validation["leyes_validas"]:
                logger.info(f"✅ Leyes válidas citadas: {validation['leyes_validas']}")

            # PASO 4.6: Validar concordancia categoría/cita - NUEVO (Optimización 2025-10-24)
            category_validation = LegalVerificationHelper.validate_category_citation_match(detected_category, answer)
            if not category_validation["is_valid"]:
                logger.error(f"❌ CITA INCORRECTA detectada:")
                for error in category_validation["errors"]:
                    logger.error(f"   • {error}")
                for correction in category_validation["corrections"]:
                    logger.info(f"   💡 {correction}")
                # Agregar nota de advertencia al final de la respuesta
                answer += "\n\n⚠️ Nota: Recomendamos verificar esta información con un Facilitador Judicial."
            else:
                if category_validation["cited_laws"]:
                    logger.info(f"✅ Citas correctas para categoría {detected_category}: {category_validation['cited_laws']}")

            # PASO 5: Formatear respuesta final con referencias
            final_answer = answer + "\n\n---\n\n**Referencias:**\n"

            for i, src in enumerate(sources, 1):
                if src["type"] == "web":
                    final_answer += f"[{i}] {src['title']} - {src['url']}\n"
                else:
                    final_answer += f"[{i}] {src['filename']} (Documento legal)\n"

            response = {
                "answer": final_answer,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "cached": False
            }

            self.cache.set(question, response)
            logger.info(f"✅ Respuesta híbrida generada en {response['processing_time']:.3f}s")
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
    """Endpoint principal para preguntas con respuestas híbridas optimizadas."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    # Convertir history de Message a dict si es necesario
    history_dicts = [msg.dict() if hasattr(msg, 'dict') else msg for msg in request.history]
    response = await bot.ask_async(request.question, history=history_dicts)

    # La respuesta ya viene formateada del nuevo flujo híbrido
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


# ============================================
# ENDPOINTS DE MODO ENTRENAMIENTO
# ============================================

from src.training_db import TrainingDatabase
from pydantic import BaseModel

# Instancia global de base de datos de entrenamiento
training_db = TrainingDatabase()


class FeedbackRequest(BaseModel):
    """Modelo para enviar feedback de una respuesta."""
    question: str
    answer: str
    sources: List[dict]
    category_detected: str
    processing_time: float
    status: str  # "approved", "rejected", "pending"
    evaluator_notes: str = ""
    feedback_items: List[dict] = []  # Lista de feedback detallado


class FeedbackItemRequest(BaseModel):
    """Modelo para agregar feedback individual."""
    evaluation_id: int
    feedback_type: str  # "citation_error", "category_error", "format_error", "content_error", "suggestion"
    field: str
    issue: str = ""
    correct_value: str = ""
    severity: str = "medium"  # "low", "medium", "high", "critical"


@app.post("/training/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Endpoint para guardar feedback de evaluación.
    Usado en modo entrenamiento para mejorar el sistema.
    """
    try:
        # Guardar evaluación principal
        evaluation_id = training_db.save_evaluation(
            question=request.question,
            answer=request.answer,
            sources=request.sources,
            category_detected=request.category_detected,
            processing_time=request.processing_time,
            status=request.status,
            evaluator_notes=request.evaluator_notes
        )

        # Guardar feedback items si existen
        for item in request.feedback_items:
            training_db.add_feedback(
                evaluation_id=evaluation_id,
                feedback_type=item.get("feedback_type", "suggestion"),
                field=item.get("field", "general"),
                issue=item.get("issue", ""),
                correct_value=item.get("correct_value", ""),
                severity=item.get("severity", "medium")
            )

        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "message": f"Feedback guardado exitosamente (Status: {request.status})"
        }

    except Exception as e:
        logger.error(f"❌ Error guardando feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/feedback-item")
async def add_feedback_item(request: FeedbackItemRequest):
    """Agrega un item de feedback a una evaluación existente."""
    try:
        feedback_id = training_db.add_feedback(
            evaluation_id=request.evaluation_id,
            feedback_type=request.feedback_type,
            field=request.field,
            issue=request.issue,
            correct_value=request.correct_value,
            severity=request.severity
        )

        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback agregado exitosamente"
        }

    except Exception as e:
        logger.error(f"❌ Error agregando feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/statistics")
async def get_training_statistics(days: int = 30):
    """Obtiene estadísticas del modo entrenamiento."""
    try:
        stats = training_db.get_statistics(days=days)
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/evaluations")
async def get_evaluations(status: str = "all", limit: int = 50):
    """Obtiene lista de evaluaciones filtradas por estado."""
    try:
        if status == "pending":
            evaluations = training_db.get_pending_evaluations(limit=limit)
        else:
            # Aquí podrías agregar más filtros si lo necesitas
            evaluations = training_db.get_pending_evaluations(limit=limit)

        return {
            "success": True,
            "evaluations": evaluations,
            "count": len(evaluations)
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo evaluaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/evaluation/{evaluation_id}")
async def get_evaluation_detail(evaluation_id: int):
    """Obtiene detalle completo de una evaluación con su feedback."""
    try:
        evaluation = training_db.get_evaluation(evaluation_id)
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluación no encontrada")

        feedback = training_db.get_feedback_for_evaluation(evaluation_id)

        return {
            "success": True,
            "evaluation": evaluation,
            "feedback": feedback
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error obteniendo evaluación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/export")
async def export_training_data(status: str = "approved"):
    """Exporta datos de entrenamiento a formato JSONL."""
    try:
        output_path = f"data/training_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        count = training_db.export_training_data(output_path, status=status)

        return {
            "success": True,
            "file_path": output_path,
            "records_exported": count,
            "message": f"Exportados {count} registros a {output_path}"
        }
    except Exception as e:
        logger.error(f"❌ Error exportando datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/training/evaluation/{evaluation_id}")
async def update_evaluation(evaluation_id: int, status: str, notes: str = ""):
    """Actualiza el estado de una evaluación."""
    try:
        training_db.update_evaluation_status(evaluation_id, status, notes)
        return {
            "success": True,
            "message": f"Evaluación {evaluation_id} actualizada a {status}"
        }
    except Exception as e:
        logger.error(f"❌ Error actualizando evaluación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
