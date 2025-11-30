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
from typing import Dict, Any, List, Optional, AsyncGenerator, Tuple, Set
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

CONTACT_TOKEN_REGEX = re.compile(r"[0-9][0-9A-Za-z\s()./-]{2,}")


def mask_contact_tokens(text: str, placeholder: str = "[dato de contacto no verificado]") -> str:
    """Reemplaza teléfonos u otros datos de contacto no verificados."""
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 4:
            return placeholder
        return token

    return CONTACT_TOKEN_REGEX.sub(_replace, text)


def extract_contact_digit_tokens(text: str) -> Set[str]:
    """Extrae secuencias numéricas (≥4 dígitos) para validar contactos."""
    allowed = set()
    if not text:
        return allowed

    for match in CONTACT_TOKEN_REGEX.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 4:
            allowed.add(digits)

    return allowed


def restrict_contacts_to_verified(text: str, allowed_digits: Set[str], placeholder: str) -> str:
    """Permite solo números presentes en allowed_digits, reemplaza el resto."""
    if not allowed_digits:
        return mask_contact_tokens(text, placeholder)

    def _replace(match: re.Match) -> str:
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 4 and digits not in allowed_digits:
            return placeholder
        return token

    return CONTACT_TOKEN_REGEX.sub(_replace, text)

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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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
    learned_from_feedback: bool = False  # Indica si la respuesta viene de corrección aprendida
    correction_type: str = ""  # Tipo de corrección si aplica
    similarity_score: float = 0.0  # Similitud con corrección aprendida
    matched_question: str = ""  # Pregunta original que matcheó
    correction_usage_id: int = 0  # ID para registrar feedback de la corrección usada
    correction_intent: str = ""


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

    # NOTA: Método eliminado - la información de contacto se aprende del modo entrenamiento
    # El sistema ya no usa respuestas hardcodeadas

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

            # Si no hay resultados filtrados, retornar vacío
            # NOTA: Información de contacto eliminada - se aprende del modo entrenamiento
            if not filtered_results:
                logger.warning(f"No se encontraron resultados de sitios costarricenses. Resultados originales: {[r.get('href') for r in results[:3]]}")
                return "", []

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
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
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
                    temperature=0.8,
                    max_tokens=2000,
                    top_p=0.95,
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
                        embedding_function=self.embedder,
                        collection_name="legal_documents"
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

        # NOTA: Las preguntas aclaratorias hardcodeadas fueron eliminadas
        # El sistema debe aprender a hacer preguntas desde las correcciones en modo entrenamiento
        # Solo mantenemos la detección de términos ambiguos básica
        AMBIGUOUS_TERMS = {
            "acoso": {"categories": ["laboral", "violencia", "penal", "menores"]},
            "denuncia": {"categories": ["penal", "violencia", "laboral", "civil"]},
            "pensión": {"categories": ["pension_alimentaria", "pension_vejez"]},
            "demanda": {"categories": ["laboral", "civil", "pension_alimentaria"]},
            "hijo": {"categories": ["pension_alimentaria", "menores", "civil"]},
            "despido": {"categories": ["laboral"]},
            "desalojo": {"categories": ["civil"]}
        }

        # Detectar términos ambiguos en la pregunta
        detected_ambiguities = []
        possible_categories = set()

        for term, config in AMBIGUOUS_TERMS.items():
            if term in question_lower:
                detected_ambiguities.append({
                    "term": term,
                    "questions": [],  # Sin preguntas hardcodeadas - se aprenden del entrenamiento
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

    def requires_verified_contact_lookup(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """
        Detecta si la pregunta (o su contexto reciente) exige datos de contacto
        o verificación explícita de instituciones. Si es así, se debe forzar
        la búsqueda web para evitar información inventada.
        """
        text_to_scan = question.lower()

        if history:
            try:
                user_messages = [
                    msg.get("content", "")
                    for msg in history
                    if isinstance(msg, dict) and msg.get("role") == "user"
                ]
                if user_messages:
                    last_context = " ".join(user_messages[-2:])
                    text_to_scan += f" {last_context.lower()}"
            except Exception:
                pass

        keyword_triggers = [
            "teléfono", "telefono", "número de teléfono", "numero de telefono",
            "contacto", "correo", "email", "whatsapp",
            "dirección", "direccion", "ubicación", "ubicacion",
            "sede", "oficina", "dónde queda", "donde queda",
            "horario", "abre", "cierran",
            "institución", "institucion", "instituciones",
            "qué institución", "que institucion", "nombre de la institución",
            "ministerio", "juzgado", "tribunal", "defensoría", "defensoria",
            "poder judicial", "inamu", "pani", "mtss", "ccss", "oij", "fiscalía", "fiscalia"
        ]

        for trigger in keyword_triggers:
            if trigger in text_to_scan:
                return True, trigger

        regex_triggers = [
            r"n[uú]mero\s+de\s+tel[eé]fono",
            r"c[uú]al\s+es\s+el\s+tel[eé]fono",
            r"c[uú]al\s+es\s+la\s+direcci[oó]n",
            r"c[uú]al\s+es\s+la\s+instituci[oó]n",
            r"d[oó]nde\s+queda\s+el\s+juzgado",
            r"c[oó]mo\s+contacto\s+al",
            r"nombre\s+del\s+ministerio"
        ]

        for pattern in regex_triggers:
            if re.search(pattern, text_to_scan):
                return True, pattern

        return False, ""

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
            force_contact_lookup, contact_lookup_reason = self.requires_verified_contact_lookup(question, history)

            # NOTA: Saludos, despedidas y preguntas sobre el bot se aprenden desde el modo entrenamiento
            # Las respuestas hardcodeadas fueron eliminadas para que el sistema aprenda de correcciones reales

            # ============================================
            # PASO 0: VERIFICAR CORRECCIONES APRENDIDAS (NUEVO - Aprendizaje en tiempo real)
            # ============================================
            # IMPORTANTE: Las correcciones NO se retornan directamente
            # Se usan como CONTEXTO DE APRENDIZAJE para que la IA genere respuestas basadas en ejemplos
            learned_correction = training_db.get_learned_correction(question)
            learned_context = None

            if learned_correction:
                similarity = learned_correction.get('similarity_score', 1.0)
                logger.info(f"🎓 Corrección aprendida encontrada: ID={learned_correction['id']}, Similitud={similarity:.3f}, Usado {learned_correction['times_used']} veces")

                # Preparar contexto de aprendizaje (NO retornar directamente)
                learned_context = {
                    "id": learned_correction['id'],
                    "question_text": learned_correction['question_text'],
                    "corrected_answer": learned_correction['corrected_answer'],
                    "correction_type": learned_correction['correction_type'],
                    "similarity_score": similarity,
                    "category": learned_correction.get('category', 'general'),
                    "intent": learned_correction.get('intent', 'general'),
                    "success_rate": learned_correction.get('success_rate', 100.0),
                    "effective_uses": learned_correction.get('effective_uses', 0.0)
                }

                logger.info(f"📚 Corrección se usará como EJEMPLO DE APRENDIZAJE, no como respuesta hardcodeada")

            # ============================================
            # PASO 0.5: DETECCIÓN DE AMBIGÜEDAD DESHABILITADA
            # ============================================
            # NOTA: Sistema deshabilitado - la IA debe responder directamente
            # incluso con preguntas cortas o aparentemente ambiguas.
            # Si necesita más detalles, lo puede preguntar al final de su respuesta.
            logger.info(f"ℹ️ Detección de ambigüedad deshabilitada - respondiendo directamente")

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
            
            # Si es conversación continua y pregunta corta, usar contexto para búsqueda
            search_query = question
            if history and len(history) > 0 and len(question.split()) < 5:
                # Pregunta corta en conversación -> probablemente es clarificación
                # Usar últimas preguntas del usuario para contexto de búsqueda
                user_questions = [msg['content'] for msg in history if msg['role'] == 'user']
                if user_questions:
                    last_user_question = user_questions[-1]
                    # Combinar última pregunta real con la actual
                    search_query = f"{last_user_question} {question}"
                    logger.info(f"🔍 Búsqueda contextualizada: '{question}' -> '{search_query}'")
            
            relevant_docs = await self.search_documents_async(search_query, k=3)
            reranked_docs_with_scores = self.rerank_documents(search_query, relevant_docs, top_k=2, return_scores=True)

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

            # PASO 2.5: Buscar en web si hace falta verificación adicional
            web_info = ""
            web_sources = []

            web_search_reasons = []
            if should_search_web:
                web_search_reasons.append("confianza baja en documentos")
            if detected_location:
                web_search_reasons.append(f"ubicación detectada: {detected_location}")
            if force_contact_lookup:
                lookup_reason = contact_lookup_reason or "solicitud explícita de contactos"
                web_search_reasons.append(f"petición de datos oficiales ({lookup_reason})")

            if should_search_web or detected_location or force_contact_lookup:
                logger.info("🌐 Activando búsqueda web debido a: " + " | ".join(web_search_reasons))
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

                sanitized_content = combined_content
                if force_contact_lookup:
                    sanitized_content = mask_contact_tokens(
                        combined_content,
                        "[dato de contacto no verificado - ignorar]"
                    )

                hybrid_context += f"--- DOCUMENTO LEGAL: {filename} ---\n"
                hybrid_context += sanitized_content[:500] + "...\n\n"  # Más breve

                snippet_preview = (
                    sanitized_content[:150] + "..."
                    if force_contact_lookup
                    else file_data["docs"][0].page_content[:150] + "..."
                )

                sources.append({
                    "reference_number": web_count + len(sources) - web_count + 1,
                    "filename": filename,
                    "content": sanitized_content if force_contact_lookup else combined_content,
                    "snippet": snippet_preview,
                    "source": file_data["metadata"].get("source", "Base de datos legal"),
                    "type": "document"
                })

            contact_guard_note = ""
            if force_contact_lookup:
                if web_info and web_sources:
                    contact_guard_note = """
📞 DATOS DE CONTACTO VERIFICADOS (CRÍTICO):
• El usuario pidió teléfonos, direcciones u oficinas oficiales
• Usá SOLO la información del bloque "INFORMACIÓN WEB ACTUALIZADA"
• Aclarales que los datos provienen de una fuente costarricense verificada
• NO inventes números adicionales ni nuevas instituciones
"""
                else:
                    contact_guard_note = """
📞 SIN DATOS OFICIALES DISPONIBLES:
• El usuario pidió teléfonos, direcciones u oficinas oficiales
• No se encontró información verificada en línea para compartir
• Decí explícitamente que no tenés un contacto confirmado en este momento
• NO inventes números ni nombres - sugerí acudir presencialmente al Poder Judicial o usar canales oficiales sin detallar números
"""

            # PASO 4: Generar respuesta BREVE con contexto híbrido
            is_follow_up = history and len(history) > 0
            logger.info(f"📝 Historial: {len(history) if history else 0} mensajes | is_follow_up: {is_follow_up}")

            # Crear lista de referencias
            refs_list = ", ".join([
                f"[{i}]" for i in range(1, len(sources) + 1)
            ])

            # Construir contexto de aprendizaje si hay correcciones
            learning_context = ""
            if learned_context:
                logger.info(f"🎓 Inyectando corrección aprendida en prompt como EJEMPLO DE APRENDIZAJE")
                learning_context = f"""
🎓 EJEMPLO DE RESPUESTA CORRECTA (ALTA PRIORIDAD):
Esta pregunta es MUY SIMILAR a una que ya se respondió correctamente antes.
Usá esta respuesta como MODELO PRINCIPAL - es exactamente el estilo, tono y nivel de detalle que deberías seguir.

📝 Pregunta similar que se hizo antes:
"{learned_context['question_text']}"

✅ Respuesta que funcionó muy bien (SEGUÍ ESTE ESTILO):
{learned_context['corrected_answer']}

⚡ IMPORTANTE - Instrucciones de cómo usar este ejemplo:
• Este es el MEJOR EJEMPLO de cómo responder este tipo de pregunta
• COPIÁ el ESTILO conversacional, el uso de emojis, y el tono cercano
• SEGUÍ la ESTRUCTURA: cómo introduce el tema, cómo explica los pasos, cómo cierra
• MANTENÉ el mismo NIVEL DE DETALLE (ni más formal, ni más técnico)
• Si la pregunta es casi idéntica, tu respuesta debe ser muy similar
• Si la pregunta varía un poco, ADAPTÁ el contenido pero mantené el mismo estilo
• Este ejemplo tiene PRIORIDAD sobre el contexto de las fuentes legales
• Categoría: {learned_context['category']} | Tipo: {learned_context['correction_type']}
• Similitud con pregunta actual: {learned_context['similarity_score']:.1%}

"""

            # Construir contexto conversacional si existe historial
            conversation_context = ""
            if is_follow_up:
                logger.info(f"💬 Conversación continua detectada: {len(history)} mensajes previos")

                # Tomar últimos 4 mensajes (2 intercambios completos) con contenido completo
                recent_history = history[-4:] if len(history) > 4 else history

                # Detectar si es una pregunta de clarificación/seguimiento
                user_last_message = question.lower().strip()
                is_clarification = any(keyword in user_last_message for keyword in [
                    'sí', 'si', 'como', 'cómo', 'explica', 'explicá', 'detalle', 'más',
                    'dime', 'cuéntame', 'y eso', 'qué es', 'que es', 'continua', 'continuá',
                    'sigue', 'seguí', 'entonces', 'ok', 'vale', 'entiendo'
                ])

                conversation_context = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                conversation_context += "💬 CONTEXTO DE CONVERSACIÓN CONTINUA\n"
                conversation_context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                if is_clarification:
                    conversation_context += "⚠️ ⚠️ ⚠️ TIPO DE MENSAJE: PREGUNTA DE SEGUIMIENTO/CLARIFICACIÓN ⚠️ ⚠️ ⚠️\n\n"
                    conversation_context += "🚨 INSTRUCCIÓN CRÍTICA MÁXIMA PRIORIDAD:\n"
                    conversation_context += "El usuario/a está pidiendo que PROFUNDICES en algo que YA MENCIONASTE en tu respuesta anterior.\n"
                    conversation_context += "Esta NO es una pregunta nueva. Es una ACLARACIÓN de tu respuesta previa.\n\n"
                    conversation_context += "🚫 🚫 🚫 PROHIBICIONES ABSOLUTAS:\n"
                    conversation_context += "• NO uses información de las 'FUENTES LEGALES' de abajo\n"
                    conversation_context += "• NO cambies de tema\n"
                    conversation_context += "• NO repitas toda la información anterior\n"
                    conversation_context += "• NO empieces desde cero\n\n"
                    conversation_context += "✅ ✅ ✅ OBLIGACIONES:\n"
                    conversation_context += "• BASA tu respuesta EXCLUSIVAMENTE en el HISTORIAL DE LA CONVERSACIÓN de abajo\n"
                    conversation_context += "• Identifica QUÉ TEMA ESPECÍFICO de tu respuesta anterior está preguntando\n"
                    conversation_context += "• Profundiza SOLO en ese aspecto concreto\n"
                    conversation_context += "• Usa frases como: 'Dale, sobre ese punto...', 'Perfecto, te explico...', 'Claro, mirá...'\n"
                    conversation_context += "• Sé MUCHO más específico y detallado que en tu respuesta anterior\n\n"
                else:
                    conversation_context += "ℹ️ TIPO DE MENSAJE: NUEVA CONSULTA EN CONVERSACIÓN EXISTENTE\n"
                    conversation_context += "El usuario/a hace una pregunta nueva pero mantén coherencia con lo anterior.\n\n"

                conversation_context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                conversation_context += "📚 HISTORIAL COMPLETO DE LA CONVERSACIÓN:\n"
                conversation_context += "(Lee todo el contexto - ESTA ES TU FUENTE PRINCIPAL)\n"
                conversation_context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                for i, msg in enumerate(recent_history, 1):
                    role_label = "👤 Usuario/a" if msg['role'] == 'user' else "⚖️ Tu respuesta anterior"
                    # NO truncar - incluir contenido completo para mejor contexto
                    conversation_context += f"{'─' * 60}\n"
                    conversation_context += f"{role_label} (Mensaje {i}):\n"
                    conversation_context += f"{msg['content']}\n"

                conversation_context += f"{'─' * 60}\n\n"

                conversation_context += "🎯 INSTRUCCIONES CRÍTICAS DE CONTINUIDAD:\n\n"

                if is_clarification:
                    conversation_context += "📌 ESTA ES UNA PREGUNTA DE CLARIFICACIÓN:\n"
                    conversation_context += "1. ❌ NO repitas los pasos o información que ya diste en tu respuesta anterior\n"
                    conversation_context += "2. ❌ NO empieces desde cero explicando todo de nuevo\n"
                    conversation_context += "3. ✅ SÍ identifica QUÉ ESPECÍFICAMENTE está preguntando el usuario/a\n"
                    conversation_context += "4. ✅ SÍ profundiza SOLO en ese punto concreto con más detalles\n"
                    conversation_context += "5. ✅ SÍ usa frases como: 'Perfecto, te explico ese punto...', 'Dale, sobre eso...', 'Claro, mirá...'\n"
                    conversation_context += "6. ✅ SÍ asume que el usuario/a ya leyó y entendió lo anterior\n"
                    conversation_context += "7. ✅ SÍ sé más específico y práctico, con ejemplos concretos si es posible\n\n"
                    conversation_context += "🔍 ANÁLISIS REQUERIDO:\n"
                    conversation_context += "Antes de responder, identifica:\n"
                    conversation_context += "• ¿Sobre qué TEMA ESPECÍFICO de tu respuesta anterior está preguntando?\n"
                    conversation_context += "• ¿Qué DETALLE o PASO necesita que amplíes?\n"
                    conversation_context += "• ¿Qué NO necesitas repetir porque ya lo dijiste?\n\n"
                else:
                    conversation_context += "1. ✅ Mantén coherencia con toda la conversación previa\n"
                    conversation_context += "2. ✅ Reconoce cualquier información que el usuario/a ya te dio\n"
                    conversation_context += "3. ✅ NO pidas datos que el usuario/a ya mencionó\n"
                    conversation_context += "4. ✅ Haz referencias naturales: 'Como te mencioné...', 'Siguiendo con lo que hablamos...'\n"
                    conversation_context += "5. ✅ Si cambia de tema, hazlo natural: 'Perfecto, ahora sobre tu nueva consulta...'\n\n"

                conversation_context += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            # Detectar si es clarificación para reordenar el prompt
            is_clarification_detected = is_follow_up and any(keyword in question.lower().strip() for keyword in [
                'sí', 'si', 'como', 'cómo', 'explica', 'explicá', 'detalle', 'más',
                'dime', 'cuéntame', 'y eso', 'qué es', 'que es', 'continua', 'continuá',
                'sigue', 'seguí', 'entonces', 'ok', 'vale', 'entiendo'
            ])

            # Para clarificaciones: SOLO contexto conversacional (NO usar fuentes legales)
            # Para preguntas normales: orden estándar
            if is_clarification_detected:
                logger.info("🎯 CLARIFICACIÓN DETECTADA - Usando SOLO historial conversacional")
                sources_section = ""  # NO incluir fuentes legales en clarificaciones
                context_section = conversation_context
            else:
                sources_section = f"""
📚 FUENTES LEGALES Y CONTEXTO:
Estas fuentes contienen información oficial verificada del sistema jurídico costarricense.
Basa tu respuesta en este contexto. NO inventes información.

{hybrid_context}

"""
                context_section = conversation_context if conversation_context else ""

            institution_policy_block = """
🏛️ INSTITUCIONES Y DATOS OFICIALES (CRÍTICO):
• Mencioná SOLO instituciones costarricenses reales
• Deben aparecer en el bloque de fuentes legales, en la información web o en la lista del ÁMBITO GEOGRÁFICO
• Si no tenés certeza del nombre oficial, decí que no contás con ese dato verificado
• Teléfonos, correos o direcciones deben salir del bloque "INFORMACIÓN WEB ACTUALIZADA" o de los documentos
• Si no hay datos verificados, dejalo en claro y evitá inventar información
"""

            prompt = f"""🧠 ROL Y PERSONALIDAD:
Sos un asistente jurídico especializado en Facilitadores Judiciales de Costa Rica, hablando con lenguaje claro, cercano y conversacional.
Explicás temas legales como si estuvieras conversando frente a frente con alguien que necesita ayuda ☕.
Tu objetivo es ser PRÁCTICO, DIRECTO y EMPÁTICO - el usuario necesita ayuda concreta, no un manual de derecho.

🇨🇷 ÁMBITO GEOGRÁFICO Y LEGAL (CRÍTICO - MÁXIMA PRIORIDAD):
• Este sistema es EXCLUSIVAMENTE para COSTA RICA 🇨🇷
• SOLO mencioná instituciones, leyes, y procedimientos de COSTA RICA
• Si no tenés información específica de Costa Rica, decilo claramente
• NUNCA inventes o asumas que leyes de otros países aplican en Costa Rica
• EJEMPLOS DE INSTITUCIONES COSTARRICENSES VÁLIDAS:
  ✅ Juzgados de Costa Rica (Violencia Doméstica, Familia, Trabajo, etc.)
  ✅ Ministerio de Trabajo y Seguridad Social (MTSS)
  ✅ Instituto Nacional de las Mujeres (INAMU)
  ✅ Poder Judicial de Costa Rica
  ✅ Caja Costarricense de Seguro Social (CCSS)
  ✅ Defensoría de los Habitantes
  ✅ Defensa Pública
• ❌ NO menciones instituciones de otros países (México, España, Argentina, etc.)
• ❌ NO cites leyes que no sean de Costa Rica
• Si la base de conocimiento no tiene información específica de Costa Rica sobre el tema, decilo honestamente

🌈 LENGUAJE INCLUSIVO (OBLIGATORIO):
• SIEMPRE usa lenguaje inclusivo: "juez o jueza", "trabajador o trabajadora", "el usuario o la usuaria"
• Alterna formas inclusivas naturalmente: "persona trabajadora", "persona profesional en derecho"
• NUNCA uses solo masculino como genérico
{institution_policy_block}
{learning_context}{sources_section}{contact_guard_note}{context_section}
🎯 ESTILO DE RESPUESTA (MUY IMPORTANTE):
• Hablá de forma natural y conversacional - usá "vos", "podés", "te explico"
• Sé DIRECTO y CONCISO - máximo 350-400 palabras por respuesta
• El usuario quiere saber QUÉ HACER, no teoría jurídica extensa
• Usá emojis relevantes para hacer el texto más amigable (⚖️ 📩 💡 ✅ ⏳ etc.)
• Dividí la información en 3-5 pasos principales, no más
• Explicá paso a paso lo que debe hacer la persona, pero sin exceso de detalles
• Priorizá información PRÁCTICA sobre tecnicismos legales
• Si mencionás leyes, hacelo de forma simple e integrada en el texto natural
• Usá viñetas o listas numeradas para que sea fácil de leer
• Terminá ofreciendo ayuda adicional: "¿Querés que te explique más sobre...?"

📝 FORMATO DE TEXTO (CRÍTICO - SEGUIR SIEMPRE):
• Para títulos principales: ## Título (solo al inicio)
• Para subtítulos dentro del texto: **Subtítulo en negrita:**
• EJEMPLOS CORRECTOS:
  ✅ **Qué necesitás presentar:**
  ✅ **Pasos prácticos:**
  ✅ **Dónde acudir:**
• EJEMPLOS INCORRECTOS (NUNCA USAR):
  ❌ ### Pasos prácticos
  ❌ #### Qué necesitás
• NUNCA uses ### o #### - se ven MAL en la interfaz
• NO uses separadores como "---" o "___"
• Usá saltos de línea vacíos para separar secciones
• Las listas numeradas (1️⃣ 2️⃣) y viñetas (•) funcionan perfecto

🚫 EVITÁ (CRÍTICO):
• Respuestas largas y formales (máximo 400-500 palabras)
• Estructura rígida de "procedimiento", "dónde acudir", "base legal"
• Exceso de tecnicismos o referencias legales
• Tono impersonal o distante
• Separadores horizontales "---" (se ven mal)
• Encabezados con ### o #### (usar **negrita:** en su lugar)
• NO agregues sección de "Referencias:" al final
• NO listes fuentes numeradas como "[1] Documento legal" al final
• NO agregues notas como "⚠️ Nota: Recomendamos verificar..." al final
• La respuesta debe terminar con tu último consejo o pregunta de seguimiento

✅ HACÉ:
• Empatizá con la situación de la persona
• Explicá los pasos concretos que debe seguir
• Mencioná instituciones específicas donde puede ir
• Agregá consejos prácticos basados en el contexto
• Si necesitás mencionar una fuente legal, integrala naturalmente en el texto
• Mantené la conversación natural y fluida
• Terminá con algo útil para el usuario, NO con referencias o notas

❓ PREGUNTA DEL USUARIO: {question}

💬 RESPONDÉ como si estuvieras hablando con un amigo o amiga que necesita orientación legal. Sé claro, práctico y cercano.

RESPUESTA:"""

            # Generar respuesta
            answer_raw = await self.llm.generate_async(prompt)
            answer = self.clean_answer(answer_raw)

            if force_contact_lookup:
                if web_info and web_sources:
                    verified_digits: Set[str] = set()
                    verified_chunks = [web_info] + [
                        f"{src.get('title', '')} {src.get('snippet', '')}"
                        for src in web_sources
                    ]
                    for chunk in verified_chunks:
                        verified_digits |= extract_contact_digit_tokens(chunk)

                    answer = restrict_contacts_to_verified(
                        answer,
                        verified_digits,
                        "[dato de contacto no verificado]"
                    )
                else:
                    answer = mask_contact_tokens(
                        answer,
                        "[no tengo un número oficial verificado en este momento]"
                    )

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
                # NO agregar nota al final - el prompt ya indica que la IA debe ser conversacional
            else:
                if category_validation["cited_laws"]:
                    logger.info(f"✅ Citas correctas para categoría {detected_category}: {category_validation['cited_laws']}")

            # PASO 5: Respuesta final sin referencias adicionales
            final_answer = answer

            response = {
                "answer": final_answer,
                "sources": sources,
                "processing_time": time.time() - start_time,
                "cached": False
            }

            # Si se usó corrección aprendida, agregar metadata y marcar como usada
            if learned_context:
                response["learned_from_feedback"] = True
                response["correction_type"] = learned_context['correction_type']
                response["similarity_score"] = learned_context['similarity_score']
                response["matched_question"] = learned_context['question_text']
                response["correction_intent"] = learned_context.get("intent", "general")

                # Marcar que se usó la corrección
                usage_id = training_db.mark_correction_used(learned_context['id'], question)
                response["correction_usage_id"] = usage_id
                logger.info(
                    f"📊 Corrección {learned_context['id']} usada "
                    f"(similitud: {learned_context['similarity_score']:.1%}, usage_id={usage_id})"
                )

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
        "system_status": "optimal"
        # NOTA: precomputed_responses eliminado - el sistema aprende desde modo entrenamiento
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


# ============================================
# ENDPOINTS PARA APRENDIZAJE EN TIEMPO REAL
# ============================================

class CorrectionRequest(BaseModel):
    """Modelo para guardar una corrección aprendida."""
    question: str
    original_answer: str
    corrected_answer: str
    correction_type: str  # "citation", "category", "content", "format"
    category: str = "general"


class CorrectionUsageFeedback(BaseModel):
    """Feedback para confirmar si la corrección ayudó o no."""
    result: str  # "success" o "fail"
    source: str = "explicit"


@app.post("/training/learn-correction")
async def learn_correction(request: CorrectionRequest):
    """
    Guarda una corrección que se aplicará inmediatamente en futuras consultas idénticas.
    APRENDIZAJE EN TIEMPO REAL.
    """
    try:
        correction_id = training_db.save_learned_correction(
            question=request.question,
            original_answer=request.original_answer,
            corrected_answer=request.corrected_answer,
            correction_type=request.correction_type,
            category=request.category
        )

        logger.info(f"🎓 APRENDIZAJE EN TIEMPO REAL: Corrección guardada ID={correction_id}")

        return {
            "success": True,
            "correction_id": correction_id,
            "message": "Corrección guardada. Se aplicará inmediatamente en futuras consultas.",
            "learned_from": request.question[:50] + "..."
        }

    except Exception as e:
        logger.error(f"❌ Error guardando corrección: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/corrections")
async def get_corrections(category: Optional[str] = None, limit: int = 100):
    """Obtiene todas las correcciones aprendidas."""
    try:
        corrections = training_db.get_all_corrections(category=category, limit=limit)

        return {
            "success": True,
            "corrections": corrections,
            "count": len(corrections),
            "category_filter": category or "all"
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo correcciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/correction-usage/{usage_id}")
async def submit_correction_usage(usage_id: int, payload: CorrectionUsageFeedback):
    """Permite registrar feedback explícito sobre una corrección aplicada."""
    try:
        training_db.finalize_correction_usage(usage_id, payload.result, payload.source)
        return {
            "success": True,
            "usage_id": usage_id,
            "result": payload.result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error registrando feedback de corrección: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/correction-stats")
async def get_correction_stats():
    """Obtiene estadísticas de las correcciones aprendidas."""
    try:
        all_corrections = training_db.get_all_corrections(limit=1000)

        total_corrections = len(all_corrections)
        total_uses = sum(c["times_used"] for c in all_corrections)

        # Agrupar por categoría
        by_category = {}
        for correction in all_corrections:
            cat = correction["category"]
            if cat not in by_category:
                by_category[cat] = {"count": 0, "uses": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["uses"] += correction["times_used"]

        # Agrupar por tipo de corrección
        by_type = {}
        for correction in all_corrections:
            ctype = correction["correction_type"]
            if ctype not in by_type:
                by_type[ctype] = {"count": 0, "uses": 0}
            by_type[ctype]["count"] += 1
            by_type[ctype]["uses"] += correction["times_used"]

        # Top 10 correcciones más usadas
        top_used = sorted(all_corrections, key=lambda x: x["times_used"], reverse=True)[:10]

        return {
            "success": True,
            "statistics": {
                "total_corrections": total_corrections,
                "total_times_used": total_uses,
                "by_category": by_category,
                "by_type": by_type,
                "top_used_corrections": [
                    {
                        "question": c["question_text"][:100],
                        "times_used": c["times_used"],
                        "category": c["category"],
                        "type": c["correction_type"]
                    }
                    for c in top_used
                ]
            }
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas de correcciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ENDPOINT PARA SUBIR DOCUMENTOS A CHROMADB
# ============================================

from fastapi import UploadFile, File
import PyPDF2
import io

@app.post("/training/upload-document")
async def upload_document(file: UploadFile = File(...), category: str = "general"):
    """
    Sube un documento (PDF o TXT) y lo agrega a la base vectorial ChromaDB.
    SOLO accesible desde modo entrenamiento.
    """
    try:
        # Validar tipo de archivo
        allowed_extensions = ['.pdf', '.txt', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}"
            )

        # Leer contenido del archivo
        content_bytes = await file.read()

        # Extraer texto según tipo de archivo
        if file_ext == '.pdf':
            # Procesar PDF
            pdf_file = io.BytesIO(content_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += f"\n--- Página {page_num + 1} ---\n{page_text}"

        elif file_ext in ['.txt', '.md']:
            # Procesar texto plano
            text_content = content_bytes.decode('utf-8', errors='ignore')

        # Validar que se extrajo contenido
        if not text_content or len(text_content.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="El documento no contiene suficiente texto válido"
            )

        # Dividir en chunks para vectorización
        chunk_size = 1000
        overlap = 200
        chunks = []

        for i in range(0, len(text_content), chunk_size - overlap):
            chunk = text_content[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Solo chunks con contenido significativo
                chunks.append(chunk)

        logger.info(f"📄 Documento procesado: {file.filename} - {len(chunks)} chunks")

        # Crear documentos de LangChain
        from langchain_core.documents import Document

        documents = []
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": f"uploaded/{file.filename}",
                    "filename": file.filename,
                    "category": category,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "upload_date": datetime.now().isoformat()
                }
            )
            documents.append(doc)

        # Agregar a ChromaDB de forma asíncrona
        loop = asyncio.get_event_loop()

        def add_to_vectordb():
            if bot.vectordb:
                bot.vectordb.add_documents(documents)
                return True
            return False

        success = await loop.run_in_executor(bot.executor, add_to_vectordb)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Base de datos vectorial no disponible"
            )

        # Obtener nuevo conteo de documentos
        new_count = await loop.run_in_executor(
            bot.executor,
            lambda: bot.vectordb._collection.count()
        )

        logger.info(f"✅ Documento agregado a ChromaDB: {file.filename} ({len(chunks)} chunks)")

        return {
            "success": True,
            "message": f"Documento '{file.filename}' agregado exitosamente",
            "filename": file.filename,
            "chunks_added": len(chunks),
            "category": category,
            "total_documents_in_db": new_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error procesando documento: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")


@app.get("/training/document-stats")
async def get_document_stats():
    """Obtiene estadísticas de documentos en ChromaDB."""
    try:
        if not bot.vectordb:
            return {
                "success": False,
                "message": "Base de datos vectorial no disponible"
            }

        loop = asyncio.get_event_loop()

        # Obtener conteo total
        total_docs = await loop.run_in_executor(
            bot.executor,
            lambda: bot.vectordb._collection.count()
        )

        # Obtener TODOS los documentos (incluyendo los importados originalmente)
        def get_all_docs():
            try:
                # Obtener documentos con IDs y contenido
                results = bot.vectordb._collection.get(
                    limit=10000,  # Límite aumentado para obtener todos los chunks
                    include=["metadatas", "documents"]
                )
                return results
            except Exception as e:
                logger.error(f"Error obteniendo documentos: {e}")
                return None

        all_results = await loop.run_in_executor(bot.executor, get_all_docs)

        all_files = {}
        
        logger.info(f"🔍 DEBUG: all_results type: {type(all_results)}")
        logger.info(f"🔍 DEBUG: all_results keys: {all_results.keys() if all_results else 'None'}")
        
        if all_results:
            ids = all_results.get('ids', [])
            documents = all_results.get('documents', [])
            metadatas = all_results.get('metadatas', [])
            
            logger.info(f"📊 Procesando {len(ids)} chunks de documentos")
            
            for i in range(len(ids)):
                doc_id = ids[i] if i < len(ids) else f'doc_{i}'
                doc_content = documents[i] if i < len(documents) else ''
                metadata = metadatas[i] if i < len(metadatas) else None
                
                # Intentar obtener el nombre del archivo de diferentes campos
                filename = None
                
                if metadata:
                    filename = (
                        metadata.get('filename') or 
                        metadata.get('source', '').split('/')[-1] or 
                        metadata.get('title')
                    )
                
                # Si no hay filename en metadata, agrupar por patrón de título en el contenido
                if not filename:
                    # Extraer título del contenido
                    lines = doc_content.split('\n')[:3]  # Primeras 3 líneas
                    title_line = ''
                    for line in lines:
                        clean_line = line.strip()
                        if len(clean_line) > 10:  # Línea significativa
                            title_line = clean_line[:80]
                            break
                    
                    # Usar título como agrupador
                    if title_line:
                        filename = title_line
                    else:
                        filename = f"Documento {doc_id}"
                
                # Crear entrada para el archivo si no existe
                if filename not in all_files:
                    display_title = filename
                    if len(filename) > 100:
                        display_title = filename[:97] + '...'
                    
                    all_files[filename] = {
                            "filename": filename,
                        "display_name": display_title,
                        "category": metadata.get('category', 'documentos_legales') if metadata else 'documentos_legales',
                        "upload_date": 'Base de datos original',
                        "chunks": 0,
                        "source": metadata.get('source', filename) if metadata else filename
                    }
                
                all_files[filename]["chunks"] += 1

        logger.info(f"✅ Agrupados en {len(all_files)} archivos únicos")

        return {
            "success": True,
            "total_documents": total_docs,
            "uploaded_files": list(all_files.values()),
            "uploaded_files_count": len(all_files)
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas de documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/document-content/{filename}")
async def get_document_content(filename: str):
    """Obtiene todos los chunks de un documento específico para revisión."""
    try:
        if not bot.vectordb:
            return {
                "success": False,
                "message": "Base de datos vectorial no disponible"
            }

        loop = asyncio.get_event_loop()

        def get_document_chunks():
            try:
                # Intentar buscar por filename primero
                results = bot.vectordb._collection.get(
                    where={"filename": filename},
                    include=["documents", "metadatas"]
                )
                
                # Si no encuentra nada, buscar por source que termine con el filename
                if not results or not results.get('documents'):
                    # Obtener todos y filtrar manualmente
                    all_results = bot.vectordb._collection.get(
                        limit=10000,
                        include=["documents", "metadatas"]
                    )
                    
                    filtered_docs = []
                    filtered_metas = []
                    filtered_ids = []
                    
                    if all_results and 'metadatas' in all_results:
                        for i, meta in enumerate(all_results['metadatas']):
                            if meta:
                                doc_name = (
                                    meta.get('filename') or 
                                    meta.get('source', '').split('/')[-1]
                                )
                                if doc_name == filename or meta.get('source', '').endswith(filename):
                                    filtered_docs.append(all_results['documents'][i])
                                    filtered_metas.append(meta)
                                    filtered_ids.append(all_results['ids'][i])
                    
                    results = {
                        'documents': filtered_docs,
                        'metadatas': filtered_metas,
                        'ids': filtered_ids
                    }
                
                return results
            except Exception as e:
                logger.error(f"Error obteniendo chunks: {e}")
                return None

        results = await loop.run_in_executor(bot.executor, get_document_chunks)

        if not results or 'documents' not in results:
            return {
                "success": False,
                "message": f"No se encontraron chunks para el documento: {filename}"
            }

        # Organizar chunks por índice
        chunks_data = []
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        ids = results.get('ids', [])

        for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
            chunk_info = {
                "id": doc_id,
                "content": doc,
                "chunk_index": meta.get('chunk_index', i),
                "total_chunks": meta.get('total_chunks', len(documents)),
                "category": meta.get('category', 'unknown'),
                "upload_date": meta.get('upload_date', 'unknown'),
                "source": meta.get('source', '')
            }
            chunks_data.append(chunk_info)

        # Ordenar por índice de chunk
        chunks_data.sort(key=lambda x: x['chunk_index'])

        return {
            "success": True,
            "filename": filename,
            "total_chunks": len(chunks_data),
            "chunks": chunks_data,
            "category": chunks_data[0]['category'] if chunks_data else 'unknown',
            "upload_date": chunks_data[0]['upload_date'] if chunks_data else 'unknown'
        }

    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas de documentos: {e}")
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
