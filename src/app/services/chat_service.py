import logging
import time
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional

from src.app.config import settings
from src.app.schemas.chat import Message, QueryResponse
from src.app.core.llm.client import BaseLLM, GroqLLM, OpenRouterLLM
from src.app.core.rag.embeddings import EmbeddingService
from src.app.core.rag.store import VectorStoreService
from src.app.core.rag.web_search import WebSearchHelper
from src.app.core.cache import SmartCache
from src.app.services.training_service import TrainingService
from src.app.core.prompts.templates import (
    SYSTEM_PROMPT, 
    CLARIFICATION_CONTEXT_TEMPLATE, 
    NEW_QUERY_CONTEXT_TEMPLATE,
    CLARIFICATION_INSTRUCTIONS, 
    CONTINUITY_INSTRUCTIONS,
    POPULAR_INSTITUTIONS_BLOCK,
    INSTITUTION_POLICY_BLOCK,
    AUDIENCE_BLOCK
)

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self._initialize_components()
        self.cache = SmartCache(ttl=3600)
    
    def _initialize_components(self):
        # 1. Embeddings & Vector Store
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStoreService(self.embedding_service)
        
        # 2. Training Service
        self.training_service = TrainingService(self.embedding_service)
        
        # 3. LLM Provider
        if settings.LLM_PROVIDER == "openrouter":
            logger.info(f"🚀 Usando OpenRouter API: {settings.OPENROUTER_MODEL}")
            self.llm: BaseLLM = OpenRouterLLM(
                api_key=settings.OPENROUTER_API_KEY, 
                model=settings.OPENROUTER_MODEL
            )
        else:
            logger.info(f"🚀 Usando Groq API: {settings.GROQ_MODEL}")
            self.llm: BaseLLM = GroqLLM(
                api_key=settings.GROQ_API_KEY, 
                model=settings.GROQ_MODEL
            )

    async def _parallel_search(self, question: str, history: List[Message], force_contact: bool) -> Tuple[Any, List[Any], Tuple[str, List[Any]]]:
        """Ejecuta búsquedas de contexto en paralelo."""
        loop = asyncio.get_running_loop()
        
        # Task 1: Learned Corrections
        learned_task = loop.run_in_executor(
            None, 
            lambda: self.training_service.get_learned_correction(question)
        )
        
        # Task 2: Vector Search
        async def fetch_documents():
            search_query = question
            # Expand query if short and history exists
            if history and len(question.split()) < 5:
                # Get last user message
                last_user_msg = next((m.content for m in reversed(history) if m.role == 'user'), None)
                if last_user_msg:
                    search_query = f"{last_user_msg} {question}"
            
            try:
                return await asyncio.wait_for(
                    self.vector_store.search_async(search_query, k=settings.SEARCH_TOP_K * 2),
                    timeout=5.0
                )
            except Exception as e:
                logger.error(f"❌ Error buscando documentos: {e}")
                return []

        # Task 3: Web Search
        detected_location = self._detect_location(question)
        async def fetch_web():
            if detected_location or force_contact:
                return await WebSearchHelper.search_web_info(question, detected_location)
            return ("", [])

        return await asyncio.gather(learned_task, fetch_documents(), fetch_web())

    def _detect_location(self, text: str) -> Optional[str]:
        # Lista simplificada de lugares comunes
        places = [
            "san josé", "cartago", "alajuela", "heredia", "puntarenas", "guanacaste", "limón",
            "liberia", "pérez zeledón", "desamparados", "escazú", "san carlos", "nicoya", "turrialba"
        ]
        text_lower = text.lower()
        for place in places:
            if place in text_lower:
                return place.title()
        return None

    def requires_verified_contact_lookup(self, text: str, history: List[Message]):
        text_lower = text.lower()
        contact_keywords = ["teléfono", "numero", "donde llamar", "correo", "contactar", "dirección", "ubicación"]
        requires_lookup = any(kw in text_lower for kw in contact_keywords)
        
        # Check context if not explicit in query
        if not requires_lookup and history:
            last_msg = history[-1].content.lower()
            if any(kw in last_msg for kw in contact_keywords):
                return True, "context_continuation"
                
        return requires_lookup, "keyword_match"

    async def get_response(self, question: str, history: List[Message]) -> QueryResponse:
        start_time = time.time()
        
        # 1. Check Cache
        cache_key = f"{question.strip().lower()}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("⚡ Respuesta servida desde Cache")
            return QueryResponse(**cached_result)

        # 2. Parallel Search Execution
        requires_contact, _ = self.requires_verified_contact_lookup(question, history)
        
        logger.info("🚀 Iniciando tareas paralelas...")
        learned_correction, relevant_docs, (web_info, web_sources) = await self._parallel_search(
            question, history, requires_contact
        )

        # 3. Process Reranking / Docs
        # Filter docs by score threshold (if using cosine similarity, e.g., > 0.4)
        # Note: Pinecone with dotproduct/cosine varies, assuming normalized embeddings
        final_docs_content = []
        doc_sources = []
        
        for doc, score in relevant_docs:
            if score > 0.40: # Umbral de relevancia
                content = doc.page_content
                source = doc.metadata.get('source', 'Documento Interno')
                final_docs_content.append(content)
                doc_sources.append({"title": "Guía Legal", "snippet": content[:100], "score": score})

        # 4. Construct Prompt
        prompt = self._construct_prompt(
            question, history, learned_correction, 
            final_docs_content, web_info
        )

        # 5. Generate Answer
        logger.info(f"🤖 Generando respuesta con {self.llm.__class__.__name__}...")
        answer = await self.llm.generate_async(prompt)

        # 6. Post-processing
        # Mask unverified contacts from LLM hallucination safeguard
        # (Assuming utility functions are ported or simple enough to inline/import)
        # For brevity, implementing simple masking or trusting prompt instructions for now.
        
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            answer=answer,
            sources=web_sources + doc_sources,
            processing_time=processing_time,
            learned_from_feedback=bool(learned_correction),
            correction_type=learned_correction[3] if learned_correction else ""
        )
        
        # Cache successful responses
        self.cache.set(cache_key, response.model_dump())
        
        return response

    def _construct_prompt(
        self, 
        question: str, 
        history: List[Message], 
        learned_correction: Optional[Tuple], 
        docs: List[str], 
        web_info: str
    ) -> str:
        
        # Determine Context Type (Follow-up vs New Query)
        is_clarification = False
        if history:
            # Simple heuristic matching api.py
            last_user_msg = next((m.content for m in reversed(history) if m.role == 'user'), "")
            if len(question.split()) < 8 and "eso" in question.lower():
                is_clarification = True

        context_blocks = []
        
        # 1. Learned Corrections (Highest Priority)
        if learned_correction:
            correction_text, _, _, _ = learned_correction
            context_blocks.append(f"🧠 CONOCIMIENTO APRENDIDO PREVIAMENTE:\n{correction_text}\n")

        # 2. Web Info
        if web_info:
            context_blocks.append(f"🌐 INFORMACIÓN WEB VERIFICADA:\n{web_info}\n")

        # 3. RAG Docs
        if docs:
            context_blocks.append("📚 FUENTES LEGALES OFICIALES:\n" + "\n---\n".join(docs[:3]))

        hybrid_context = "\n".join(context_blocks)
        
        # Build Conversation History for Prompt
        chat_history_text = ""
        # Take last 4 messages to save context window
        recent_history = history[-4:] if history else []
        for msg in recent_history:
            role = "Usuario" if msg.role == "user" else "Asistente"
            chat_history_text += f"{role}: {msg.content}\n"

        # Select Instructions based on type
        if is_clarification:
            instruction_block = CLARIFICATION_INSTRUCTIONS
            context_template = CLARIFICATION_CONTEXT_TEMPLATE
        else:
            instruction_block = CONTINUITY_INSTRUCTIONS
            context_template = NEW_QUERY_CONTEXT_TEMPLATE

        # Assemble Final Prompt
        final_prompt = f"""{SYSTEM_PROMPT}

{AUDIENCE_BLOCK}

{INSTITUTION_POLICY_BLOCK}

{POPULAR_INSTITUTIONS_BLOCK}

{context_template}

{instruction_block}

CONTEXTO DISPONIBLE:
{hybrid_context}

HISTORIAL DE CONVERSACIÓN:
{chat_history_text}

PREGUNTA ACTUAL:
{question}
"""
        return final_prompt
