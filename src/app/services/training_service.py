import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.app.core.rag.store import VectorStoreService
from src.app.schemas.feedback import FeedbackItemRequest

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Clasificador heurístico ligero."""
    RULES = [
        ("definition", ["¿qué es", "que es", "definición", "definicion", "significa"]),
        ("procedure", ["¿cómo", "como ", "tramitar", "pasos", "proceso", "solicito", "hacer que", "presentar"]),
        ("requirements", ["requisitos", "documentos", "necesito llevar", "qué necesito", "que necesito"]),
        ("comparison", ["diferencia", "vs", "igual que", "lo mismo que", "comparación"]),
        ("rights", ["derecho", "obligación", "puedo", "debo", "me corresponde", "beneficio"]),
        ("contact", ["teléfono", "telefono", "contacto", "dirección", "direccion", "sede", "dónde queda"]),
    ]

    @classmethod
    def detect(cls, question: str) -> str:
        text = question.lower()
        for intent, keywords in cls.RULES:
            if any(keyword in text for keyword in keywords):
                return intent
        if len(text.split()) <= 3:
            return "follow_up"
        return "general"

class TrainingService:
    """Servicio de entrenamiento persistente usando Pinecone namespace 'corrections'."""
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        logger.info("✅ TrainingService inicializado con Pinecone (namespace=corrections)")

    def _generate_correction_id(self, question: str, timestamp: str) -> str:
        """Genera un ID único para la corrección."""
        raw = f"{question}_{timestamp}"
        return f"corr_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def learn_correction(self, item: FeedbackItemRequest):
        """Aprende una corrección y la guarda en Pinecone namespace 'corrections'."""
        try:
            timestamp = datetime.now().isoformat()
            correction_id = self._generate_correction_id(item.original_question, timestamp)
            trainer = getattr(item, 'trainer_name', None) or 'anon'
            
            success = self.vector_store.upsert_correction(
                correction_id=correction_id,
                question=item.original_question,
                correction_text=item.feedback,
                intent=item.intent,
                trainer=trainer
            )
            
            if success:
                logger.info(
                    f"✅ Corrección aprendida [{correction_id}] "
                    f"por [{trainer}]: {item.original_question[:40]}..."
                )
            else:
                logger.error(f"❌ No se pudo guardar corrección para: {item.original_question[:40]}...")
                
        except Exception as e:
            logger.error(f"❌ Error aprendiendo corrección: {e}")

    async def get_learned_correction_async(self, question: str) -> Optional[Tuple[str, float, str, str]]:
        """
        Busca correcciones aprendidas en Pinecone.
        Returns: (correction_text, score, correction_id, intent) or None
        """
        try:
            corrections = await self.vector_store.search_corrections_async(
                query=question, 
                k=3, 
                threshold=0.80
            )
            
            if corrections:
                best = corrections[0]  # Ya ordenado por score descendente
                logger.info(
                    f"📝 Corrección encontrada: score={best['score']:.4f}, "
                    f"trainer={best['trainer']}, "
                    f"pregunta original: {best['original_question'][:40]}..."
                )
                return (
                    best['correction'],
                    best['score'],
                    best['id'],
                    best['intent']
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error buscando corrección aprendida: {e}")
            return None

    def log_feedback(self, question: str, feedback: str, rating: int):
        """Log general feedback (for analytics, stored as correction if relevant)."""
        logger.info(
            f"📊 Feedback recibido: rating={rating}, "
            f"pregunta={question[:30]}..."
        )
