import os
import io
import sqlite3
import json
import logging
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.app.core.rag.embeddings import EmbeddingService
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
    def __init__(self, embedding_service: EmbeddingService, db_path: str = "data/training.db"):
        self.db_path = db_path
        self.embedding_service = embedding_service
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de evaluaciones generales
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    user_feedback TEXT,
                    rating INTEGER, -- 1 like, -1 dislike
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla de aprendizaje de correcciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_question TEXT,
                    context_hash TEXT, -- Hash del contexto original para matching
                    corrected_response TEXT,
                    intent_category TEXT,
                    embedding BLOB,
                    confidence_score REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def _encode_embedding(self, text: str) -> bytes:
        """Genera embedding y lo serializa a bytes."""
        try:
            vector = self.embedding_service.embed_query_sync(text)
            # Normalizar vector
            np_vector = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(np_vector)
            if norm > 0:
                np_vector = np_vector / norm
            return np_vector.tobytes()
        except Exception as e:
            logger.error(f"Error generando embedding para training: {e}")
            return b""

    def _decode_embedding(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def learn_correction(self, item: FeedbackItemRequest):
        """Aprende una corrección basada en feedback explícito."""
        try:
            embedding_blob = self._encode_embedding(item.original_question)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO learned_corrections 
                    (original_question, corrected_response, intent_category, embedding, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    item.original_question,
                    item.feedback, # El feedback se toma como la respuesta correcta o instrucción
                    item.intent,
                    embedding_blob,
                    1.0
                ))
                conn.commit()
                logger.info(f"✅ Aprendida nueva corrección para: {item.original_question[:30]}...")
        except Exception as e:
            logger.error(f"Error aprendiendo corrección: {e}")

    def get_learned_correction(self, question: str, threshold: float = 0.85) -> Optional[Tuple[str, float, int, str]]:
        """Busca si existe una corrección aprendida similar."""
        try:
            query_vector = self.embedding_service.embed_query_sync(question)
            query_np = np.array(query_vector, dtype=np.float32)
            
            best_match = None
            best_score = -1.0
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Optimización: traer solo lo necesario.
                # En producción real con muchos datos, usar FAISS o similar.
                cursor.execute("SELECT id, corrected_response, intent_category, embedding, usage_count FROM learned_corrections")
                rows = cursor.fetchall()
                
                for row in rows:
                    if not row['embedding']:
                        continue
                        
                    db_vector = self._decode_embedding(row['embedding'])
                    score = self._cosine_similarity(query_np, db_vector)
                    
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = row

                if best_match:
                    # Actualizar contador de uso
                    cursor.execute("UPDATE learned_corrections SET usage_count = usage_count + 1 WHERE id = ?", (best_match['id'],))
                    conn.commit()
                    
                    return (
                        best_match['corrected_response'],
                        float(best_score),
                        best_match['id'],
                        best_match['intent_category']
                    )
                    
            return None
            
        except Exception as e:
            logger.error(f"Error buscando corrección aprendida: {e}")
            return None

    def log_feedback(self, question: str, feedback: str, rating: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO evaluations (question, user_feedback, rating) VALUES (?, ?, ?)",
                    (question, feedback, rating)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error guardando feedback general: {e}")
