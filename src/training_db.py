"""
Sistema de Base de Datos para Modo Entrenamiento
Permite guardar feedback, evaluaciones y datos de mejora del sistema
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TrainingDatabase:
    """Base de datos para almacenar feedback y evaluaciones del sistema."""

    def __init__(self, db_path: str = "data/training.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Inicializa las tablas de la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla de conversaciones evaluadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                category_detected TEXT,
                processing_time REAL,
                status TEXT NOT NULL,
                evaluator_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabla de feedback detallado
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                field TEXT NOT NULL,
                issue TEXT,
                correct_value TEXT,
                severity TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
            )
        """)

        # Tabla de métricas agregadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_evaluations INTEGER,
                approved INTEGER,
                rejected INTEGER,
                approval_rate REAL,
                avg_processing_time REAL,
                common_issues TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabla de correcciones aprendidas (NUEVA - Aprendizaje en tiempo real)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT NOT NULL,
                question_text TEXT NOT NULL,
                original_answer TEXT NOT NULL,
                corrected_answer TEXT NOT NULL,
                correction_type TEXT NOT NULL,
                category TEXT,
                times_used INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 100.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_used TEXT
            )
        """)

        # Índice para búsqueda rápida de correcciones por hash de pregunta
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_hash ON learned_corrections(question_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON learned_corrections(category)")

        # Índices para búsquedas rápidas
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON evaluations(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")

        conn.commit()
        conn.close()
        logger.info(f"✅ Base de datos de entrenamiento inicializada: {self.db_path}")

    def save_evaluation(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        category_detected: str,
        processing_time: float,
        status: str,
        evaluator_notes: str = ""
    ) -> int:
        """
        Guarda una evaluación en la base de datos.

        Args:
            question: Pregunta del usuario
            answer: Respuesta generada por el sistema
            sources: Lista de fuentes utilizadas
            category_detected: Categoría detectada por el sistema
            processing_time: Tiempo de procesamiento en segundos
            status: "approved", "rejected", "pending"
            evaluator_notes: Notas del evaluador

        Returns:
            ID de la evaluación guardada
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO evaluations
            (timestamp, question, answer, sources, category_detected, processing_time, status, evaluator_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            question,
            answer,
            json.dumps(sources, ensure_ascii=False),
            category_detected,
            processing_time,
            status,
            evaluator_notes
        ))

        evaluation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"✅ Evaluación guardada: ID={evaluation_id}, Status={status}")
        return evaluation_id

    def add_feedback(
        self,
        evaluation_id: int,
        feedback_type: str,
        field: str,
        issue: str = "",
        correct_value: str = "",
        severity: str = "medium"
    ) -> int:
        """
        Agrega feedback detallado a una evaluación.

        Args:
            evaluation_id: ID de la evaluación
            feedback_type: "citation_error", "category_error", "format_error", "content_error", "suggestion"
            field: Campo específico con error (ej: "legal_citation", "institution", "category")
            issue: Descripción del problema
            correct_value: Valor correcto (si aplica)
            severity: "low", "medium", "high", "critical"

        Returns:
            ID del feedback guardado
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO feedback
            (evaluation_id, feedback_type, field, issue, correct_value, severity)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            evaluation_id,
            feedback_type,
            field,
            issue,
            correct_value,
            severity
        ))

        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"✅ Feedback agregado: Evaluation={evaluation_id}, Type={feedback_type}, Severity={severity}")
        return feedback_id

    def get_evaluation(self, evaluation_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene una evaluación por ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            "id": row[0],
            "timestamp": row[1],
            "question": row[2],
            "answer": row[3],
            "sources": json.loads(row[4]) if row[4] else [],
            "category_detected": row[5],
            "processing_time": row[6],
            "status": row[7],
            "evaluator_notes": row[8],
            "created_at": row[9]
        }

    def get_pending_evaluations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene evaluaciones pendientes de revisión."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, timestamp, question, answer, category_detected, processing_time, status
            FROM evaluations
            WHERE status = 'pending'
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        evaluations = []
        for row in cursor.fetchall():
            evaluations.append({
                "id": row[0],
                "timestamp": row[1],
                "question": row[2],
                "answer": row[3],
                "category_detected": row[4],
                "processing_time": row[5],
                "status": row[6]
            })

        conn.close()
        return evaluations

    def get_feedback_for_evaluation(self, evaluation_id: int) -> List[Dict[str, Any]]:
        """Obtiene todo el feedback de una evaluación."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, feedback_type, field, issue, correct_value, severity, created_at
            FROM feedback
            WHERE evaluation_id = ?
            ORDER BY created_at DESC
        """, (evaluation_id,))

        feedback_list = []
        for row in cursor.fetchall():
            feedback_list.append({
                "id": row[0],
                "feedback_type": row[1],
                "field": row[2],
                "issue": row[3],
                "correct_value": row[4],
                "severity": row[5],
                "created_at": row[6]
            })

        conn.close()
        return feedback_list

    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Obtiene estadísticas de las últimas evaluaciones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Estadísticas generales
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                AVG(processing_time) as avg_time
            FROM evaluations
            WHERE date(timestamp) >= date('now', '-' || ? || ' days')
        """, (days,))

        row = cursor.fetchone()
        total = row[0] or 0
        approved = row[1] or 0
        rejected = row[2] or 0
        pending = row[3] or 0
        avg_time = row[4] or 0

        # Errores más comunes
        cursor.execute("""
            SELECT feedback_type, COUNT(*) as count
            FROM feedback
            WHERE evaluation_id IN (
                SELECT id FROM evaluations
                WHERE date(timestamp) >= date('now', '-' || ? || ' days')
            )
            GROUP BY feedback_type
            ORDER BY count DESC
            LIMIT 5
        """, (days,))

        common_errors = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]

        conn.close()

        return {
            "total_evaluations": total,
            "approved": approved,
            "rejected": rejected,
            "pending": pending,
            "approval_rate": (approved / total * 100) if total > 0 else 0,
            "avg_processing_time": avg_time,
            "common_errors": common_errors
        }

    def export_training_data(self, output_path: str, status: str = "approved") -> int:
        """
        Exporta datos de entrenamiento a formato JSONL para fine-tuning.

        Args:
            output_path: Ruta del archivo de salida
            status: Filtrar por estado ("approved", "rejected", "all")

        Returns:
            Número de registros exportados
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT question, answer, sources, category_detected FROM evaluations"
        if status != "all":
            query += f" WHERE status = '{status}'"

        cursor.execute(query)

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in cursor.fetchall():
                data = {
                    "question": row[0],
                    "answer": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "category": row[3]
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1

        conn.close()
        logger.info(f"✅ Exportados {count} registros a {output_path}")
        return count

    def update_evaluation_status(self, evaluation_id: int, status: str, notes: str = ""):
        """Actualiza el estado de una evaluación."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE evaluations
            SET status = ?, evaluator_notes = ?
            WHERE id = ?
        """, (status, notes, evaluation_id))

        conn.commit()
        conn.close()
        logger.info(f"✅ Evaluación {evaluation_id} actualizada a: {status}")

    # ============================================
    # MÉTODOS PARA APRENDIZAJE EN TIEMPO REAL
    # ============================================

    def _hash_question(self, question: str) -> str:
        """Genera un hash normalizado de la pregunta para matching."""
        import hashlib
        # Normalizar: lowercase, sin espacios extra, sin puntuación al final
        normalized = question.lower().strip().rstrip('?¿!.,;:')
        normalized = ' '.join(normalized.split())  # Eliminar espacios múltiples
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def save_learned_correction(
        self,
        question: str,
        original_answer: str,
        corrected_answer: str,
        correction_type: str,
        category: str = "general"
    ) -> int:
        """
        Guarda una corrección aprendida para uso futuro inmediato.

        Args:
            question: Pregunta original del usuario
            original_answer: Respuesta original (incorrecta)
            corrected_answer: Respuesta corregida por el evaluador
            correction_type: Tipo de corrección (citation, category, content, format)
            category: Categoría de la pregunta

        Returns:
            ID de la corrección guardada
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        question_hash = self._hash_question(question)

        # Verificar si ya existe una corrección para esta pregunta
        cursor.execute("""
            SELECT id FROM learned_corrections
            WHERE question_hash = ?
        """, (question_hash,))

        existing = cursor.fetchone()

        if existing:
            # Actualizar corrección existente
            cursor.execute("""
                UPDATE learned_corrections
                SET corrected_answer = ?,
                    correction_type = ?,
                    category = ?,
                    created_at = ?
                WHERE id = ?
            """, (corrected_answer, correction_type, category, datetime.now().isoformat(), existing[0]))
            correction_id = existing[0]
            logger.info(f"🔄 Corrección actualizada: ID={correction_id}")
        else:
            # Insertar nueva corrección
            cursor.execute("""
                INSERT INTO learned_corrections
                (question_hash, question_text, original_answer, corrected_answer, correction_type, category)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (question_hash, question, original_answer, corrected_answer, correction_type, category))
            correction_id = cursor.lastrowid
            logger.info(f"✅ Nueva corrección aprendida: ID={correction_id}, Category={category}")

        conn.commit()
        conn.close()

        return correction_id

    def get_learned_correction(self, question: str, similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """
        Busca una corrección aprendida para la pregunta dada usando búsqueda semántica.

        NUEVO: Ahora usa embeddings para encontrar preguntas similares, no solo idénticas.

        Args:
            question: Pregunta del usuario
            similarity_threshold: Umbral de similitud mínima (0.0 a 1.0, default 0.85)

        Returns:
            Diccionario con la corrección si existe, None si no
        """
        try:
            # Intentar primero con hash exacto (más rápido)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            question_hash = self._hash_question(question)

            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category, times_used
                FROM learned_corrections
                WHERE question_hash = ?
            """, (question_hash,))

            row = cursor.fetchone()

            if row:
                conn.close()
                logger.info(f"✅ Corrección encontrada con hash exacto")
                return {
                    "id": row[0],
                    "question_text": row[1],
                    "corrected_answer": row[2],
                    "correction_type": row[3],
                    "category": row[4],
                    "times_used": row[5],
                    "similarity_score": 1.0
                }

            # Si no hay match exacto, usar búsqueda semántica
            logger.info(f"🔍 No hay match exacto. Buscando correcciones similares...")

            # Obtener todas las correcciones para comparar
            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category, times_used
                FROM learned_corrections
                ORDER BY times_used DESC
                LIMIT 100
            """)

            all_corrections = cursor.fetchall()
            conn.close()

            if not all_corrections:
                return None

            # Calcular embeddings y similitud
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Cargar modelo (esto se cachea automáticamente)
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Embedding de la pregunta actual
            question_embedding = model.encode([question])[0]

            # Embeddings de todas las preguntas guardadas
            saved_questions = [row[1] for row in all_corrections]
            saved_embeddings = model.encode(saved_questions)

            # Calcular similitud coseno
            from numpy.linalg import norm
            similarities = [
                np.dot(question_embedding, saved_emb) / (norm(question_embedding) * norm(saved_emb))
                for saved_emb in saved_embeddings
            ]

            # Encontrar la mejor coincidencia
            max_similarity = max(similarities)
            best_idx = similarities.index(max_similarity)

            logger.info(f"🎯 Mejor similitud: {max_similarity:.3f} con pregunta: '{saved_questions[best_idx][:60]}...'")

            if max_similarity >= similarity_threshold:
                best_match = all_corrections[best_idx]
                logger.info(f"✅ Corrección similar encontrada (similitud: {max_similarity:.3f})")
                return {
                    "id": best_match[0],
                    "question_text": best_match[1],
                    "corrected_answer": best_match[2],
                    "correction_type": best_match[3],
                    "category": best_match[4],
                    "times_used": best_match[5],
                    "similarity_score": float(max_similarity)
                }
            else:
                logger.info(f"❌ No se encontró corrección similar (mejor similitud: {max_similarity:.3f} < {similarity_threshold})")
                return None

        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {e}")
            return None

    def mark_correction_used(self, correction_id: int):
        """Marca que una corrección fue usada y actualiza estadísticas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE learned_corrections
            SET times_used = times_used + 1,
                last_used = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), correction_id))

        conn.commit()
        conn.close()
        logger.info(f"📊 Corrección {correction_id} marcada como usada")

    def get_all_corrections(self, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene todas las correcciones aprendidas, opcionalmente filtradas por categoría.

        Args:
            category: Filtrar por categoría (None = todas)
            limit: Límite de resultados

        Returns:
            Lista de correcciones
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if category:
            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category, times_used, created_at
                FROM learned_corrections
                WHERE category = ?
                ORDER BY times_used DESC, created_at DESC
                LIMIT ?
            """, (category, limit))
        else:
            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category, times_used, created_at
                FROM learned_corrections
                ORDER BY times_used DESC, created_at DESC
                LIMIT ?
            """, (limit,))

        corrections = []
        for row in cursor.fetchall():
            corrections.append({
                "id": row[0],
                "question_text": row[1],
                "corrected_answer": row[2],
                "correction_type": row[3],
                "category": row[4],
                "times_used": row[5],
                "created_at": row[6]
            })

        conn.close()
        return corrections
