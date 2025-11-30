"""
Sistema de Base de Datos para Modo Entrenamiento
Permite guardar feedback, evaluaciones y datos de mejora del sistema
"""

import math
import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Clasificador heurístico ligero para identificar la intención de una pregunta."""

    RULES = [
        ("definition", ["¿qué es", "que es", "definición", "definicion", "significa"]),
        ("procedure", ["¿cómo", "como ", "tramitar", "pasos", "proceso", "solicito", "hacer que", "guiame", "presentar"]),
        ("requirements", ["requisitos", "documentos", "necesito llevar", "qué necesito", "que necesito", "presentar", "adjuntar"]),
        ("comparison", ["diferencia", "vs", "igual que", "lo mismo que", "comparación", "comparacion"]),
        ("rights", ["derecho", "obligación", "obligacion", "puedo", "debo", "me corresponde", "beneficio"]),
        ("contact", ["teléfono", "telefono", "contacto", "dirección", "direccion", "sede", "dónde queda", "donde queda"]),
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


class TrainingDatabase:
    """Base de datos para almacenar feedback y evaluaciones del sistema."""

    def __init__(self, db_path: str = "data/training.db"):
        self.db_path = db_path
        self._embedder: Optional[SentenceTransformer] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _get_embedder(self) -> SentenceTransformer:
        """Carga perezosamente el modelo de embeddings usado para aprendizaje."""
        if self._embedder is None:
            model_name = os.getenv("LEARNING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"🔤 Cargando modelo de embeddings para aprendizaje: {model_name}")
            self._embedder = SentenceTransformer(model_name)
        return self._embedder

    def _encode_vector(self, text: str) -> np.ndarray:
        """Retorna el vector de embeddings en formato numpy."""
        return self._get_embedder().encode([text or ""])[0].astype(np.float32)

    def _encode_embedding(self, text: str) -> bytes:
        """Genera el embedding en formato binario listo para SQLite."""
        vector = self._encode_vector(text)
        return vector.tobytes()

    def _embedding_from_blob(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    def _ensure_column(self, cursor: sqlite3.Cursor, table: str, column: str, definition: str):
        """Agrega una columna si no existe (migraciones rápidas)."""
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            logger.info(f"🆕 Columna agregada: {table}.{column}")

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
                effective_uses REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_used TEXT,
                intent TEXT DEFAULT 'general',
                embedding BLOB
            )
        """)

        self._ensure_column(cursor, "learned_corrections", "intent", "TEXT DEFAULT 'general'")
        self._ensure_column(cursor, "learned_corrections", "embedding", "BLOB")
        self._ensure_column(cursor, "learned_corrections", "effective_uses", "REAL DEFAULT 0.0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correction_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                correction_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                mode TEXT DEFAULT 'chat',
                result TEXT DEFAULT 'pending',
                source TEXT DEFAULT 'implicit',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                FOREIGN KEY (correction_id) REFERENCES learned_corrections(id)
            )
        """)

        # Índice para búsqueda rápida de correcciones por hash de pregunta
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_hash ON learned_corrections(question_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON learned_corrections(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intent ON learned_corrections(intent)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_usage_correction ON correction_usage(correction_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_usage_result ON correction_usage(result)")

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
        intent = IntentClassifier.detect(question)
        embedding_blob = sqlite3.Binary(self._encode_embedding(question))

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
                    created_at = ?,
                    intent = ?,
                    embedding = ?
                WHERE id = ?
            """, (
                corrected_answer,
                correction_type,
                category,
                datetime.now().isoformat(),
                intent,
                embedding_blob,
                existing[0]
            ))
            correction_id = existing[0]
            logger.info(f"🔄 Corrección actualizada: ID={correction_id}")
        else:
            # Insertar nueva corrección
            cursor.execute("""
                INSERT INTO learned_corrections
                (question_hash, question_text, original_answer, corrected_answer, correction_type, category, intent, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                question_hash,
                question,
                original_answer,
                corrected_answer,
                correction_type,
                category,
                intent,
                embedding_blob
            ))
            correction_id = cursor.lastrowid
            logger.info(f"✅ Nueva corrección aprendida: ID={correction_id}, Category={category}")

        conn.commit()
        conn.close()

        return correction_id

    def get_learned_correction(self, question: str, similarity_threshold: float = 0.75) -> Optional[Dict[str, Any]]:
        """
        Busca una corrección aprendida para la pregunta dada usando búsqueda semántica.

        NUEVO: Ahora usa embeddings para encontrar preguntas similares, no solo idénticas.

        Args:
            question: Pregunta del usuario
            similarity_threshold: Umbral de similitud mínima (0.0 a 1.0, default 0.75 - balance entre variaciones legítimas y falsos positivos)

        Returns:
            Diccionario con la corrección si existe, None si no
        """
        try:
            # Intentar primero con hash exacto (más rápido)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            question_hash = self._hash_question(question)
            target_intent = IntentClassifier.detect(question)

            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category,
                       times_used, intent, embedding, success_rate, effective_uses, last_used
                FROM learned_corrections
                WHERE question_hash = ?
            """, (question_hash,))

            row = cursor.fetchone()

            if row:
                if not row[7]:
                    embedding_blob = sqlite3.Binary(self._encode_embedding(row[1]))
                    cursor.execute("UPDATE learned_corrections SET embedding = ? WHERE id = ?", (embedding_blob, row[0]))
                    conn.commit()
                conn.close()
                logger.info(f"✅ Corrección encontrada con hash exacto")
                return {
                    "id": row[0],
                    "question_text": row[1],
                    "corrected_answer": row[2],
                    "correction_type": row[3],
                    "category": row[4],
                    "times_used": row[5],
                    "similarity_score": 1.0,
                    "intent": row[6],
                    "success_rate": row[8],
                    "effective_uses": row[9]
                }

            # Si no hay match exacto, usar búsqueda semántica
            logger.info(f"🔍 No hay match exacto. Buscando correcciones similares...")

            # Obtener todas las correcciones para comparar
            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category,
                       times_used, intent, embedding, success_rate, effective_uses, last_used
                FROM learned_corrections
                ORDER BY times_used DESC, COALESCE(last_used, created_at) DESC
                LIMIT 300
            """)

            all_corrections = cursor.fetchall()
            conn.close()

            if not all_corrections:
                return None

            filtered = [row for row in all_corrections if row[6] == target_intent]
            if not filtered and target_intent != "general":
                filtered = [row for row in all_corrections if row[6] == "general"]
            if not filtered:
                filtered = all_corrections

            question_vector = self._encode_vector(question)
            question_norm = np.linalg.norm(question_vector)

            candidates = []

            # Reutilizar conexión para actualizar embeddings faltantes
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for row in filtered:
                correction_id = row[0]
                question_text = row[1]
                embedding_blob = row[7]
                saved_vector = self._embedding_from_blob(embedding_blob)
                if saved_vector is None:
                    saved_vector = self._encode_vector(question_text)
                    cursor.execute(
                        "UPDATE learned_corrections SET embedding = ? WHERE id = ?",
                        (sqlite3.Binary(saved_vector.tobytes()), correction_id)
                    )

                saved_norm = np.linalg.norm(saved_vector)
                if question_norm == 0 or saved_norm == 0:
                    continue

                similarity = float(np.dot(question_vector, saved_vector) / (question_norm * saved_norm))
                success_rate = row[8] or 100.0
                effective_uses = row[9] or 0.0
                last_used = row[10]

                recency_score = 0.5
                if last_used:
                    try:
                        last_dt = datetime.fromisoformat(last_used)
                        days_since = (datetime.now() - last_dt).days
                        recency_score = max(0.0, 1 - min(days_since, 90) / 90)
                    except ValueError:
                        recency_score = 0.5

                usage_score = min(1.0, math.log1p(effective_uses + 1) / 3.0)
                success_norm = min(1.0, max(0.0, success_rate / 100))

                final_score = (
                    0.45 * similarity +
                    0.25 * success_norm +
                    0.20 * recency_score +
                    0.10 * usage_score
                )

                candidates.append({
                    "row": row,
                    "similarity": similarity,
                    "final_score": final_score,
                    "success_rate": success_rate,
                    "effective_uses": effective_uses
                })

            conn.commit()
            conn.close()

            eligible = [c for c in candidates if c["similarity"] >= similarity_threshold]
            if eligible:
                best_candidate = max(eligible, key=lambda c: c["final_score"])
                best_match = best_candidate["row"]
                logger.info(
                    f"✅ Corrección similar encontrada (intención: {target_intent}, "
                    f"similitud: {best_candidate['similarity']:.3f}, score: {best_candidate['final_score']:.3f})"
                )
                return {
                    "id": best_match[0],
                    "question_text": best_match[1],
                    "corrected_answer": best_match[2],
                    "correction_type": best_match[3],
                    "category": best_match[4],
                    "times_used": best_match[5],
                    "similarity_score": float(best_candidate["similarity"]),
                    "intent": best_match[6],
                    "success_rate": best_candidate["success_rate"],
                    "effective_uses": best_candidate["effective_uses"]
                }

            best_similarity = max((c["similarity"] for c in candidates), default=0.0)
            logger.info(f"❌ No se encontró corrección similar (mejor similitud: {best_similarity:.3f} < {similarity_threshold})")
            return None

        except Exception as e:
            logger.error(f"❌ Error en búsqueda semántica: {e}")
            return None

    def _recalculate_correction_metrics(self, cursor: sqlite3.Cursor, correction_id: int):
        cursor.execute("""
            SELECT result, created_at FROM correction_usage
            WHERE correction_id = ?
        """, (correction_id,))
        rows = cursor.fetchall()

        if not rows:
            cursor.execute("""
                UPDATE learned_corrections
                SET success_rate = 100.0,
                    effective_uses = 0.0
                WHERE id = ?
            """, (correction_id,))
            return

        successes = 0
        failures = 0
        effective_uses = 0.0
        now = datetime.now()

        for result, created_at in rows:
            created_dt = datetime.fromisoformat(created_at) if created_at else now
            days = max((now - created_dt).days, 0)
            weight = 0.9 ** (days / 30)  # decaimiento aproximado cada mes
            effective_uses += weight

            if result == "success":
                successes += 1
            elif result == "fail":
                failures += 1

        considered = successes + failures
        success_rate = (successes / considered * 100) if considered > 0 else 100.0

        cursor.execute("""
            UPDATE learned_corrections
            SET success_rate = ?,
                effective_uses = ?,
                times_used = (
                    SELECT COUNT(*) FROM correction_usage WHERE correction_id = ?
                )
            WHERE id = ?
        """, (success_rate, effective_uses, correction_id, correction_id))

    def log_correction_usage(self, correction_id: int, question: str, mode: str = "chat") -> int:
        """Registra un uso pendiente para posterior retroalimentación."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO correction_usage (correction_id, question, mode, result)
            VALUES (?, ?, ?, 'pending')
        """, (correction_id, question, mode))

        usage_id = cursor.lastrowid

        cursor.execute("""
            UPDATE learned_corrections
            SET times_used = times_used + 1,
                last_used = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), correction_id))

        conn.commit()
        conn.close()
        logger.info(f"📊 Uso registrado para corrección {correction_id} (usage_id={usage_id})")
        return usage_id

    def finalize_correction_usage(self, usage_id: int, result: str, source: str = "explicit") -> None:
        """Marca un uso como éxito o fallo y recalcula métricas."""
        if result not in {"success", "fail"}:
            raise ValueError("result debe ser 'success' o 'fail'")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT correction_id FROM correction_usage
            WHERE id = ?
        """, (usage_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"usage_id {usage_id} no encontrado")

        correction_id = row[0]

        cursor.execute("""
            UPDATE correction_usage
            SET result = ?, source = ?, updated_at = ?
            WHERE id = ?
        """, (result, source, datetime.now().isoformat(), usage_id))

        self._recalculate_correction_metrics(cursor, correction_id)

        conn.commit()
        conn.close()
        logger.info(f"✅ Uso {usage_id} registrado como {result} (corrección {correction_id})")

    def mark_correction_used(self, correction_id: int, question: str, mode: str = "chat") -> int:
        """Compatibilidad con llamadas existentes."""
        return self.log_correction_usage(correction_id, question, mode)

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
                SELECT id, question_text, corrected_answer, correction_type, category,
                       times_used, created_at, intent, success_rate, effective_uses
                FROM learned_corrections
                WHERE category = ?
                ORDER BY times_used DESC, created_at DESC
                LIMIT ?
            """, (category, limit))
        else:
            cursor.execute("""
                SELECT id, question_text, corrected_answer, correction_type, category,
                       times_used, created_at, intent, success_rate, effective_uses
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
                "created_at": row[6],
                "intent": row[7],
                "success_rate": row[8] if len(row) > 8 else None,
                "effective_uses": row[9] if len(row) > 9 else None
            })

        conn.close()
        return corrections
