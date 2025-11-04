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
