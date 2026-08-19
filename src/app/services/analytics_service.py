"""
Registro de uso: qué se pregunta y qué documentos se consultan.

Sirve para dos cosas que pidió la oficina: saber cuáles documentos sostienen
realmente las respuestas y poder revisar el historial reciente de consultas.

Decisiones de privacidad:
  - Solo se guarda la pregunta, la respuesta y los nombres de los documentos
    usados. No se guarda IP, identificador de sesión ni nada que permita
    reconstruir quién preguntó.
  - El historial se conserva 7 días y se purga solo.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

COLECCION_USO = "document_usage"
COLECCION_HISTORIAL = "history"
DIAS_DE_RETENCION = 7


def _firestore():
    try:
        import firebase_admin
        from firebase_admin import firestore

        if firebase_admin._apps:
            return firestore.client()
    except Exception as e:
        logger.warning(f"⚠️ Firestore no disponible para analítica: {e}")
    return None


def _clave(nombre: str) -> str:
    """Firestore no admite '/' en los identificadores de documento."""
    return nombre.replace("/", "__")[:1500] or "sin-nombre"


class AnalyticsService:
    def __init__(self):
        self.db = _firestore()

    @property
    def disponible(self) -> bool:
        return self.db is not None

    # ---------- Escritura ----------

    def registrar_consulta(
        self,
        pregunta: str,
        respuesta: str,
        fuentes: List[Dict[str, Any]],
        proveedor: str = "",
    ) -> None:
        """
        Deja constancia de una consulta. Nunca interrumpe la respuesta: si algo
        falla aquí, la persona ya recibió lo que preguntó.
        """
        if not self.db:
            return

        try:
            from firebase_admin import firestore

            ahora = datetime.now(timezone.utc)
            documentos = sorted({
                # Solo el nombre del archivo: en el índice conviven rutas
                # completas y nombres sueltos según cómo se ingirió cada uno.
                (f.get("title") or f.get("filename") or "").strip().split("/")[-1]
                for f in fuentes
                if isinstance(f, dict) and (f.get("title") or f.get("filename"))
            })

            self.db.collection(COLECCION_HISTORIAL).add({
                "pregunta": pregunta[:2000],
                "respuesta": respuesta[:8000],
                "documentos": documentos,
                "proveedor": proveedor,
                "creado_en": ahora.isoformat(),
            })

            for nombre in documentos:
                self.db.collection(COLECCION_USO).document(_clave(nombre)).set(
                    {
                        "documento": nombre,
                        "consultas": firestore.Increment(1),
                        "ultima_consulta": ahora.isoformat(),
                    },
                    merge=True,
                )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo registrar la consulta: {e}")

    # ---------- Lectura ----------

    def documentos_mas_consultados(self, limite: int = 25) -> Dict[str, Any]:
        if not self.db:
            return {"documentos": [], "total_consultas": 0, "disponible": False}

        registros = []
        try:
            for snap in self.db.collection(COLECCION_USO).stream():
                d = snap.to_dict() or {}
                registros.append({
                    "documento": d.get("documento", snap.id),
                    "consultas": d.get("consultas", 0),
                    "ultima_consulta": d.get("ultima_consulta", ""),
                })
        except Exception as e:
            logger.error(f"❌ Error leyendo el uso de documentos: {e}")
            return {"documentos": [], "total_consultas": 0, "disponible": False}

        registros.sort(key=lambda r: r["consultas"], reverse=True)
        total = sum(r["consultas"] for r in registros)
        for r in registros:
            r["porcentaje"] = round(r["consultas"] / total * 100, 1) if total else 0.0

        return {
            "documentos": registros[:limite],
            "total_consultas": total,
            "documentos_distintos": len(registros),
            "disponible": True,
        }

    def historial(self, dias: int = DIAS_DE_RETENCION, limite: int = 200) -> Dict[str, Any]:
        if not self.db:
            return {"consultas": [], "disponible": False}

        corte = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
        consultas = []
        try:
            for snap in self.db.collection(COLECCION_HISTORIAL).stream():
                d = snap.to_dict() or {}
                if d.get("creado_en", "") >= corte:
                    d["id"] = snap.id
                    consultas.append(d)
        except Exception as e:
            logger.error(f"❌ Error leyendo el historial: {e}")
            return {"consultas": [], "disponible": False}

        consultas.sort(key=lambda c: c.get("creado_en", ""), reverse=True)
        return {
            "consultas": consultas[:limite],
            "total": len(consultas),
            "dias": dias,
            "disponible": True,
        }

    def purgar_historial(self, dias: int = DIAS_DE_RETENCION) -> int:
        """Borra lo que exceda la ventana de retención. Devuelve cuántos borró."""
        if not self.db:
            return 0

        corte = (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()
        borrados = 0
        try:
            for snap in self.db.collection(COLECCION_HISTORIAL).stream():
                if (snap.to_dict() or {}).get("creado_en", "") < corte:
                    snap.reference.delete()
                    borrados += 1
        except Exception as e:
            logger.error(f"❌ Error purgando el historial: {e}")
        if borrados:
            logger.info(f"🧹 Historial: {borrados} consultas con más de {dias} días eliminadas")
        return borrados
