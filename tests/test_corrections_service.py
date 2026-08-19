"""
Gestión de las correcciones aprendidas.

Importan más que otros datos: tienen prioridad sobre los documentos oficiales,
así que una corrección mal escrita cambia todas las respuestas parecidas.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException

from src.app.core.security import ROLE_ADMIN, CurrentUser
from src.app.services.corrections_service import CorrectionsService


class IndiceFalso:
    def __init__(self):
        self.upserts = []
        self.eliminados = []

    def upsert(self, vectors, namespace):
        self.upserts.append((vectors, namespace))

    def delete(self, ids, namespace):
        self.eliminados.append((ids, namespace))


class EmbeddingsFalsos:
    def embed_query_sync(self, texto):
        return [0.1] * 8


class VectorStoreFalso:
    def __init__(self, metadatos=None):
        self.pinecone_index = IndiceFalso()
        self.embedding_service = EmbeddingsFalsos()
        self.metadatos = metadatos or {}

    def list_vector_ids(self, prefijo, namespace=""):
        return list(self.metadatos.keys())

    def fetch_vector_metadata(self, ids, namespace=""):
        return {i: self.metadatos[i] for i in ids if i in self.metadatos}


def correccion(pregunta, texto, fecha, entrenador="ana"):
    return {
        "original_question": pregunta,
        "text": texto,
        "timestamp": fecha,
        "trainer": entrenador,
        "intent": "correction",
    }


class BaseCorrecciones(unittest.TestCase):
    def setUp(self):
        self.store = VectorStoreFalso({
            "corr_1": correccion("¿Qué es el apremio?", "Es una orden de detención.", "2026-08-01T10:00:00"),
            "corr_2": correccion("¿Cuánto cuesta?", "El trámite es gratuito.", "2026-08-15T10:00:00"),
        })
        self.service = CorrectionsService(self.store)
        self.actor = CurrentUser(uid="u1", email="admin@ejemplo.cr", role=ROLE_ADMIN)


class TestListado(BaseCorrecciones):
    def test_lista_todas_las_correcciones(self):
        resultado = self.service.listar()
        self.assertEqual(resultado["total"], 2)
        self.assertEqual(len(resultado["correcciones"]), 2)

    def test_ordena_de_la_mas_nueva_a_la_mas_vieja(self):
        correcciones = self.service.listar()["correcciones"]
        self.assertEqual(correcciones[0]["pregunta"], "¿Cuánto cuesta?")

    def test_traduce_los_campos_al_vocabulario_del_panel(self):
        c = self.service.listar()["correcciones"][0]
        self.assertIn("pregunta", c)
        self.assertIn("correccion", c)
        self.assertIn("entrenador", c)


class TestEdicion(BaseCorrecciones):
    def test_edita_conservando_el_identificador(self):
        self.service.actualizar("corr_1", self.actor, correccion="Texto corregido.")
        vectores, namespace = self.store.pinecone_index.upserts[0]
        self.assertEqual(vectores[0][0], "corr_1")
        self.assertEqual(namespace, "corrections")
        self.assertEqual(vectores[0][2]["text"], "Texto corregido.")

    def test_deja_constancia_de_quien_editó(self):
        self.service.actualizar("corr_1", self.actor, correccion="Otro texto.")
        metadata = self.store.pinecone_index.upserts[0][0][0][2]
        self.assertEqual(metadata["edited_by"], "admin@ejemplo.cr")

    def test_conserva_al_entrenador_original(self):
        """Editar no borra a quien enseñó la corrección."""
        self.service.actualizar("corr_1", self.actor, correccion="Texto nuevo.")
        metadata = self.store.pinecone_index.upserts[0][0][0][2]
        self.assertEqual(metadata["trainer"], "ana")

    def test_rechaza_dejar_la_correccion_vacia(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar("corr_1", self.actor, correccion="   ")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_una_correccion_inexistente_da_404(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar("corr_999", self.actor, correccion="x")
        self.assertEqual(ctx.exception.status_code, 404)


class TestEliminacion(BaseCorrecciones):
    def test_elimina_del_namespace_correcto(self):
        resultado = self.service.eliminar("corr_2", self.actor)
        self.assertEqual(resultado["status"], "success")
        self.assertEqual(self.store.pinecone_index.eliminados[0], (["corr_2"], "corrections"))

    def test_no_se_puede_eliminar_lo_que_no_existe(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.eliminar("corr_999", self.actor)
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
