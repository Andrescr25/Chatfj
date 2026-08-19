"""
Registro de uso: documentos consultados e historial.

El historial guarda preguntas de personas usuarias, así que las reglas de
privacidad y de retención son parte del comportamiento a probar.
"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.services.analytics_service import AnalyticsService


class DocumentoFalso:
    def __init__(self, id_, datos):
        self.id = id_
        self._datos = datos
        self.reference = self
        self.borrado = False

    def to_dict(self):
        return self._datos

    def delete(self):
        self.borrado = True


class ColeccionFalsa:
    def __init__(self, documentos=None):
        self.documentos = documentos or []
        self.agregados = []
        self.escrituras = []

    def stream(self):
        return [d for d in self.documentos if not d.borrado]

    def add(self, datos):
        self.agregados.append(datos)

    def document(self, clave):
        coleccion = self

        class Ref:
            def set(self, datos, merge=False):
                coleccion.escrituras.append((clave, datos, merge))

        return Ref()


class FirestoreFalso:
    def __init__(self, colecciones):
        self.colecciones = colecciones

    def collection(self, nombre):
        return self.colecciones.setdefault(nombre, ColeccionFalsa())


def hace(dias):
    return (datetime.now(timezone.utc) - timedelta(days=dias)).isoformat()


class TestRegistro(unittest.TestCase):
    def setUp(self):
        self.colecciones = {}
        self.service = AnalyticsService()
        self.service.db = FirestoreFalso(self.colecciones)

    def test_guarda_pregunta_respuesta_y_documentos(self):
        self.service.registrar_consulta(
            "¿Qué es el apremio?", "Es una orden de detención.",
            [{"title": "Codigo_Penal.pdf"}], "groq",
        )
        registro = self.colecciones["history"].agregados[0]
        self.assertEqual(registro["pregunta"], "¿Qué es el apremio?")
        self.assertEqual(registro["documentos"], ["Codigo_Penal.pdf"])

    def test_no_guarda_nada_que_identifique_a_quien_pregunta(self):
        """El historial es de consultas, no de personas."""
        self.service.registrar_consulta("pregunta", "respuesta", [{"title": "a.pdf"}])
        registro = self.colecciones["history"].agregados[0]
        prohibidos = {"ip", "usuario", "user", "session", "sesion", "email", "uid"}
        self.assertEqual(prohibidos & set(registro.keys()), set())

    def test_guarda_solo_el_nombre_del_archivo(self):
        """En el índice conviven rutas completas y nombres sueltos."""
        self.service.registrar_consulta(
            "p", "r", [{"title": "data/docs/Codigo_Civil.pdf"}, {"filename": "otro.pdf"}]
        )
        self.assertEqual(
            self.colecciones["history"].agregados[0]["documentos"],
            ["Codigo_Civil.pdf", "otro.pdf"],
        )

    def test_cuenta_una_vez_por_documento_aunque_se_cite_dos_veces(self):
        self.service.registrar_consulta(
            "p", "r", [{"title": "a.pdf"}, {"title": "a.pdf"}, {"title": "b.pdf"}]
        )
        self.assertEqual(self.colecciones["history"].agregados[0]["documentos"], ["a.pdf", "b.pdf"])

    def test_un_fallo_al_registrar_no_interrumpe_nada(self):
        """La persona ya recibió su respuesta: esto no puede tumbar la petición."""
        class DbRoto:
            def collection(self, nombre):
                raise RuntimeError("Firestore caído")

        self.service.db = DbRoto()
        self.service.registrar_consulta("p", "r", [])  # no debe lanzar

    def test_sin_firestore_no_hace_nada(self):
        self.service.db = None
        self.service.registrar_consulta("p", "r", [])
        self.assertFalse(self.service.disponible)


class TestConsultas(unittest.TestCase):
    def setUp(self):
        self.colecciones = {
            "document_usage": ColeccionFalsa([
                DocumentoFalso("a", {"documento": "Codigo_Civil.pdf", "consultas": 7}),
                DocumentoFalso("b", {"documento": "Ley_RAC.pdf", "consultas": 3}),
            ]),
            "history": ColeccionFalsa([
                DocumentoFalso("h1", {"pregunta": "reciente", "creado_en": hace(1)}),
                DocumentoFalso("h2", {"pregunta": "vieja", "creado_en": hace(30)}),
            ]),
        }
        self.service = AnalyticsService()
        self.service.db = FirestoreFalso(self.colecciones)

    def test_ordena_los_documentos_por_uso(self):
        datos = self.service.documentos_mas_consultados()
        self.assertEqual(datos["documentos"][0]["documento"], "Codigo_Civil.pdf")
        self.assertEqual(datos["total_consultas"], 10)

    def test_calcula_el_porcentaje_de_cada_documento(self):
        datos = self.service.documentos_mas_consultados()
        self.assertEqual(datos["documentos"][0]["porcentaje"], 70.0)
        self.assertEqual(datos["documentos"][1]["porcentaje"], 30.0)

    def test_el_historial_solo_muestra_la_ventana_pedida(self):
        datos = self.service.historial(dias=7)
        preguntas = [c["pregunta"] for c in datos["consultas"]]
        self.assertIn("reciente", preguntas)
        self.assertNotIn("vieja", preguntas)

    def test_purga_lo_que_excede_la_retencion(self):
        borrados = self.service.purgar_historial(dias=7)
        self.assertEqual(borrados, 1)
        self.assertEqual([c["pregunta"] for c in self.service.historial()["consultas"]], ["reciente"])


if __name__ == "__main__":
    unittest.main()
