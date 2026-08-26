"""
Doble indexado: un espacio por modelo de embeddings.

La regla que estas pruebas protegen: un vector solo puede compararse con
vectores del mismo modelo. Buscar en el espacio equivocado no lanza ningún
error, devuelve resultados sin sentido, y nadie se entera. Es la falla más
peligrosa del sistema.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core.rag.embeddings import ESPACIO_E5, ESPACIO_GEMINI, EmbeddingService


class ProveedorFalso:
    def __init__(self, valor=0.1, error=None, espacio=ESPACIO_E5):
        self.valor = valor
        self.error = error
        self.espacio = espacio
        self.nombre = f"falso({espacio or 'e5'})"
        self.llamadas = 0

    def embed_query(self, texto):
        self.llamadas += 1
        if self.error:
            raise self.error
        return [self.valor] * 1024

    def embed_documents(self, textos):
        self.llamadas += 1
        if self.error:
            raise self.error
        return [[self.valor] * 1024 for _ in textos]


def servicio(principal=None, respaldo=None, gemini=None):
    s = EmbeddingService.__new__(EmbeddingService)
    s.client = principal
    s.respaldo = respaldo
    s.gemini = gemini
    return s


class TestEleccionDelEspacio(unittest.TestCase):
    def test_con_e5_disponible_se_consulta_su_espacio(self):
        vector, espacio = servicio(
            principal=ProveedorFalso(0.1), gemini=ProveedorFalso(0.9, espacio=ESPACIO_GEMINI)
        ).embed_query_con_espacio("hola")
        self.assertEqual(espacio, ESPACIO_E5)
        self.assertEqual(vector[0], 0.1)

    def test_si_e5_falla_se_consulta_el_espacio_de_gemini(self):
        """El caso real: HuggingFace sin crédito."""
        gemini = ProveedorFalso(0.9, espacio=ESPACIO_GEMINI)
        vector, espacio = servicio(
            principal=ProveedorFalso(error=RuntimeError("402 sin crédito")), gemini=gemini
        ).embed_query_con_espacio("hola")
        self.assertEqual(espacio, ESPACIO_GEMINI)
        self.assertEqual(vector[0], 0.9)

    def test_el_vector_de_gemini_nunca_se_atribuye_al_espacio_de_e5(self):
        """La confusión que haría devolver resultados sin sentido."""
        gemini = ProveedorFalso(0.9, espacio=ESPACIO_GEMINI)
        _vector, espacio = servicio(
            principal=ProveedorFalso(error=RuntimeError("402")), gemini=gemini
        ).embed_query_con_espacio("hola")
        self.assertNotEqual(espacio, ESPACIO_E5)

    def test_el_respaldo_del_mismo_modelo_conserva_el_espacio(self):
        """DeepInfra sirve el mismo modelo: sus vectores van al espacio de e5."""
        _vector, espacio = servicio(
            principal=ProveedorFalso(error=RuntimeError("402")),
            respaldo=ProveedorFalso(0.5),
            gemini=ProveedorFalso(0.9, espacio=ESPACIO_GEMINI),
        ).embed_query_con_espacio("hola")
        self.assertEqual(espacio, ESPACIO_E5)

    def test_prefiere_el_mismo_modelo_antes_que_cambiar_de_espacio(self):
        gemini = ProveedorFalso(0.9, espacio=ESPACIO_GEMINI)
        servicio(
            principal=ProveedorFalso(error=RuntimeError("402")),
            respaldo=ProveedorFalso(0.5),
            gemini=gemini,
        ).embed_query_con_espacio("hola")
        self.assertEqual(gemini.llamadas, 0)

    def test_sin_ningun_proveedor_falla_de_forma_visible(self):
        """Sin embeddings hay que fallar, no responder sin documentos en silencio."""
        with self.assertRaises(RuntimeError):
            servicio(principal=ProveedorFalso(error=RuntimeError("402"))).embed_query_con_espacio("hola")


class TestIndexadoEnAmbosEspacios(unittest.TestCase):
    def test_genera_vectores_para_los_dos_espacios(self):
        por_espacio = servicio(
            principal=ProveedorFalso(0.1), gemini=ProveedorFalso(0.9, espacio=ESPACIO_GEMINI)
        ).embed_documents_por_espacio(["a", "b"])
        self.assertEqual(sorted(por_espacio.keys()), sorted([ESPACIO_E5, ESPACIO_GEMINI]))
        self.assertEqual(por_espacio[ESPACIO_E5][0][0], 0.1)
        self.assertEqual(por_espacio[ESPACIO_GEMINI][0][0], 0.9)

    def test_si_falta_un_modelo_se_indexa_en_el_otro(self):
        por_espacio = servicio(
            principal=ProveedorFalso(0.1),
            gemini=ProveedorFalso(error=RuntimeError("429 sin cupo"), espacio=ESPACIO_GEMINI),
        ).embed_documents_por_espacio(["a"])
        self.assertEqual(list(por_espacio.keys()), [ESPACIO_E5])

    def test_sin_ningun_modelo_no_se_indexa_a_medias(self):
        with self.assertRaises(RuntimeError):
            servicio(principal=ProveedorFalso(error=RuntimeError("402"))).embed_documents_por_espacio(["a"])

    def test_el_modelo_de_gemini_produce_1024_dimensiones(self):
        """El índice está fijado en 1024: otra medida no cabría."""
        from src.app.core.rag.embeddings import EmbeddingsGemini

        with patch("src.app.config.settings.GEMINI_API_KEY", "llave"):
            cliente = EmbeddingsGemini(api_key="llave", model="gemini-embedding-001")
        self.assertEqual(cliente.dimensiones, 1024)
        self.assertEqual(cliente.espacio, ESPACIO_GEMINI)


if __name__ == "__main__":
    unittest.main()
