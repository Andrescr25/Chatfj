"""
Respaldo de embeddings.

Los embeddings son el punto más crítico del sistema: sin ellos no se indexa ni
se busca, y el asistente responde sin documentos **en silencio**, que es la
falla más peligrosa. Ocurrió de verdad: el crédito de HuggingFace se agotó y
producción pasó a responder con 0 fuentes.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core.rag.embeddings import EmbeddingService


class ProveedorFalso:
    def __init__(self, vector=None, error=None):
        self.vector = vector
        self.error = error
        self.llamadas = 0

    def embed_query(self, texto):
        self.llamadas += 1
        if self.error:
            raise self.error
        return self.vector

    def embed_documents(self, textos):
        self.llamadas += 1
        if self.error:
            raise self.error
        return [self.vector for _ in textos]


def servicio(principal, respaldo=None):
    s = EmbeddingService.__new__(EmbeddingService)
    s.client = principal
    s.respaldo = respaldo
    return s


class TestCascadaDeEmbeddings(unittest.TestCase):
    def test_usa_el_principal_cuando_responde(self):
        principal = ProveedorFalso(vector=[0.1] * 1024)
        respaldo = ProveedorFalso(vector=[0.9] * 1024)
        self.assertEqual(servicio(principal, respaldo).embed_query_sync("hola")[0], 0.1)
        self.assertEqual(respaldo.llamadas, 0)

    def test_sin_credito_pasa_al_respaldo(self):
        """El caso real: HuggingFace devolvió 402 por crédito agotado."""
        error = RuntimeError('402: "You have depleted your monthly included credits"')
        principal = ProveedorFalso(error=error)
        respaldo = ProveedorFalso(vector=[0.9] * 1024)
        self.assertEqual(servicio(principal, respaldo).embed_query_sync("hola")[0], 0.9)

    def test_una_respuesta_vacia_tambien_cede_el_turno(self):
        """HuggingFace devolvía listas vacías en vez de fallar."""
        principal = ProveedorFalso(vector=[])
        respaldo = ProveedorFalso(vector=[0.9] * 1024)
        self.assertEqual(len(servicio(principal, respaldo).embed_query_sync("hola")), 1024)

    def test_el_respaldo_también_cubre_la_indexación(self):
        principal = ProveedorFalso(error=RuntimeError("402"))
        respaldo = ProveedorFalso(vector=[0.5] * 1024)
        vectores = servicio(principal, respaldo).embed_documents(["a", "b", "c"])
        self.assertEqual(len(vectores), 3)

    def test_sin_respaldo_el_error_se_propaga(self):
        """Si no hay respaldo, el fallo debe verse, no quedar en silencio."""
        principal = ProveedorFalso(error=RuntimeError("402 sin crédito"))
        with self.assertRaises(RuntimeError):
            servicio(principal, None).embed_query_sync("hola")


class TestModeloDelRespaldo(unittest.TestCase):
    def test_el_respaldo_usa_el_mismo_modelo_del_indice(self):
        """
        Vectores de otro modelo no son comparables: las búsquedas devolverían
        resultados sin sentido contra los 7.000 fragmentos ya indexados.
        """
        from src.app.config import settings

        with patch("src.app.config.settings.EMBEDDINGS_FALLBACK_API_KEY", "llave"):
            s = EmbeddingService.__new__(EmbeddingService)
            s.respaldo = None
            s._inicializar_respaldo()
            self.assertEqual(s.respaldo.model, settings.EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
