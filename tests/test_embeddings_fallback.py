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

from src.app.core.rag.embeddings import EmbeddingService, ErrorDeEmbeddings


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
    s.clientes_hf = [principal] if principal else []
    s.respaldo = respaldo
    s.gemini = None
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


class TestVariasLlavesDeHuggingFace(unittest.TestCase):
    """
    El crédito de HuggingFace es por cuenta, así que varias llaves dan varios
    cupos del mismo modelo. Al ser el mismo modelo, sus vectores comparten el
    espacio del índice: no hace falta reindexar ni cambiar umbrales.
    """

    def test_ordena_las_llaves_sin_repetir(self):
        with patch("src.app.config.settings.HUGGINGFACEHUB_API_TOKEN", "hf_uno"), \
             patch("src.app.config.settings.HUGGINGFACE_TOKENS_EXTRA", "hf_dos, hf_uno ,hf_tres"):
            from src.app.config import settings
            self.assertEqual(settings.huggingface_tokens, ["hf_uno", "hf_dos", "hf_tres"])

    def test_tolera_comillas_y_espacios_del_panel(self):
        with patch("src.app.config.settings.HUGGINGFACEHUB_API_TOKEN", "hf_uno"), \
             patch("src.app.config.settings.HUGGINGFACE_TOKENS_EXTRA", '"hf_dos"'):
            from src.app.config import settings
            self.assertEqual(settings.huggingface_tokens, ["hf_uno", "hf_dos"])

    def test_sin_llaves_extra_queda_solo_la_principal(self):
        with patch("src.app.config.settings.HUGGINGFACEHUB_API_TOKEN", "hf_uno"), \
             patch("src.app.config.settings.HUGGINGFACE_TOKENS_EXTRA", ""):
            from src.app.config import settings
            self.assertEqual(settings.huggingface_tokens, ["hf_uno"])

    def test_si_la_primera_llave_se_queda_sin_credito_responde_la_segunda(self):
        """El caso real: 402 en la primera, la segunda entrega el vector."""
        primera = ProveedorFalso(error=RuntimeError('402 depleted monthly credits'))
        segunda = ProveedorFalso(vector=[0.7] * 1024)
        s = servicio(primera)
        s.clientes_hf = [primera, segunda]
        self.assertEqual(s.embed_query_sync("hola")[0], 0.7)

    def test_todas_las_llaves_usan_el_mismo_espacio(self):
        """Mismo modelo, mismos vectores: no se separa el índice."""
        from src.app.core.rag.embeddings import ESPACIO_E5, ProveedorConNombre

        envuelto = ProveedorConNombre(ProveedorFalso(vector=[0.1] * 1024), "huggingface[2]", ESPACIO_E5)
        self.assertEqual(envuelto.espacio, ESPACIO_E5)



class TestConsumoDeCredito(unittest.TestCase):
    """
    El crédito de embeddings es el recurso más escaso del sistema: se agota por
    cuenta y por mes. Cada llamada de más acerca la fecha en que producción
    vuelve a responder sin documentos.
    """

    def test_la_misma_consulta_se_embebe_una_sola_vez(self):
        """
        Cada pregunta disparaba dos llamadas idénticas en paralelo: una para
        buscar documentos y otra para buscar correcciones.
        """
        principal = ProveedorFalso(vector=[0.3] * 1024)
        s = servicio(principal)
        primero = s.embed_query_sync("¿puedo denunciar a mi vecino?")
        segundo = s.embed_query_sync("¿puedo denunciar a mi vecino?")
        self.assertEqual(primero, segundo)
        self.assertEqual(principal.llamadas, 1, "la segunda búsqueda gastó crédito otra vez")

    def test_preguntas_distintas_no_comparten_vector(self):
        principal = ProveedorFalso(vector=[0.3] * 1024)
        s = servicio(principal)
        s.embed_query_sync("pensión alimentaria")
        s.embed_query_sync("apremio corporal")
        self.assertEqual(principal.llamadas, 2)

    def test_la_llave_sin_credito_se_aparta_y_no_se_reintenta(self):
        """
        Un 402 dura hasta el próximo ciclo mensual. Reintentarlo en cada
        consulta gasta tiempo y llena los registros de errores engañosos.
        """
        agotada = ProveedorFalso(error=ErrorDeEmbeddings("402 depleted", 402))
        buena = ProveedorFalso(vector=[0.7] * 1024)
        s = servicio(agotada)
        s.clientes_hf = [agotada, buena]

        s.embed_query_sync("primera pregunta")
        s.embed_query_sync("segunda pregunta")

        self.assertEqual(agotada.llamadas, 1, "se volvió a intentar la llave agotada")
        self.assertEqual(buena.llamadas, 2)

    def test_un_fallo_pasajero_no_aparta_la_llave(self):
        """Un 503 se resuelve solo: la llave debe seguir en el turno."""
        intermitente = ProveedorFalso(error=ErrorDeEmbeddings("503 modelo cargando", 503))
        buena = ProveedorFalso(vector=[0.7] * 1024)
        s = servicio(intermitente)
        s.clientes_hf = [intermitente, buena]

        s.embed_query_sync("primera pregunta")
        s.embed_query_sync("segunda pregunta")

        self.assertEqual(intermitente.llamadas, 2)

    def test_si_todas_estan_apartadas_igual_se_intenta(self):
        """Más vale reintentar una llave dudosa que responder sin documentos."""
        agotada = ProveedorFalso(error=ErrorDeEmbeddings("402 depleted", 402))
        s = servicio(agotada)
        with self.assertRaises(ErrorDeEmbeddings):
            s.embed_query_sync("primera")
        with self.assertRaises(ErrorDeEmbeddings):
            s.embed_query_sync("segunda")
        self.assertEqual(agotada.llamadas, 2)


class TestErrorDeHuggingFace(unittest.TestCase):
    def test_un_402_no_se_reintenta_tres_veces(self):
        """Reintentar un crédito agotado multiplica la espera sin ganar nada."""
        from src.app.core.rag.embeddings import SafeHuggingFaceEmbeddings

        cliente = SafeHuggingFaceEmbeddings(api_key="hf_prueba", model_name="modelo/x")

        class RespuestaFalsa:
            status_code = 402
            text = '{"error":"You have depleted your monthly included credits."}'
            headers = {}

        with patch("src.app.core.rag.embeddings.requests.post", return_value=RespuestaFalsa()) as post, \
             self.assertRaises(ErrorDeEmbeddings) as capturado:
            cliente.embed_documents(["hola"])

        self.assertEqual(capturado.exception.status_code, 402)
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
