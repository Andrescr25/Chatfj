"""
Cascada de proveedores de modelos de lenguaje.

Nace de una caída real: Gemini devolvió 429 "prepayment credits are depleted" y
el chat quedó sin responder durante horas, aunque la llave de Groq funcionaba.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core.llm.client import BaseLLM, FallbackLLM, es_error_transitorio


class ProveedorFalso(BaseLLM):
    def __init__(self, nombre, respuesta=None, error=None):
        self.nombre = nombre
        self.respuesta = respuesta
        self.error = error
        self.llamadas = 0

    async def generate_async(self, prompt, system_message=None):
        self.llamadas += 1
        if self.error:
            raise self.error
        return self.respuesta


def error_con_codigo(codigo, mensaje="fallo"):
    e = RuntimeError(mensaje)
    e.status_code = codigo
    return e


class TestClasificacionDeErrores(unittest.TestCase):
    def test_el_error_real_de_gemini_es_transitorio(self):
        """El mensaje exacto que tumbó producción."""
        error = Exception(
            "429 Your prepayment credits are depleted. Please go to AI Studio "
            "at https://ai.studio/projects to manage your project and billing."
        )
        self.assertTrue(es_error_transitorio(error))

    def test_limite_de_peticiones_es_transitorio(self):
        self.assertTrue(es_error_transitorio(error_con_codigo(429)))

    def test_errores_del_servidor_son_transitorios(self):
        for codigo in (500, 502, 503, 504):
            self.assertTrue(es_error_transitorio(error_con_codigo(codigo)), codigo)

    def test_saturacion_y_tiempos_de_espera(self):
        for mensaje in ["Model is overloaded", "Request timed out", "resource exhausted"]:
            self.assertTrue(es_error_transitorio(Exception(mensaje)), mensaje)

    def test_una_peticion_invalida_no_es_transitoria(self):
        self.assertFalse(es_error_transitorio(error_con_codigo(400, "campo inválido")))


class TestCascada(unittest.IsolatedAsyncioTestCase):
    async def test_si_el_primero_responde_no_se_llama_al_resto(self):
        primero = ProveedorFalso("primero", respuesta="respuesta buena")
        segundo = ProveedorFalso("segundo", respuesta="no debería usarse")

        cascada = FallbackLLM([primero, segundo])
        self.assertEqual(await cascada.generate_async("hola"), "respuesta buena")
        self.assertEqual(segundo.llamadas, 0)
        self.assertEqual(cascada.ultimo_usado, "primero")

    async def test_sin_cupo_responde_el_siguiente(self):
        """El escenario que pidieron: se acaba el cupo y contesta el respaldo."""
        gemini = ProveedorFalso("gemini", error=error_con_codigo(429, "quota exceeded"))
        cerebras = ProveedorFalso("cerebras", respuesta="respuesta del respaldo")

        cascada = FallbackLLM([gemini, cerebras])
        self.assertEqual(await cascada.generate_async("hola"), "respuesta del respaldo")
        self.assertEqual(gemini.llamadas, 1)
        self.assertEqual(cascada.ultimo_usado, "cerebras")

    async def test_recorre_toda_la_cascada_hasta_encontrar_uno_que_sirva(self):
        proveedores = [
            ProveedorFalso("uno", error=error_con_codigo(429)),
            ProveedorFalso("dos", error=error_con_codigo(503)),
            ProveedorFalso("tres", respuesta="al tercer intento"),
        ]
        cascada = FallbackLLM(proveedores)
        self.assertEqual(await cascada.generate_async("hola"), "al tercer intento")
        self.assertEqual([p.llamadas for p in proveedores], [1, 1, 1])

    async def test_una_respuesta_vacia_cuenta_como_fallo(self):
        vacio = ProveedorFalso("vacio", respuesta="")
        bueno = ProveedorFalso("bueno", respuesta="contenido real")
        cascada = FallbackLLM([vacio, bueno])
        self.assertEqual(await cascada.generate_async("hola"), "contenido real")

    async def test_si_todos_fallan_se_propaga_el_error(self):
        cascada = FallbackLLM([
            ProveedorFalso("uno", error=error_con_codigo(429)),
            ProveedorFalso("dos", error=error_con_codigo(500, "último fallo")),
        ])
        with self.assertRaises(RuntimeError) as ctx:
            await cascada.generate_async("hola")
        self.assertIn("último fallo", str(ctx.exception))

    async def test_un_error_no_transitorio_tampoco_deja_al_usuario_sin_respuesta(self):
        cascada = FallbackLLM([
            ProveedorFalso("uno", error=error_con_codigo(400, "petición inválida")),
            ProveedorFalso("dos", respuesta="respondió el segundo"),
        ])
        self.assertEqual(await cascada.generate_async("hola"), "respondió el segundo")

    def test_una_cascada_vacia_es_un_error_de_configuracion(self):
        with self.assertRaises(ValueError):
            FallbackLLM([])


class TestOrdenDeLaCascada(unittest.TestCase):
    def test_lee_el_orden_declarado(self):
        with patch("src.app.config.settings.LLM_CHAIN", "groq, cerebras ,gemini"):
            from src.app.config import settings
            self.assertEqual(settings.llm_chain, ["groq", "cerebras", "gemini"])

    def test_sin_cascada_declarada_usa_el_proveedor_principal(self):
        with patch("src.app.config.settings.LLM_CHAIN", ""), \
             patch("src.app.config.settings.LLM_PROVIDER", "groq"):
            from src.app.config import settings
            self.assertEqual(settings.llm_chain, ["groq"])

    def test_no_repite_proveedores(self):
        with patch("src.app.config.settings.LLM_CHAIN", "groq,groq,gemini"):
            from src.app.config import settings
            self.assertEqual(settings.llm_chain, ["groq", "gemini"])


class TestConstruccionDeLaCascada(unittest.TestCase):
    def test_omite_los_proveedores_sin_llave(self):
        """Falta la llave de uno: se sigue con los demás en vez de no arrancar."""
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN", "cerebras,groq"), \
             patch("src.app.config.settings.CEREBRAS_API_KEY", None), \
             patch("src.app.config.settings.GROQ_API_KEY", "llave-de-prueba"):
            cascada = construir_cascada()
        self.assertEqual(len(cascada.proveedores), 1)
        self.assertIn("groq", cascada.nombre)

    def test_huggingface_reusa_el_token_de_los_embeddings(self):
        """El respaldo no exige registrarse en ningún lado nuevo."""
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN", "huggingface"), \
             patch("src.app.config.settings.HUGGINGFACEHUB_API_TOKEN", "hf_token_de_prueba"):
            cascada = construir_cascada()
        proveedor = cascada.proveedores[0]
        self.assertEqual(proveedor.base_url, "https://router.huggingface.co/v1")
        self.assertEqual(proveedor.model, "zai-org/GLM-5.2")

    def test_cerebras_usa_el_modelo_configurado(self):
        """El respaldo debe apuntar a gpt-oss-120b y al endpoint de Cerebras."""
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN", "cerebras"), \
             patch("src.app.config.settings.CEREBRAS_API_KEY", "llave-cerebras"):
            cascada = construir_cascada()
        proveedor = cascada.proveedores[0]
        self.assertEqual(proveedor.model, "gpt-oss-120b")
        self.assertEqual(proveedor.base_url, "https://api.cerebras.ai/v1")
        self.assertEqual(proveedor.nombre, "cerebras:gpt-oss-120b")

    def test_sin_ninguna_llave_avisa_con_claridad(self):
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN", "cerebras,gemini"), \
             patch("src.app.config.settings.CEREBRAS_API_KEY", None), \
             patch("src.app.config.settings.GEMINI_API_KEY", None), \
             self.assertRaises(RuntimeError) as ctx:
            construir_cascada()
        self.assertIn("Ningún proveedor", str(ctx.exception))

    def test_arma_la_cascada_completa_cuando_hay_llaves(self):
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN", "groq,cerebras"), \
             patch("src.app.config.settings.CEREBRAS_API_KEY", "llave-cerebras"), \
             patch("src.app.config.settings.GROQ_API_KEY", "llave-groq"):
            cascada = construir_cascada()
        self.assertEqual(len(cascada.proveedores), 2)
        self.assertTrue(cascada.nombre.startswith("groq"))
        self.assertIn("cerebras:gpt-oss-120b", cascada.nombre)


if __name__ == "__main__":
    unittest.main()


class TestDiagnostico(unittest.IsolatedAsyncioTestCase):
    """El endpoint que dice, en segundos, cuál proveedor está fallando."""

    async def test_reporta_el_estado_de_cada_proveedor(self):
        from src.app.api.v1.endpoints.diagnostics import diagnostico_modelos
        from src.app.core.security import ROLE_ADMIN, CurrentUser

        def crear_falso(nombre):
            if nombre == "groq":
                return ProveedorFalso("groq:modelo", respuesta="ok")
            if nombre == "cerebras":
                raise ValueError("Falta la llave de API de cerebras")
            return ProveedorFalso("gemini:modelo", error=error_con_codigo(429, "quota"))

        with patch("src.app.config.settings.LLM_CHAIN", "groq,cerebras,gemini"), \
             patch("src.app.api.v1.endpoints.diagnostics._crear", side_effect=crear_falso):
            reporte = await diagnostico_modelos(
                CurrentUser(uid="u", email="a@b.cr", role=ROLE_ADMIN)
            )

        estados = {p["proveedor"]: p["estado"] for p in reporte["proveedores"]}
        self.assertEqual(estados["groq"], "funciona")
        self.assertEqual(estados["cerebras"], "sin configurar")
        self.assertEqual(estados["gemini"], "falla")
        self.assertFalse(reporte["hay_respaldo"])
        self.assertIn("1 de 3", reporte["resumen"])


class TestModeloPorEntrada(unittest.TestCase):
    """
    La cascada admite "proveedor:modelo".

    Las cuotas gratuitas se cuentan por modelo, así que encadenar dos modelos
    del mismo proveedor da un respaldo extra sin cuentas nuevas.
    """

    def test_conserva_el_modelo_tal_como_se_escribe(self):
        with patch("src.app.config.settings.LLM_CHAIN",
                   "groq, huggingface:zai-org/GLM-5.2 , gemini"):
            from src.app.config import settings
            self.assertEqual(
                settings.llm_chain,
                ["groq", "huggingface:zai-org/GLM-5.2", "gemini"],
            )

    def test_admite_el_mismo_proveedor_con_modelos_distintos(self):
        with patch("src.app.config.settings.LLM_CHAIN",
                   "huggingface:zai-org/GLM-5.2,huggingface:meta-llama/Llama-3.3-70B-Instruct"):
            from src.app.config import settings
            self.assertEqual(len(settings.llm_chain), 2)

    def test_descarta_entradas_repetidas_exactas(self):
        with patch("src.app.config.settings.LLM_CHAIN", "groq,huggingface:a/b,groq,huggingface:a/b"):
            from src.app.config import settings
            self.assertEqual(settings.llm_chain, ["groq", "huggingface:a/b"])

    def test_un_modelo_con_dos_puntos_no_se_parte_mal(self):
        """Hay identificadores como 'openai/gpt-oss-120b:free'."""
        from src.app.core.llm.factory import _partir
        self.assertEqual(
            _partir("openrouter:openai/gpt-oss-120b:free"),
            ("openrouter", "openai/gpt-oss-120b:free"),
        )

    def test_sin_modelo_usa_el_de_la_configuracion(self):
        from src.app.core.llm.factory import _partir
        self.assertEqual(_partir("groq"), ("groq", None))

    def test_construye_dos_clientes_del_mismo_proveedor(self):
        from src.app.core.llm.factory import construir_cascada

        with patch("src.app.config.settings.LLM_CHAIN",
                   "huggingface:zai-org/GLM-5.2,huggingface:meta-llama/Llama-3.3-70B-Instruct"), \
             patch("src.app.config.settings.HUGGINGFACEHUB_API_TOKEN", "hf_prueba"):
            cascada = construir_cascada()

        modelos = [p.model for p in cascada.proveedores]
        self.assertEqual(modelos, ["zai-org/GLM-5.2", "meta-llama/Llama-3.3-70B-Instruct"])


class TestValorPegadoDesdeUnPanel(unittest.TestCase):
    """
    Al pegar LLM_CHAIN en el panel de Render es fácil arrastrar comillas o un
    salto de línea. Antes eso convertía el primer proveedor en '"groq' y lo
    descartaba en silencio, dejando la cascada sin su eslabón principal.
    """

    def _cadena(self, valor):
        with patch("src.app.config.settings.LLM_CHAIN", valor):
            from src.app.config import settings
            return settings.llm_chain

    def test_ignora_comillas_envolventes(self):
        self.assertEqual(self._cadena('"groq,gemini"'), ["groq", "gemini"])

    def test_ignora_salto_de_linea_final(self):
        self.assertEqual(self._cadena("groq,gemini\n"), ["groq", "gemini"])

    def test_ignora_comillas_y_salto_juntos(self):
        self.assertEqual(self._cadena('"groq,huggingface:zai-org/GLM-5.2\n"'),
                         ["groq", "huggingface:zai-org/GLM-5.2"])

    def test_ignora_comillas_por_entrada(self):
        self.assertEqual(self._cadena("'groq' , 'gemini'"), ["groq", "gemini"])
