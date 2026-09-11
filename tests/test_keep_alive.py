"""
Latido que mantiene despierto el servicio en Render.

El keep-alive anterior era un flujo programado de GitHub Actions: cada 10
minutos sobre el papel, una vez cada dos horas en la práctica. Todas sus
ejecuciones figuraban como exitosas mientras el servicio pasaba dormido la mayor
parte del día. Un keep-alive que falla en silencio es peor que ninguno, porque
nadie lo revisa.
"""
import asyncio
import sys
import unittest
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core import keep_alive

URL = "https://chatfj.onrender.com"


def hora_cr(hora, minuto=0):
    """Instante en UTC que corresponde a esa hora de Costa Rica."""
    return datetime(2026, 9, 10, hora, minuto, tzinfo=keep_alive.ZONA_CR).astimezone(timezone.utc)


class TestHorario(unittest.TestCase):
    def test_24h_mantiene_despierto_a_toda_hora(self):
        franja = keep_alive.interpretar_horario("24h")
        self.assertTrue(all(keep_alive.toca_mantener_despierto(franja, hora_cr(h)) for h in range(24)))

    def test_vacio_equivale_a_24h(self):
        self.assertEqual(keep_alive.interpretar_horario(""), keep_alive.SIEMPRE)

    def test_tolera_comillas_y_mayusculas_del_panel(self):
        self.assertEqual(keep_alive.interpretar_horario(' "OFF" '), None)

    def test_off_apaga_el_latido(self):
        self.assertIsNone(keep_alive.interpretar_horario("off"))

    def test_franja_diurna_en_hora_de_costa_rica(self):
        franja = keep_alive.interpretar_horario("7-22")
        self.assertFalse(keep_alive.toca_mantener_despierto(franja, hora_cr(6, 59)))
        self.assertTrue(keep_alive.toca_mantener_despierto(franja, hora_cr(7)))
        self.assertTrue(keep_alive.toca_mantener_despierto(franja, hora_cr(21, 59)))
        self.assertFalse(keep_alive.toca_mantener_despierto(franja, hora_cr(22)))

    def test_franja_que_cruza_la_medianoche(self):
        franja = keep_alive.interpretar_horario("22-6")
        self.assertTrue(keep_alive.toca_mantener_despierto(franja, hora_cr(23)))
        self.assertTrue(keep_alive.toca_mantener_despierto(franja, hora_cr(3)))
        self.assertFalse(keep_alive.toca_mantener_despierto(franja, hora_cr(12)))

    def test_un_valor_invalido_no_deja_dormir_el_servicio(self):
        """Un error de tipeo en el panel de Render no debe apagar el latido en silencio."""
        with self.assertLogs(keep_alive.logger, level="WARNING"):
            self.assertEqual(keep_alive.interpretar_horario("7a22"), keep_alive.SIEMPRE)

    def test_el_intervalo_nunca_alcanza_los_15_minutos_de_render(self):
        self.assertLess(keep_alive.intervalo_seguro(3600), 15 * 60)


class TestLatido(unittest.TestCase):
    def correr(self, franja, respuestas, vueltas, hora=12):
        consultas, esperas = [], []

        def consultar(url):
            consultas.append(url)
            respuesta = respuestas.pop(0)
            if isinstance(respuesta, Exception):
                raise respuesta
            return respuesta

        async def esperar(segundos):
            esperas.append(segundos)

        asyncio.run(keep_alive.latir(
            URL, franja, 600,
            consultar=consultar, ahora=lambda: hora_cr(hora), esperar=esperar, vueltas=vueltas,
        ))
        return consultas, esperas

    def test_late_una_vez_por_intervalo(self):
        consultas, esperas = self.correr(keep_alive.SIEMPRE, [200, 200, 200], vueltas=3)
        self.assertEqual(len(consultas), 3)
        self.assertEqual(esperas, [600, 600, 600])

    def test_consulta_health_por_la_direccion_publica(self):
        """
        Por localhost Render no vería el tráfico. Y /robots.txt tampoco sirve:
        Render lo contesta por su cuenta mientras el servicio duerme.
        """
        with patch("src.app.core.keep_alive.requests.get") as get:
            get.return_value.status_code = 200
            keep_alive.consultar_salud(URL + "/")
        self.assertEqual(get.call_args.args[0], URL + "/health")

    def test_tras_un_fallo_reintenta_al_minuto(self):
        """
        Esperar otros 10 minutos tras un fallo dejaría pasar 20 sin tráfico, y
        Render duerme el servicio a los 15.
        """
        _, esperas = self.correr(keep_alive.SIEMPRE, [ConnectionError("sin red"), 200, 200], vueltas=3)
        self.assertEqual(esperas, [600, 60, 600])

    def test_un_fallo_no_detiene_el_latido(self):
        consultas, _ = self.correr(keep_alive.SIEMPRE, [ConnectionError("sin red"), 503, 200], vueltas=3)
        self.assertEqual(len(consultas), 3)

    def test_fuera_de_horario_no_consulta(self):
        consultas, _ = self.correr((7, 22), [], vueltas=3, hora=3)
        self.assertEqual(consultas, [])

    def test_registra_los_cambios_y_no_cada_latido(self):
        """Visible cuando empieza o se recupera, sin llenar los registros cada 10 minutos."""
        with self.assertLogs(keep_alive.logger, level="INFO") as registros:
            self.correr(keep_alive.SIEMPRE, [200, 200, 200], vueltas=3)
        self.assertEqual(sum("respondió 200" in r for r in registros.output), 1)


class TestArranque(unittest.TestCase):
    def test_fuera_de_render_no_arranca(self):
        """En local y en CI no hay RENDER_EXTERNAL_URL: no debe consultarse a sí mismo."""
        self.assertIsNone(keep_alive.iniciar(None, "24h", 600))

    def test_apagado_por_configuracion_no_arranca(self):
        self.assertIsNone(keep_alive.iniciar(URL, "off", 600))

    def test_en_render_arranca_una_tarea(self):
        async def probar():
            tarea = keep_alive.iniciar(URL, "24h", 600)
            self.assertIsInstance(tarea, asyncio.Task)
            tarea.cancel()
            with suppress(asyncio.CancelledError):
                await tarea

        asyncio.run(probar())


class TestSalud(unittest.TestCase):
    def test_health_dice_desde_cuando_esta_activo(self):
        """Si 'activo_desde' cambia entre dos consultas, el servicio se durmió o se reinició."""
        from src.app.main import health_check

        datos = health_check()
        self.assertEqual(datos["status"], "ok")
        self.assertIn("activo_desde", datos)
        self.assertGreaterEqual(datos["segundos_activo"], 0)


if __name__ == "__main__":
    unittest.main()
