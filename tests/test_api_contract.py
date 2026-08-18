"""
El cliente y el servidor deben hablar de las mismas rutas.

Durante meses el frontend llamó a siete rutas que el servidor no tenía
(/training/ask, /training/feedback, /ask/stream...), heredadas de una versión
anterior. Nadie se enteró porque fallaban en silencio. Esta prueba compara lo
que declara FastAPI contra lo que invoca src/api/client.js.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAIZ = Path(__file__).parent.parent
CLIENTE = RAIZ / "frontend" / "src" / "api" / "client.js"

PARAMETRO = re.compile(r"\{[^}]+\}|\$\{[^}]+\}")


def normalizar(ruta: str) -> str:
    """/documents/{doc_id}/content y /documents/${x}/content son la misma ruta."""
    ruta = ruta.split("?")[0].rstrip("/")
    return PARAMETRO.sub("{param}", ruta)


def rutas_del_servidor() -> set:
    """
    Se leen del esquema OpenAPI y no de app.routes: FastAPI ya no aplana los
    routers incluidos dentro de app.routes, así que recorrerlo devolvía vacío.
    """
    from src.app.config import settings
    from src.app.main import app

    prefijo = settings.API_V1_STR
    return {
        normalizar(camino[len(prefijo):])
        for camino in app.openapi().get("paths", {})
        if camino.startswith(prefijo)
    }


def rutas_del_cliente() -> set:
    texto = CLIENTE.read_text(encoding="utf-8")
    rutas = set()
    # this.request('/documents') y this.request(`/documents/${id}/content`)
    for m in re.finditer(r"this\.request\(\s*[`'\"]([^`'\"]+)", texto):
        rutas.add(normalizar(m.group(1)))
    # fetch(`${this.baseURL}/documents`)
    for m in re.finditer(r"\$\{this\.baseURL\}([^`'\"]+)", texto):
        rutas.add(normalizar(m.group(1)))
    return {r for r in rutas if r.startswith("/")}


class TestContratoDeApi(unittest.TestCase):
    def test_el_cliente_declara_rutas(self):
        """Si el análisis deja de encontrar rutas, la prueba dejaría de proteger."""
        self.assertGreaterEqual(len(rutas_del_cliente()), 5)

    def test_toda_ruta_del_cliente_existe_en_el_servidor(self):
        servidor = rutas_del_servidor()
        inexistentes = sorted(rutas_del_cliente() - servidor)
        self.assertEqual(
            inexistentes, [],
            f"El frontend llama a rutas que el servidor no tiene: {inexistentes}. "
            f"Rutas disponibles: {sorted(servidor)}",
        )


if __name__ == "__main__":
    unittest.main()
