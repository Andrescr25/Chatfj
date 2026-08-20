"""
Toda dependencia que el código importa debe estar declarada.

Nace de una caída real: pydantic-settings llegaba de rebote como dependencia de
otro paquete. Al cambiar la versión de Python en Render, la instalación se
resolvió distinto, esa dependencia dejó de llegar y el servicio no levantó.
"""
import ast
import sys
import unittest
from importlib.metadata import packages_distributions
from pathlib import Path

RAIZ = Path(__file__).parent.parent


class TestDependenciasDeclaradas(unittest.TestCase):
    def test_todo_lo_que_se_importa_esta_en_requirements(self):
        requisitos = (RAIZ / "requirements.txt").read_text().lower()
        mapa = packages_distributions()
        estandar = set(sys.stdlib_module_names)

        importados = set()
        for archivo in list((RAIZ / "src").rglob("*.py")) + list((RAIZ / "scripts").rglob("*.py")):
            for nodo in ast.walk(ast.parse(archivo.read_text())):
                if isinstance(nodo, ast.Import):
                    importados.update(n.name.split(".")[0] for n in nodo.names)
                elif isinstance(nodo, ast.ImportFrom) and nodo.level == 0 and nodo.module:
                    importados.add(nodo.module.split(".")[0])

        faltantes = []
        for modulo in sorted(importados):
            if modulo in estandar or modulo == "src":
                continue
            distribuciones = mapa.get(modulo, [])
            if distribuciones and not any(
                d.lower().replace("_", "-") in requisitos for d in distribuciones
            ):
                faltantes.append(f"{modulo} (paquete: {distribuciones[0]})")

        self.assertEqual(
            faltantes, [],
            "Estos módulos se importan pero no están en requirements.txt: "
            + ", ".join(faltantes),
        )


if __name__ == "__main__":
    unittest.main()
