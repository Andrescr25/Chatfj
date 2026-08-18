"""
Tests de lectura de documentos.

El objetivo principal es que la ruta de PDF esté cubierta: el panel falló en
producción con "No module named 'PyPDF2'" porque el código importaba una
librería que el proyecto ya no declara, y las pruebas solo ejercitaban texto
plano.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.core.rag.loaders import SUPPORTED_EXTENSIONS, read_file


def pdf_minimo(texto: str) -> bytes:
    """PDF válido de una página, para no depender de archivos fuera del repositorio."""
    flujo = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("latin-1")
    objetos = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(flujo)).encode() + b" >>\nstream\n" + flujo + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    salida = bytearray(b"%PDF-1.4\n")
    posiciones = []
    for i, obj in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += f"xref\n0 {len(objetos) + 1}\n".encode() + b"0000000000 65535 f \n"
    for pos in posiciones:
        salida += f"{pos:010d} 00000 n \n".encode()
    salida += (
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\n"
        f"startxref\n{inicio_xref}\n%%EOF\n"
    ).encode()
    return bytes(salida)


class TestLectorDePdf(unittest.TestCase):
    def test_la_libreria_de_pdf_esta_disponible(self):
        """Falla si el entorno no trae ninguna de las dos librerías soportadas."""
        try:
            from pypdf import PdfReader  # noqa: F401
        except ImportError:
            from PyPDF2 import PdfReader  # noqa: F401

    def test_extrae_el_texto_de_un_pdf(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_minimo("Orientacion legal para facilitadores judiciales"))
            ruta = Path(tmp.name)
        try:
            texto = read_file(ruta)
            self.assertIn("facilitadores judiciales", texto)
        finally:
            ruta.unlink(missing_ok=True)

    def test_pdf_ilegible_no_revienta(self):
        """Un archivo corrupto devuelve vacío; el servicio lo reporta como error."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"esto no es un PDF")
            ruta = Path(tmp.name)
        try:
            self.assertEqual(read_file(ruta).strip(), "")
        finally:
            ruta.unlink(missing_ok=True)


class TestOtrosFormatos(unittest.TestCase):
    def test_lee_texto_plano(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write("Pensión alimentaria: el trámite es gratuito.")
            ruta = Path(tmp.name)
        try:
            self.assertIn("gratuito", read_file(ruta))
        finally:
            ruta.unlink(missing_ok=True)

    def test_extension_no_soportada_devuelve_vacio(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
            tmp.write(b"binario")
            ruta = Path(tmp.name)
        try:
            self.assertEqual(read_file(ruta), "")
        finally:
            ruta.unlink(missing_ok=True)

    def test_todas_las_extensiones_admitidas_tienen_lector(self):
        """Ninguna extensión ofrecida en el panel debe quedar sin lector."""
        for ext in SUPPORTED_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(pdf_minimo("x") if ext == ".pdf" else b"contenido")
                ruta = Path(tmp.name)
            try:
                # No debe lanzar excepción con ninguna extensión declarada
                read_file(ruta)
            finally:
                ruta.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
