#!/usr/bin/env python3
"""
Procesador de PDFs Legales de Costa Rica
Convierte documentos jurídicos en bloques estructurados para indexación vectorial
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2


class LegalDocumentProcessor:
    """Procesador especializado en documentos legales costarricenses."""

    # Categorías legales reconocidas
    CATEGORIES = {
        "violencia": [
            "violencia doméstica", "violencia intrafamiliar", "agresión",
            "maltrato", "medidas de protección", "orden de alejamiento"
        ],
        "pension_alimentaria": [
            "pensión alimentaria", "pensiones alimentarias", "cuota alimentaria",
            "apremio corporal", "obligación alimentaria"
        ],
        "menores": [
            "niñez", "adolescencia", "menor", "pani", "patria potestad",
            "derechos del niño", "protección integral"
        ],
        "laboral": [
            "trabajo", "laboral", "empleador", "trabajador", "despido",
            "salario", "jornada", "horas extra", "cesantía"
        ],
        "civil": [
            "arrendamiento", "desalojo", "contrato", "obligaciones",
            "propiedad", "inquilino", "arrendador"
        ],
        "penal": [
            "delito", "pena", "prisión", "contravención", "denuncia penal",
            "organismo de investigación"
        ],
        "migracion": [
            "migración", "extranjero", "refugiado", "asilo",
            "residencia", "visa"
        ],
        "conciliacion": [
            "conciliación", "mediación", "facilitador judicial",
            "resolución alterna", "arreglo amistoso"
        ],
        "constitucional": [
            "constitucional", "amparo", "habeas corpus", "sala constitucional",
            "derechos fundamentales"
        ]
    }

    # Patrones de leyes conocidas
    KNOWN_LAWS = {
        "7586": {"nombre": "Ley contra la Violencia Doméstica", "categoria": "violencia"},
        "7654": {"nombre": "Ley de Pensiones Alimentarias", "categoria": "pension_alimentaria"},
        "7739": {"nombre": "Código de la Niñez y la Adolescencia", "categoria": "menores"},
        "7600": {"nombre": "Ley de Igualdad de Oportunidades para Personas con Discapacidad", "categoria": "civil"},
        "7935": {"nombre": "Ley Integral para la Persona Adulta Mayor", "categoria": "civil"},
    }

    def __init__(self):
        self.current_law = None
        self.current_category = "desconocida"

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extrae texto de PDF con información de página."""
        pages_data = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages_data.append({
                            "page": page_num,
                            "text": text
                        })

        except Exception as e:
            print(f"Error leyendo PDF {pdf_path}: {e}", file=sys.stderr)
            return []

        return pages_data

    def clean_text(self, text: str) -> str:
        """Limpia texto de artefactos de OCR y formato."""
        # Eliminar encabezados comunes
        text = re.sub(r'Página \d+( de \d+)?', '', text)
        text = re.sub(r'Sistema Costarricense de Información Jurídica', '', text)
        text = re.sub(r'www\.pgrweb\.go\.cr', '', text, flags=re.IGNORECASE)

        # Eliminar saltos de línea dentro de párrafos
        text = re.sub(r'([a-zñáéíóú])\s*\n\s*([a-zñáéíóú])', r'\1 \2', text)

        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Corregir errores comunes de OCR
        text = text.replace('Articulo', 'Artículo')
        text = text.replace('Art.', 'Artículo')
        text = text.replace('articulo', 'artículo')

        return text.strip()

    def detect_law_info(self, text: str) -> Optional[Dict[str, str]]:
        """Detecta información de la ley en el texto."""
        # Buscar número de ley
        ley_match = re.search(r'LEY\s+N[°º\.]*\s*(\d+)', text, re.IGNORECASE)
        if ley_match:
            ley_num = ley_match.group(1)
            if ley_num in self.KNOWN_LAWS:
                return {
                    "numero": ley_num,
                    "nombre": self.KNOWN_LAWS[ley_num]["nombre"],
                    "categoria": self.KNOWN_LAWS[ley_num]["categoria"]
                }
            return {"numero": ley_num, "nombre": f"Ley {ley_num}", "categoria": None}

        # Buscar códigos
        if "código" in text.lower():
            if "niñez" in text.lower() or "adolescencia" in text.lower():
                return {"numero": "7739", "nombre": "Código de la Niñez y Adolescencia", "categoria": "menores"}
            if "trabajo" in text.lower():
                return {"numero": "codigo_trabajo", "nombre": "Código de Trabajo", "categoria": "laboral"}
            if "civil" in text.lower():
                return {"numero": "codigo_civil", "nombre": "Código Civil", "categoria": "civil"}
            if "penal" in text.lower():
                return {"numero": "codigo_penal", "nombre": "Código Penal", "categoria": "penal"}

        return None

    def classify_text(self, text: str) -> str:
        """Clasifica texto en una categoría legal."""
        text_lower = text.lower()

        # Contar coincidencias por categoría
        scores = {}
        for category, keywords in self.CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)

        return "desconocida"

    def segment_by_articles(self, text: str, filename: str, page_num: int) -> List[Dict]:
        """Segmenta texto por artículos."""
        blocks = []

        # Patrón para detectar artículos
        article_pattern = r'(Artículo|ARTÍCULO)\s+(\d+)[°\.]*\s*[—\-–]?\s*(.+?)(?=(?:Artículo|ARTÍCULO)\s+\d+|$)'

        matches = list(re.finditer(article_pattern, text, re.IGNORECASE | re.DOTALL))

        if matches:
            for match in matches:
                article_num = match.group(2)
                article_text = match.group(3).strip()

                # Limpiar texto del artículo
                article_text = self.clean_text(article_text)

                # Limitar tamaño (máximo 500 tokens ≈ 2000 caracteres)
                if len(article_text) > 2000:
                    article_text = article_text[:2000] + "..."

                # Clasificar
                category = self.classify_text(article_text)

                block = {
                    "ley": self.current_law if self.current_law else "Desconocida",
                    "articulo": f"Artículo {article_num}",
                    "texto": article_text,
                    "categoria": category if category != "desconocida" else self.current_category,
                    "documento": filename,
                    "pagina": page_num
                }
                blocks.append(block)
        else:
            # Si no hay artículos, segmentar por párrafos o tamaño
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                para = self.clean_text(para)
                if len(para) > 100:  # Mínimo 100 caracteres
                    category = self.classify_text(para)
                    block = {
                        "ley": self.current_law if self.current_law else "Desconocida",
                        "articulo": "",
                        "texto": para[:2000],  # Limitar tamaño
                        "categoria": category if category != "desconocida" else self.current_category,
                        "documento": filename,
                        "pagina": page_num
                    }
                    blocks.append(block)

        return blocks

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Procesa un PDF completo y retorna bloques estructurados."""
        pdf_path = Path(pdf_path)
        filename = pdf_path.name

        print(f"📄 Procesando: {filename}", file=sys.stderr)

        # Extraer texto por páginas
        pages_data = self.extract_text_from_pdf(str(pdf_path))

        if not pages_data:
            print(f"⚠️  No se pudo extraer texto de {filename}", file=sys.stderr)
            return []

        all_blocks = []

        # Detectar información de ley en primera página
        first_page_text = pages_data[0]["text"]
        law_info = self.detect_law_info(first_page_text)

        if law_info:
            self.current_law = law_info.get("nombre", "Desconocida")
            self.current_category = law_info.get("categoria", "desconocida")
            print(f"   📖 Ley detectada: {self.current_law} (Categoría: {self.current_category})", file=sys.stderr)

        # Procesar cada página
        for page_data in pages_data:
            page_num = page_data["page"]
            text = page_data["text"]

            # Limpiar texto
            clean_text = self.clean_text(text)

            # Segmentar por artículos
            blocks = self.segment_by_articles(clean_text, filename, page_num)
            all_blocks.extend(blocks)

        print(f"   ✅ {len(all_blocks)} bloques extraídos", file=sys.stderr)
        return all_blocks

    def process_directory(self, directory_path: str) -> List[Dict]:
        """Procesa todos los PDFs en un directorio."""
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠️  No se encontraron PDFs en {directory_path}", file=sys.stderr)
            return []

        print(f"📁 Procesando {len(pdf_files)} archivos PDF...\n", file=sys.stderr)

        all_blocks = []
        for pdf_file in pdf_files:
            blocks = self.process_pdf(str(pdf_file))
            all_blocks.extend(blocks)

        return all_blocks


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python3 process_legal_pdf.py <archivo.pdf|directorio>", file=sys.stderr)
        print("\nEjemplos:", file=sys.stderr)
        print("  python3 process_legal_pdf.py Ley_7586.pdf", file=sys.stderr)
        print("  python3 process_legal_pdf.py data/docs/", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    path = Path(input_path)

    processor = LegalDocumentProcessor()

    # Procesar
    if path.is_file() and path.suffix == '.pdf':
        blocks = processor.process_pdf(str(path))
    elif path.is_dir():
        blocks = processor.process_directory(str(path))
    else:
        print(f"❌ Error: {input_path} no es un PDF válido ni un directorio", file=sys.stderr)
        sys.exit(1)

    # Salida JSON (uno por línea)
    for block in blocks:
        print(json.dumps(block, ensure_ascii=False))

    print(f"\n✅ Total: {len(blocks)} bloques generados", file=sys.stderr)


if __name__ == "__main__":
    main()
