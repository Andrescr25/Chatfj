#!/usr/bin/env python3
"""
ingest_docs_new.py
------------------
Sube los archivos del batch 'docs_new' a Pinecone sin afectar los vectores
ya existentes. Soporta PDF, DOCX y XLSX.

Los IDs de vectores usan el prefijo 'docs_new__' para garantizar unicidad.

Uso:
    python scripts/ingest_docs_new.py

Modo dry-run (solo muestra cuántos chunks generaría, sin subir nada):
    python scripts/ingest_docs_new.py --dry-run
"""
# NOTA: HuggingFace migró al endpoint router.huggingface.co en 2025.
# El endpoint antiguo (api-inference.huggingface.co/pipeline) devuelve 404.

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# ──────────────────────────────────────────
# Parsear argumentos
# ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ingest docs_new to Pinecone")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Simula la ingesta sin subir nada a Pinecone"
)
args = parser.parse_args()

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# Variables de entorno
# ──────────────────────────────────────────
load_dotenv("config/config.env")
load_dotenv(find_dotenv())

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatfj-legal-index")
HF_TOKEN           = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_MODEL    = "intfloat/multilingual-e5-large"

# Archivos a procesar (los que llegaron en docs_new)
DOCS_NEW_FILES = [
    "data/docs/Base de Datos Consolidada-números telefónicos.xlsx",
    "data/docs/Código procesal de familia y protección cautelar.docx",
    "data/docs/Materia agraria y ley de aguas.docx",
    "data/docs/Of-78-CONAMAJ-2026 Constancia Andrés Vargas Araya.pdf",
    "data/docs/ley de acceso a la justicia a pueblos indígenas.pdf",
]

# ID prefix para no pisar vectores existentes
ID_PREFIX = "docs_new__"

# ──────────────────────────────────────────
# Lectores de documentos
# ──────────────────────────────────────────

def read_pdf(file_path: Path) -> str:
    """Lee texto de un archivo PDF."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader = PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.error(f"Error leyendo PDF {file_path.name}: {e}")
    return text


def read_docx(file_path: Path) -> str:
    """Lee texto de un archivo Word (.docx)."""
    from docx import Document
    text = ""
    try:
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Error leyendo DOCX {file_path.name}: {e}")
    return text


def read_xlsx(file_path: Path) -> str:
    """
    Lee un Excel y convierte cada fila en un string legible.
    Formato: 'Columna1: valor | Columna2: valor | ...'
    Esto permite búsqueda semántica sobre datos de contacto.
    """
    import openpyxl
    rows_text = []
    try:
        wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
        for sheet in wb.worksheets:
            logger.info(f"  Hoja: '{sheet.title}'")
            headers = []
            first_row = True
            for row in sheet.iter_rows(values_only=True):
                # Ignorar filas completamente vacías
                if all(v is None for v in row):
                    continue
                if first_row:
                    # Primera fila = encabezados
                    headers = [str(h).strip() if h is not None else f"Col{i}" 
                                for i, h in enumerate(row)]
                    first_row = False
                    continue
                # Construir texto descriptivo de la fila
                row_parts = []
                for h, v in zip(headers, row):
                    if v is not None and str(v).strip():
                        row_parts.append(f"{h}: {str(v).strip()}")
                if row_parts:
                    rows_text.append(" | ".join(row_parts))
        wb.close()
    except Exception as e:
        logger.error(f"Error leyendo XLSX {file_path.name}: {e}")
    return "\n".join(rows_text)


def read_file(file_path: Path) -> str:
    """Dispatcher: elige el lector según la extensión."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext in (".xlsx", ".xls"):
        return read_xlsx(file_path)
    else:
        logger.warning(f"Tipo de archivo no soportado: {ext} → {file_path.name}")
        return ""


# ──────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Divide el texto en fragmentos con overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= text_len:
            break
    return chunks


# ──────────────────────────────────────────
# Embeddings (HuggingFace API)
# ──────────────────────────────────────────

# URL del nuevo router de HuggingFace (endpoint actualizado 2025)
HF_ROUTER_URL = (
    f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"
)


def get_embeddings(_, texts: list[str]) -> list | None:
    """
    Genera embeddings via HuggingFace Router API (nuevo endpoint 2025).
    El parámetro _ se ignora (compatibilidad con llamadas existentes).
    """
    import requests
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }
    for attempt in range(5):
        try:
            r = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            result = r.json()
            time.sleep(0.3)
            return result
        except Exception as e:
            logger.warning(f"Error HF Router (intento {attempt + 1}/5): {e}")
            time.sleep(10)
    logger.error("No se pudo obtener embeddings tras 5 intentos.")
    return None


def build_embeddings():
    """Placeholder — la lógica real está en get_embeddings()."""
    return None


# ──────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────

def main():
    # Validaciones
    if not PINECONE_API_KEY and not args.dry_run:
        logger.error("❌ PINECONE_API_KEY no encontrada en config/config.env")
        sys.exit(1)
    if not HF_TOKEN and not args.dry_run:
        logger.error("❌ HUGGINGFACEHUB_API_TOKEN no encontrada en config/config.env")
        sys.exit(1)

    if args.dry_run:
        logger.info("🔍 MODO DRY-RUN: no se subirá nada a Pinecone")

    # Conectar a Pinecone
    index = None
    if not args.dry_run:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"✅ Conectado a Pinecone → índice: '{PINECONE_INDEX_NAME}'")

    # Embeddings se llaman directamente via requests (ver get_embeddings)
    hf_embeddings = None  # Mantenido por compatibilidad, no se usa
    if not args.dry_run:
        logger.info(f"✅ HuggingFace Router configurado → modelo: {EMBEDDING_MODEL}")

    total_chunks_global = 0
    files_procesados = 0

    for file_str in DOCS_NEW_FILES:
        file_path = Path(file_str)

        if not file_path.exists():
            logger.warning(f"⚠️  Archivo no encontrado, se omite: {file_path}")
            continue

        logger.info(f"\n📄 Procesando: {file_path.name} ({file_path.suffix.upper()})")

        # Leer contenido
        text = read_file(file_path)
        if not text.strip():
            logger.warning(f"  ⚠️  Contenido vacío, se omite.")
            continue

        # Elegir tamaño de chunk según tipo
        # Excel: datos cortos → chunks más pequeños
        if file_path.suffix.lower() in (".xlsx", ".xls"):
            chunks = chunk_text(text, chunk_size=500, overlap=50)
        else:
            chunks = chunk_text(text, chunk_size=1000, overlap=200)

        logger.info(f"  📝 {len(chunks)} fragmentos generados")
        total_chunks_global += len(chunks)

        if args.dry_run:
            # Mostrar preview del primer chunk
            preview = chunks[0][:200].replace("\n", " ") if chunks else ""
            logger.info(f"  Preview chunk[0]: {preview}...")
            continue

        # Subir a Pinecone en lotes de 20
        batch_size = 20
        total_chunks_file = 0

        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i : i + batch_size]
            # IDs únicos: prefijo + nombre archivo (sin espacios) + posición
            safe_name = file_path.name.replace(" ", "_").replace(".", "_")
            batch_ids = [
                f"{ID_PREFIX}{safe_name}_chunk_{i + j}"
                for j in range(len(batch_texts))
            ]
            batch_metadata = [
                {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "batch": "docs_new",
                    "text": txt
                }
                for txt in batch_texts
            ]

            vectors = get_embeddings(hf_embeddings, batch_texts)
            if vectors:
                upsert_data = list(zip(batch_ids, vectors, batch_metadata))
                index.upsert(vectors=upsert_data, namespace="")
                total_chunks_file += len(vectors)
                logger.info(
                    f"  ✅ Lote subido: {len(vectors)} fragmentos "
                    f"(acumulado archivo: {total_chunks_file})"
                )
            else:
                logger.error(f"  ❌ Fallo al generar embeddings para lote {i}")

        files_procesados += 1
        logger.info(f"  ✅ '{file_path.name}' completado → {total_chunks_file} fragmentos subidos")

    # Resumen final
    logger.info("\n" + "=" * 60)
    if args.dry_run:
        logger.info(f"🔍 DRY-RUN completado.")
        logger.info(f"   Archivos a procesar: {len(DOCS_NEW_FILES)}")
        logger.info(f"   Total chunks que se generarían: {total_chunks_global}")
        logger.info("   ➡️  Ejecuta sin --dry-run para subir a Pinecone.")
    else:
        logger.info(f"🎉 Ingesta completa.")
        logger.info(f"   Archivos procesados : {files_procesados}")
        logger.info(f"   Total chunks subidos: {total_chunks_global}")
        logger.info(f"   Índice Pinecone     : {PINECONE_INDEX_NAME}")
        logger.info(f"   ID prefix usado     : '{ID_PREFIX}'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
