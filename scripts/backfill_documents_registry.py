#!/usr/bin/env python3
"""
Reconcilia el catálogo de documentos con lo que ya está en Pinecone.

Los documentos indexados antes del panel no tienen registro: sin este paso el
panel arrancaría vacío y esos documentos serían invisibles e imposibles de
eliminar desde la interfaz.

El script recorre los IDs de vectores del índice, los agrupa por prefijo
('{archivo}_chunk_{n}' y 'docs_new__{archivo}_chunk_{n}') y crea un registro por
documento con su conteo real de fragmentos.

Uso:
    python scripts/backfill_documents_registry.py --dry-run
    python scripts/backfill_documents_registry.py
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import firebase_admin
from firebase_admin import credentials
from pinecone import Pinecone

from src.app.config import settings
from src.app.core.registry import STATUS_INDEXED, DocumentRegistry, utcnow_iso
from src.app.services.document_service import slugify

# Patrón principal de la ingesta original: '{archivo}_chunk_{n}'
CHUNK_ID_RE = re.compile(r"^(?P<prefix>.+_chunk_)(?P<n>\d+)$")
# Respaldo para lotes que usaron nombres cortos: '{prefijo}_{n}'
GENERIC_ID_RE = re.compile(r"^(?P<prefix>.+_)(?P<n>\d+)$")
DOCS_NEW_PREFIX = "docs_new__"


def init_firebase():
    if firebase_admin._apps:
        return
    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    try:
        if os.path.exists(cred_path):
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        else:
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"⚠️ Firebase no disponible ({e}). El catálogo se guardará en archivo local.")


def nombre_desde_prefijo(prefix: str) -> str:
    """'Codigo_Civil.pdf_chunk_' -> 'Codigo_Civil.pdf'"""
    base = prefix[: -len("_chunk_")] if prefix.endswith("_chunk_") else prefix.rstrip("_")
    if base.startswith(DOCS_NEW_PREFIX):
        base = base[len(DOCS_NEW_PREFIX):]
        # ingest_docs_new sustituyó los puntos por guiones bajos al construir el ID
        for ext in ("_pdf", "_docx", "_xlsx", "_txt", "_md"):
            if base.endswith(ext):
                base = base[: -len(ext)] + "." + ext[1:]
                break
    return base


def main():
    parser = argparse.ArgumentParser(description="Reconciliar catálogo de documentos con Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra lo que haría")
    args = parser.parse_args()

    if not settings.PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY no configurada.")
        sys.exit(1)

    init_firebase()

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    print(f"🌲 Índice: {settings.PINECONE_INDEX_NAME}")

    grupos = defaultdict(list)
    sin_patron = []
    total_ids = 0

    try:
        for page in index.list(namespace=""):
            ids = [page] if isinstance(page, str) else page
            for vid in ids:
                total_ids += 1
                m = CHUNK_ID_RE.match(vid) or GENERIC_ID_RE.match(vid)
                if m:
                    grupos[m.group("prefix")].append(int(m.group("n")))
                else:
                    sin_patron.append(vid)
    except Exception as e:
        print(f"❌ No se pudieron listar los IDs del índice: {e}")
        print("   (Los índices pod-based no admiten listado por prefijo.)")
        sys.exit(1)

    print(f"🔍 {total_ids} vectores revisados → {len(grupos)} documentos detectados")
    if sin_patron:
        print(f"⚠️ {len(sin_patron)} vectores no siguen el patrón conocido y quedan fuera del catálogo:")
        for vid in sin_patron[:10]:
            print(f"     - {vid}")
        if len(sin_patron) > 10:
            print(f"     ... y {len(sin_patron) - 10} más")

    registry = DocumentRegistry()
    print(f"📇 Catálogo: {registry.backend}")

    existentes = {r.get("vector_prefix") for r in registry.list(include_deleted=True)}
    creados = 0
    omitidos = 0

    for prefix, indices in sorted(grupos.items()):
        if prefix in existentes:
            omitidos += 1
            continue

        filename = nombre_desde_prefijo(prefix)
        doc_id = f"{slugify(filename)}-legacy"

        # Nombre real del archivo desde la metadata, si está disponible
        source = filename
        try:
            fetched = index.fetch(ids=[f"{prefix}{min(indices)}"], namespace="")
            vectors = getattr(fetched, "vectors", None) or {}
            for _, v in vectors.items():
                md = getattr(v, "metadata", None) or {}
                source = md.get("filename") or Path(str(md.get("source", filename))).name or filename
        except Exception:
            pass

        record = {
            "doc_id": doc_id,
            "filename": source,
            "title": Path(source).stem,
            "category": "general",
            "extension": Path(source).suffix.lower(),
            "size_bytes": 0,
            "file_hash": "",
            "vector_prefix": prefix,
            "chunks": len(indices),
            "chunks_total": max(indices) + 1,
            "status": STATUS_INDEXED,
            "error": "",
            "uploaded_by": "ingesta inicial (línea de comandos)",
            "uploaded_at": utcnow_iso(),
            "updated_at": utcnow_iso(),
            "indexed_at": utcnow_iso(),
            "legacy": True,
            "storage_backend": "",
            "storage_path": "",
        }

        print(f"  + {source:60.60s} {len(indices):5d} fragmentos")
        if not args.dry_run:
            registry.save(record)
        creados += 1

    print()
    if args.dry_run:
        print(f"🔎 Simulación: se crearían {creados} registros ({omitidos} ya existían).")
    else:
        print(f"✅ Catálogo actualizado: {creados} documentos registrados ({omitidos} ya existían).")
        print("   Los documentos heredados no tienen archivo original guardado:")
        print("   se pueden eliminar desde el panel, pero no reindexar.")


if __name__ == "__main__":
    main()
