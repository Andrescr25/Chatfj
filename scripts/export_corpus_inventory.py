#!/usr/bin/env python3
"""
Genera el inventario público de documentos indexados (docs/corpus.md).

Los archivos del corpus no se versionan (pesan), así que este inventario es el
registro público de con qué información se entrenó el asistente. Se lee del
catálogo en Firestore, que es lo que el sistema realmente tiene indexado, no de
lo que haya en una carpeta.

Uso:
    python scripts/export_corpus_inventory.py
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import firebase_admin
from firebase_admin import credentials

from src.app.config import settings

SALIDA = Path("docs/corpus.md")


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    if os.path.exists(cred_path):
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    else:
        firebase_admin.initialize_app()


def fecha_corta(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
    except Exception:
        return "—"


def main() -> None:
    init_firebase()
    from src.app.core.registry import DocumentRegistry

    documentos = sorted(
        DocumentRegistry().list(),
        key=lambda d: (d.get("filename") or "").lower(),
    )
    if not documentos:
        print("El catálogo está vacío. ¿Ejecutó scripts/backfill_documents_registry.py?")
        sys.exit(1)

    total_fragmentos = sum(d.get("chunks", 0) or 0 for d in documentos)
    hoy = datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y")

    lineas = [
        "# Inventario del corpus",
        "",
        "Documentos oficiales con los que responde Chat FJ. El asistente no",
        "inventa normativa: recupera fragmentos de estos documentos y responde",
        "a partir de ellos.",
        "",
        f"**{len(documentos)} documentos · {total_fragmentos:,} fragmentos indexados · "
        f"actualizado el {hoy}**".replace(",", " "),
        "",
        "Los archivos en sí no se versionan por su peso. Este inventario se genera",
        "del catálogo real del sistema con `python scripts/export_corpus_inventory.py`.",
        "",
        "| Documento | Materia | Fragmentos | Incorporado |",
        "|---|---|---:|---|",
    ]

    for d in documentos:
        origen = "ingesta inicial" if d.get("legacy") else fecha_corta(d.get("uploaded_at", ""))
        lineas.append(
            f"| {d.get('filename', '—')} "
            f"| {d.get('category', 'general')} "
            f"| {d.get('chunks', 0):,} ".replace(",", " ")
            + f"| {origen} |"
        )

    lineas += [
        "",
        "## Qué no contiene",
        "",
        "El corpus son documentos públicos y oficiales. No incluye expedientes,",
        "datos personales de personas usuarias ni información confidencial.",
        "",
    ]

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text("\n".join(lineas), encoding="utf-8")
    print(f"✅ {SALIDA}: {len(documentos)} documentos, {total_fragmentos} fragmentos")


if __name__ == "__main__":
    main()
