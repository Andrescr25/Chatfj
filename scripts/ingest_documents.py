#!/usr/bin/env python3
"""
Ingesta de documentos desde la línea de comandos.

Usa exactamente el mismo servicio que el panel de administración
(src/app/services/document_service.py), así que un documento subido por aquí
queda igual que uno subido desde la web: mismo troceado, mismo esquema de IDs,
registro en el catálogo y original respaldado.

Antes había dos scripts con su propia copia de los lectores y de la lógica de
troceado. Esa duplicación fue la que dejó pasar el fallo de PyPDF2.

Uso:
    python scripts/ingest_documents.py data/docs                 # una carpeta
    python scripts/ingest_documents.py archivo.pdf otro.docx     # archivos sueltos
    python scripts/ingest_documents.py data/docs --dry-run       # solo listar
    python scripts/ingest_documents.py data/docs --categoria familia
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import firebase_admin
from firebase_admin import credentials

from src.app.config import settings
from src.app.core.rag.loaders import SUPPORTED_EXTENSIONS
from src.app.core.security import ROLE_ADMIN, CurrentUser


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    try:
        if os.path.exists(cred_path):
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        else:
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"⚠️  Firebase no disponible ({e}). El catálogo se guardará en archivo local.")


def reunir_archivos(rutas: list[str]) -> list[Path]:
    archivos: list[Path] = []
    for ruta in rutas:
        p = Path(ruta)
        if p.is_dir():
            archivos += [
                f for f in sorted(p.rglob("*"))
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        elif p.is_file():
            archivos.append(p)
        else:
            print(f"⚠️  No existe: {ruta}")
    return archivos


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos en Pinecone")
    parser.add_argument("rutas", nargs="+", help="Archivos o carpetas a indexar")
    parser.add_argument("--categoria", default="general", help="Materia del documento")
    parser.add_argument("--dry-run", action="store_true", help="Solo muestra qué haría")
    parser.add_argument(
        "--force", action="store_true",
        help="Indexa aunque ya exista un documento con el mismo nombre en el catálogo",
    )
    args = parser.parse_args()

    archivos = reunir_archivos(args.rutas)
    if not archivos:
        print("No hay archivos admitidos para indexar.")
        sys.exit(1)

    print(f"📚 {len(archivos)} archivo(s) por indexar:")
    for f in archivos:
        print(f"   - {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    if args.dry_run:
        print("\n🔎 Simulación: no se subió nada.")
        return

    init_firebase()

    # Se construyen aquí y no al importar: así --dry-run no exige credenciales
    from src.app.core.rag.embeddings import EmbeddingService
    from src.app.core.rag.store import VectorStoreService
    from src.app.services.document_service import DocumentService

    embeddings = EmbeddingService()
    servicio = DocumentService(VectorStoreService(embeddings), embeddings)
    actor = CurrentUser(
        uid="cli",
        email="ingesta por línea de comandos",
        name="ingesta por línea de comandos",
        role=ROLE_ADMIN,
    )

    # Los documentos de la ingesta original no tienen hash, así que el descarte
    # por contenido no los detecta: sin esta guarda, reingestar data/docs
    # duplicaría el índice entero.
    ya_indexados = {d.get("filename") for d in servicio.list_documents()}

    indexados = fallidos = omitidos = 0
    for archivo in archivos:
        print(f"\n📄 {archivo.name}")
        if archivo.name in ya_indexados and not args.force:
            print("   ⏭️  Ya está en el catálogo (use --force para indexarlo de nuevo)")
            omitidos += 1
            continue
        try:
            registro = servicio.create_document(
                filename=archivo.name,
                content=archivo.read_bytes(),
                actor=actor,
                category=args.categoria,
            )
        except Exception as e:
            detalle = getattr(e, "detail", str(e))
            print(f"   ⏭️  Se omite: {detalle}")
            omitidos += 1
            continue

        resultado = servicio.index_document(registro["doc_id"])
        if resultado.get("status") == "indexado":
            print(f"   ✅ {resultado.get('chunks', 0)} fragmentos indexados")
            indexados += 1
        else:
            print(f"   ❌ Error: {resultado.get('error', 'desconocido')}")
            fallidos += 1

    print(f"\n✅ Indexados: {indexados} | ⏭️  Omitidos: {omitidos} | ❌ Con error: {fallidos}")


if __name__ == "__main__":
    main()
