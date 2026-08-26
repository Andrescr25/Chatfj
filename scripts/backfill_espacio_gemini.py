#!/usr/bin/env python3
"""
Copia los fragmentos ya indexados al espacio de Gemini.

Los documentos existentes solo tienen vectores de e5. Este script los vuelve a
convertir con Gemini y los guarda en el espacio 'gemini', para que ese modelo
sirva de respaldo real cuando el otro se quede sin cupo.

No hacen falta los archivos originales: el texto de cada fragmento ya está en
la metadata del índice.

Uso:
    python scripts/backfill_espacio_gemini.py --dry-run
    python scripts/backfill_espacio_gemini.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import firebase_admin
from firebase_admin import credentials

from src.app.config import settings
from src.app.core.rag.embeddings import ESPACIO_E5, ESPACIO_GEMINI

# Gemini cuenta cada elemento del lote como una solicitud y limita por minuto,
# así que conviene ir de a poco y con pausas.
LOTE = 10
PAUSA = 4.0


def init_firebase() -> None:
    if firebase_admin._apps:
        return
    try:
        cred_path = settings.FIREBASE_CREDENTIALS_PATH
        if os.path.exists(cred_path):
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        else:
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"⚠️ Firebase no disponible ({e}); se continúa sin catálogo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplica el índice al espacio de Gemini")
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta, no escribe")
    parser.add_argument("--limite", type=int, default=0, help="Máximo de fragmentos a copiar")
    parser.add_argument("--lote", type=int, default=LOTE, help="Fragmentos por solicitud")
    parser.add_argument("--pausa", type=float, default=PAUSA, help="Segundos entre lotes")
    args = parser.parse_args()

    init_firebase()

    from src.app.core.rag.embeddings import EmbeddingService
    from src.app.core.rag.store import VectorStoreService

    embeddings = EmbeddingService()
    if not embeddings.gemini:
        print("❌ Gemini no está configurado (GEMINI_API_KEY / EMBEDDINGS_GEMINI_ENABLED).")
        sys.exit(1)

    store = VectorStoreService(embeddings)
    indice = store.pinecone_index
    if not indice:
        print("❌ Pinecone no está disponible.")
        sys.exit(1)

    ids = store.list_vector_ids("", namespace=ESPACIO_E5) or []
    ya = set(store.list_vector_ids("", namespace=ESPACIO_GEMINI) or [])
    pendientes = [i for i in ids if i not in ya]
    if args.limite:
        pendientes = pendientes[: args.limite]

    print(f"📊 Fragmentos en el espacio de e5: {len(ids)}")
    print(f"📊 Ya copiados al de Gemini:      {len(ya)}")
    print(f"📊 Pendientes:                    {len(pendientes)}")

    if args.dry_run or not pendientes:
        print("\n🔎 Sin cambios." if args.dry_run else "\n✅ No hay nada pendiente.")
        return

    copiados = fallidos = 0
    inicio = time.time()

    for i in range(0, len(pendientes), args.lote):
        lote = pendientes[i:i + args.lote]
        completos = store.fetch_vectors_full(lote, namespace=ESPACIO_E5)
        textos, metadatos, identificadores = [], [], []
        for vid, _valores, metadata in completos:
            texto = (metadata or {}).get("text", "")
            if not texto.strip():
                continue
            identificadores.append(vid)
            textos.append(texto)
            metadatos.append(metadata)

        if not textos:
            continue

        # Ante un 429 se espera cada vez más: el límite es por minuto y se repone solo
        espera = args.pausa
        for intento in range(6):
            try:
                vectores = embeddings.gemini.embed_documents(textos)
                if len(vectores) != len(textos):
                    raise RuntimeError(f"devolvió {len(vectores)}/{len(textos)} vectores")
                store.upsert_vectors(
                    list(zip(identificadores, vectores, metadatos, strict=False)),
                    namespace=ESPACIO_GEMINI,
                )
                copiados += len(textos)
                break
            except Exception as e:
                if "429" not in str(e) or intento == 5:
                    fallidos += len(textos)
                    print(f"\n   ⚠️ Lote con error: {str(e)[:110]}")
                    break
                espera = min(espera * 2, 70)
                time.sleep(espera)

        hechos = i + len(lote)
        ritmo = copiados / max(time.time() - inicio, 1) * 60
        print(f"   {hechos}/{len(pendientes)} · copiados {copiados} · "
              f"{ritmo:.0f}/min · {time.time() - inicio:.0f}s", end="\r")
        time.sleep(args.pausa)

    print()
    print(f"✅ Copiados: {copiados} | ❌ Con error: {fallidos} | ⏱️ {time.time() - inicio:.0f}s")
    print("   El espacio de Gemini queda como respaldo de la búsqueda.")


if __name__ == "__main__":
    main()
