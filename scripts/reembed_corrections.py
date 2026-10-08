#!/usr/bin/env python3
"""
Regenera los vectores de las correcciones con los embeddings de Pinecone.

Una corrección se encuentra comparando la pregunta nueva con la pregunta que se
corrigió. Esos vectores los generó HuggingFace, sin el prefijo "query: " que
antepone Pinecone, y esa diferencia baja los puntajes justo alrededor del umbral
de 0,85. Medido sobre las 70 correcciones existentes: la misma pregunta dejaba
de dar 1,00 (bajaba hasta 0,95) y se perdía un 22% de las coincidencias entre
preguntas parecidas. Con los dos lados generados igual vuelve el comportamiento
anterior.

Solo cambia el vector: el texto y los demás datos de cada corrección se
conservan. Antes de escribir guarda un respaldo en data/respaldos/, que no se
versiona.

Uso:
    python scripts/reembed_corrections.py --dry-run
    python scripts/reembed_corrections.py
    python scripts/reembed_corrections.py --restaurar data/respaldos/<archivo>.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

ESPACIO = "corrections"
LOTE = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenera los vectores de las correcciones")
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta, no escribe")
    parser.add_argument("--restaurar", metavar="ARCHIVO", help="Vuelve a subir un respaldo anterior")
    args = parser.parse_args()

    from src.app.core.rag.embeddings import EmbeddingService
    from src.app.core.rag.store import VectorStoreService

    embeddings = EmbeddingService()
    store = VectorStoreService(embeddings)
    if not store.pinecone_index:
        print("❌ Pinecone no está disponible.")
        sys.exit(1)

    if args.restaurar:
        guardados = json.loads(Path(args.restaurar).read_text(encoding="utf-8"))
        vectores = [(c["id"], c["values"], c["metadata"]) for c in guardados]
        for i in range(0, len(vectores), LOTE):
            if not store.upsert_vectors(vectores[i:i + LOTE], namespace=ESPACIO):
                print("❌ Pinecone rechazó la subida.")
                sys.exit(1)
        print(f"✅ Restauradas {len(vectores)} correcciones desde {args.restaurar}")
        return

    # Se llama a Pinecone directamente y no a la cascada: si respondiera otro
    # proveedor, el espacio quedaría mezclado sin que nadie lo notara.
    if not embeddings.pinecone:
        print("❌ Los embeddings por Pinecone no están activos (PINECONE_API_KEY / EMBEDDINGS_PINECONE_ENABLED).")
        sys.exit(1)

    ids = store.list_vector_ids("", namespace=ESPACIO) or []
    actuales = []
    for i in range(0, len(ids), LOTE):
        actuales.extend(store.fetch_vectors_full(ids[i:i + LOTE], namespace=ESPACIO))
    con_pregunta = [c for c in actuales if (c[2].get("original_question") or "").strip()]

    print(f"📊 Correcciones en el índice: {len(actuales)}")
    print(f"📊 Con pregunta para regenerar: {len(con_pregunta)}")
    if args.dry_run or not con_pregunta:
        print("\n🔎 Sin cambios.")
        return

    carpeta = RAIZ / "data" / "respaldos"
    carpeta.mkdir(parents=True, exist_ok=True)
    respaldo = carpeta / f"correcciones_{datetime.now():%Y%m%d_%H%M%S}.json"
    respaldo.write_text(
        json.dumps([{"id": i, "values": v, "metadata": m} for i, v, m in actuales], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"💾 Respaldo de los vectores actuales: {respaldo.relative_to(RAIZ)}")

    nuevos = [
        (vid, embeddings.pinecone.embed_query(metadata["original_question"]), metadata)
        for vid, _valores, metadata in con_pregunta
    ]
    for i in range(0, len(nuevos), LOTE):
        if not store.upsert_vectors(nuevos[i:i + LOTE], namespace=ESPACIO):
            print(f"❌ Pinecone rechazó la subida. Para deshacer: --restaurar {respaldo.relative_to(RAIZ)}")
            sys.exit(1)

    print(f"✅ Regeneradas {len(nuevos)} correcciones con {embeddings.pinecone.nombre}")


if __name__ == "__main__":
    main()
