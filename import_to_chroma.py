#!/usr/bin/env python3
"""
Importador de bloques JSON a ChromaDB
Lee bloques estructurados y los indexa en la base vectorial
"""

import json
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings


def import_blocks_to_chroma(jsonl_file: str, persist_dir: str = "data/chroma_db"):
    """Importa bloques JSON a ChromaDB."""

    print(f"📦 Inicializando ChromaDB en {persist_dir}...")

    # Inicializar cliente persistente
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # Obtener o crear colección
    collection = client.get_or_create_collection(
        name="legal_docs",
        metadata={"description": "Documentos legales de Costa Rica"}
    )

    print(f"✅ Colección 'legal_docs' lista\n")

    # Leer bloques desde archivo JSONL
    blocks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                block = json.loads(line)
                blocks.append(block)
            except json.JSONDecodeError as e:
                print(f"⚠️  Error en línea {line_num}: {e}", file=sys.stderr)
                continue

    if not blocks:
        print("❌ No se encontraron bloques válidos", file=sys.stderr)
        sys.exit(1)

    print(f"📄 {len(blocks)} bloques cargados\n")

    # Preparar datos para ChromaDB
    documents = []
    metadatas = []
    ids = []

    for idx, block in enumerate(blocks):
        # Crear documento con contexto completo
        doc_text = block.get("texto", "")

        # Agregar contexto de ley y artículo si existe
        if block.get("ley") and block.get("articulo"):
            doc_text = f"{block['ley']}, {block['articulo']}: {doc_text}"
        elif block.get("ley"):
            doc_text = f"{block['ley']}: {doc_text}"

        documents.append(doc_text)

        # Metadata - convertir None a valores válidos
        metadata = {
            "ley": str(block.get("ley") or "Desconocida"),
            "articulo": str(block.get("articulo") or ""),
            "categoria": str(block.get("categoria") or "desconocida"),
            "documento": str(block.get("documento") or ""),
            "pagina": int(block.get("pagina") or 0)
        }
        metadatas.append(metadata)

        # ID único
        doc_name = block.get("documento", "doc").replace(".pdf", "")
        ids.append(f"{doc_name}_p{block.get('pagina', 0)}_{idx}")

    # Añadir a ChromaDB en lotes
    batch_size = 100
    total_added = 0

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            total_added += len(batch_docs)
            print(f"   ✅ Lote {i//batch_size + 1}: {len(batch_docs)} documentos añadidos (Total: {total_added})")
        except Exception as e:
            print(f"   ⚠️  Error en lote {i//batch_size + 1}: {e}", file=sys.stderr)

    print(f"\n🎉 Importación completada: {total_added} documentos indexados")

    # Estadísticas
    stats = collection.count()
    print(f"📊 Total en colección: {stats} documentos")

    return total_added


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python3 import_to_chroma.py <archivo.jsonl> [directorio_chroma]", file=sys.stderr)
        print("\nEjemplo:", file=sys.stderr)
        print("  python3 import_to_chroma.py bloques_ley7586.jsonl", file=sys.stderr)
        print("  python3 import_to_chroma.py bloques_ley7586.jsonl data/chroma_db", file=sys.stderr)
        sys.exit(1)

    jsonl_file = sys.argv[1]
    persist_dir = sys.argv[2] if len(sys.argv) > 2 else "data/chroma_db"

    if not Path(jsonl_file).exists():
        print(f"❌ Error: Archivo {jsonl_file} no existe", file=sys.stderr)
        sys.exit(1)

    import_blocks_to_chroma(jsonl_file, persist_dir)


if __name__ == "__main__":
    main()
