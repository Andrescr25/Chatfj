#!/usr/bin/env python3
"""
Importa bloques procesados a ChromaDB usando LangChain's Chroma (compatible con la API).
"""
import json
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def import_blocks(jsonl_file: str, persist_dir: str = "data/chroma_db"):
    print(f"📦 Inicializando ChromaDB (LangChain) en {persist_dir}...")
    
    # Inicializar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Leer bloques
    blocks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))
    
    print(f"📄 {len(blocks)} bloques cargados\n")
    
    # Preparar documentos
    texts = []
    metadatas = []
    ids = []
    
    for idx, block in enumerate(blocks):
        texts.append(block.get("texto", ""))
        metadatas.append({
            "ley": str(block.get("ley") or "Desconocida"),
            "articulo": str(block.get("articulo") or ""),
            "categoria": str(block.get("categoria") or "desconocida"),
            "documento": str(block.get("documento") or ""),
            "pagina": int(block.get("pagina") or 0)
        })
        ids.append(f"doc_{idx}")
    
    # Crear vectorstore
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir
    )
    
    print(f"🎉 Importación completada: {len(blocks)} documentos indexados")
    print(f"📊 Total en colección: {vectorstore._collection.count()} documentos")

if __name__ == "__main__":
    import_blocks("data/bloques_limpios.jsonl")
