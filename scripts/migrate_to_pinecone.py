import os
import sys
import time
from typing import List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Configuración
load_dotenv("config/config.env")
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-index")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def migrate():
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY no encontrada en variables de entorno.")
        return

    print(f"🚀 Iniciando migración de Chroma ({CHROMA_DIR}) a Pinecone ({INDEX_NAME})...")

    # 1. Cargar Embeddings y Chroma local
    print("📦 Cargando modelo de embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("📂 Abriendo ChromaDB local...")
    try:
        if not os.path.exists(CHROMA_DIR):
            print(f"❌ Error: Directorio Chroma no existe: {CHROMA_DIR}")
            return
            
        chroma_db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="legal_documents"
        )
        # Obtener todos los documentos
        collection_data = chroma_db.get(include=['embeddings', 'metadatas', 'documents'])
        ids = collection_data['ids']
        vectors = collection_data['embeddings']
        metadatas = collection_data['metadatas']
        texts = collection_data['documents']
        
        total_docs = len(ids)
        print(f"✅ Se encontraron {total_docs} documentos en ChromaDB.")
        
    except Exception as e:
        print(f"❌ Error leyendo ChromaDB: {e}")
        return

    # 2. Inicializar Pinecone
    print("connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Verificar si el índice existe
    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"✨ Creando índice '{INDEX_NAME}'...")
        # Dimension 768 es típica para mpnet-base-v2
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Esperar a que se inicialice
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            print(".", end="", flush=True)
        print(" Listo!")
    else:
        print(f"ℹ️ Índice '{INDEX_NAME}' ya existe.")

    index = pc.Index(INDEX_NAME)

    # 3. Subir datos en batches
    BATCH_SIZE = 100
    print(f"⬆️ Subiendo vectores en lotes de {BATCH_SIZE}...")

    # Pinecone espera: (id, vector, metadata)
    # Metadata debe incluir el texto para que Langchain pueda usarlo luego
    
    data_to_upsert = []
    
    for i in range(total_docs):
        # Preparar metadata: asegurar que 'text' esté presente
        meta = metadatas[i] if metadatas[i] else {}
        meta['text'] = texts[i]
        
        # Filtrar valores nulos en metadata porque Pinecone falla
        clean_meta = {k: v for k, v in meta.items() if v is not None}
        
        data_to_upsert.append((ids[i], vectors[i], clean_meta))

        # Subir cuando alcanzamos el batch
        if len(data_to_upsert) >= BATCH_SIZE:
            try:
                index.upsert(vectors=data_to_upsert)
                print(f"   Procesados {i+1}/{total_docs}...")
                data_to_upsert = []
            except Exception as e:
                print(f"⚠️ Error subiendo lote: {e}")

    # Subir remanentes
    if data_to_upsert:
        try:
            index.upsert(vectors=data_to_upsert)
            print(f"   Procesados {total_docs}/{total_docs}.")
        except Exception as e:
            print(f"⚠️ Error subiendo lote final: {e}")

    # Verificar
    stats = index.describe_index_stats()
    print("\n✅ Migración completada.")
    print(f"📊 Estadísticas del índice: {stats}")

if __name__ == "__main__":
    migrate()
