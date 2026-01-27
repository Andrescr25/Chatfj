#!/usr/bin/env python3
"""
Script de Ingesta para Pinecone (Producción)
--------------------------------------------
Este script carga documentos PDF/TXT, los fragmenta y los sube al índice de Pinecone.
Usa la misma configuración de Embeddings que el backend para asegurar compatibilidad via HuggingFace API.

Requisitos:
- PINECONE_API_KEY
- HUGGINGFACEHUB_API_TOKEN (Opcional, usa local si falta)
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Intentar importar dependencias
try:
    from dotenv import load_dotenv, find_dotenv
    # Load from config/config.env explicitly first, then .env
    load_dotenv("config/config.env") 
    load_dotenv(find_dotenv())
    
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, DirectoryLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Try import fallback
    try:
        from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    from langchain_community.embeddings import SentenceTransformerEmbeddings
        
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"❌ Error importando dependencias: {e}")
    logger.error("Ejecuta: pip install langchain-pinecone pinecone-client langchain-huggingface python-dotenv pypdf sentence-transformers")
    sys.exit(1)

# Configuración
DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatfj-legal-index")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def validate_env():
    if not PINECONE_API_KEY:
        logger.error("❌ PINECONE_API_KEY no encontrada en variables de entorno (ni en config/config.env).")
        return False
    if not HF_TOKEN:
        logger.warning("⚠️ HUGGINGFACEHUB_API_TOKEN no encontrada. Usaremos modelo LOCAL (descargando ~400MB).")
    return True

def load_documents(data_dir: str) -> List[Document]:
    documents = []
    path = Path(data_dir)
    
    if not path.exists():
        logger.error(f"❌ Directorio {data_dir} no existe.")
        return []
        
    logger.info(f"📂 Buscando documentos en {data_dir}...")
    
    # PDFs
    for pdf in path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": str(pdf),
                    "filename": pdf.name,
                    "type": "pdf",
                    "ingested_at": datetime.now().isoformat()
                })
            documents.extend(docs)
            logger.info(f"  ✅ PDF cargado: {pdf.name} ({len(docs)} págs)")
        except Exception as e:
            logger.error(f"  ❌ Error en {pdf.name}: {e}")

    # TXT/MD
    for txt in list(path.glob("**/*.txt")) + list(path.glob("**/*.md")):
        try:
            loader = TextLoader(str(txt), encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source": str(txt),
                    "filename": txt.name,
                    "type": "text",
                    "ingested_at": datetime.now().isoformat()
                })
            documents.extend(docs)
            logger.info(f"  ✅ Texto cargado: {txt.name}")
        except Exception as e:
            logger.error(f"  ❌ Error en {txt.name}: {e}")
            
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"🔪 Documentos fragmentados en {len(chunks)} chunks.")
    return chunks

def ingest_to_pinecone(chunks: List[Document]):
    logger.info("📡 Conectando a servicios de Embedding y Vector DB...")
    
    try:
        embeddings = None
        if HF_TOKEN:
            # Embeddings via API (Zero RAM usage locally)
            logger.info("☁️ Usando HuggingFace API para embeddings...")
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN,
                model_name=EMBEDDING_MODEL
            )
        else:
            # Fallback to Local Model
            logger.info(f"💻 Usando modelo LOCAL: {EMBEDDING_MODEL}")
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

        
        # Test Embeddings
        logger.info("🧪 Probando generación de embeddings...")
        try:
            test_emb = embeddings.embed_query("test")
            if isinstance(test_emb, dict): 
                raise ValueError(f"API retornó error: {test_emb}")
            logger.info("✅ Embeddings funcionando.")
        except Exception as e:
            logger.error(f"❌ Fallo en Embeddings API: {e}")
            return

        # Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
        
        # Verificar stats antes
        stats = index.describe_index_stats()
        logger.info(f"📊 Estado actual del índice: {stats}")
        
        logger.info(f"🚀 Subiendo {len(chunks)} chunks a Pinecone index '{PINECONE_INDEX_NAME}'...")
        
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        vector_store.add_documents(chunks)
        
        logger.info("✅ ¡Ingesta Completada!")
        
    except Exception as e:
        logger.error(f"❌ Error crítico en ingesta: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if not validate_env():
        sys.exit(1)
        
    docs = load_documents(DATA_DIR)
    if docs:
        chunks = split_documents(docs)
        ingest_to_pinecone(chunks)
    else:
        logger.warning("No se encontraron documentos válidos. Verifica la ruta DATA_DIR.")
