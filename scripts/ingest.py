#!/usr/bin/env python3
"""
Script de ingesta de documentos para el bot de Facilitadores Judiciales.
Procesa documentos PDF, DOCX y TXT, los fragmenta y genera embeddings.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importaciones de LangChain
try:
    from langchain_community.document_loaders import (
        TextLoader, 
        PyPDFLoader,
        Docx2txtLoader,
        DirectoryLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Error importando LangChain: {e}")
    logger.error("Ejecuta: pip install langchain langchain-community pypdf")
    sys.exit(1)

# Configuración
DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
PERSIST_DIR = os.getenv("VECTOR_DB_DIR", "./data/chroma")
MODEL_EMBED = "all-MiniLM-L6-v2"  # Modelo ligero y eficiente
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_documents(data_dir: str) -> List[Document]:
    """
    Carga documentos desde el directorio especificado.
    Soporta: .txt, .pdf, .docx
    """
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Directorio {data_dir} no existe. Creándolo...")
        data_path.mkdir(parents=True, exist_ok=True)
        return documents
    
    # Buscar archivos soportados
    supported_extensions = {'.txt', '.pdf', '.docx'}
    files_found = []
    
    for ext in supported_extensions:
        files_found.extend(data_path.rglob(f"*{ext}"))
    
    if not files_found:
        logger.warning(f"No se encontraron documentos en {data_dir}")
        logger.info("Formatos soportados: .txt, .pdf, .docx")
        return documents
    
    logger.info(f"Encontrados {len(files_found)} archivos para procesar")
    
    # Procesar cada archivo
    for file_path in files_found:
        try:
            logger.info(f"Procesando: {file_path.name}")
            
            if file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                continue
            
            docs = loader.load()
            
            # Agregar metadatos del archivo
            for doc in docs:
                doc.metadata.update({
                    'source': str(file_path),
                    'filename': file_path.name,
                    'file_type': file_path.suffix.lower()
                })
            
            documents.extend(docs)
            logger.info(f"✅ {file_path.name}: {len(docs)} páginas cargadas")
            
        except Exception as e:
            logger.error(f"❌ Error procesando {file_path.name}: {e}")
            continue
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Fragmenta los documentos en chunks más pequeños para mejor procesamiento.
    """
    if not documents:
        return documents
    
    logger.info("Fragmentando documentos...")
    
    # Configurar el splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Fragmentar documentos
    split_docs = text_splitter.split_documents(documents)
    
    logger.info(f"Documentos fragmentados: {len(split_docs)} chunks")
    logger.info(f"Tamaño promedio de chunk: {sum(len(doc.page_content) for doc in split_docs) // len(split_docs)} caracteres")
    
    return split_docs

def create_embeddings(split_docs: List[Document]) -> Chroma:
    """
    Crea embeddings y los guarda en ChromaDB.
    """
    if not split_docs:
        logger.error("No hay documentos para procesar")
        return None
    
    logger.info("Generando embeddings...")
    
    try:
        # Crear embedder
        embedder = SentenceTransformerEmbeddings(model_name=MODEL_EMBED)
        logger.info(f"Modelo de embeddings: {MODEL_EMBED}")
        
        # Crear o cargar base de datos vectorial
        persist_path = Path(PERSIST_DIR)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar si ya existe una base de datos
        if (persist_path / "chroma.sqlite3").exists():
            logger.info("Base de datos existente encontrada. Agregando nuevos documentos...")
            vectordb = Chroma(
                persist_directory=str(persist_path),
                embedding_function=embedder
            )
            # Agregar nuevos documentos
            vectordb.add_documents(split_docs)
        else:
            logger.info("Creando nueva base de datos vectorial...")
            vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=embedder,
                persist_directory=str(persist_path)
            )
        
        # Persistir la base de datos
        vectordb.persist()
        
        # Verificar el contenido
        collection_count = vectordb._collection.count()
        logger.info(f"✅ Base de datos vectorial creada con {collection_count} documentos")
        
        return vectordb
        
    except Exception as e:
        logger.error(f"Error creando embeddings: {e}")
        return None

def test_retrieval(vectordb: Chroma, test_query: str = "procedimiento judicial"):
    """
    Prueba la recuperación de documentos con una consulta de ejemplo.
    """
    if not vectordb:
        return
    
    logger.info(f"Probando recuperación con consulta: '{test_query}'")
    
    try:
        # Buscar documentos similares
        results = vectordb.similarity_search(test_query, k=3)
        
        logger.info(f"Encontrados {len(results)} documentos relevantes:")
        for i, doc in enumerate(results, 1):
            logger.info(f"  {i}. {doc.metadata.get('filename', 'Sin nombre')} (similaridad: {doc.metadata.get('score', 'N/A')})")
            logger.info(f"     Fragmento: {doc.page_content[:100]}...")
    
    except Exception as e:
        logger.error(f"Error en prueba de recuperación: {e}")

def main():
    """
    Función principal del script de ingesta.
    """
    logger.info("🚀 Iniciando ingesta de documentos para Facilitadores Judiciales")
    logger.info(f"Directorio de datos: {DATA_DIR}")
    logger.info(f"Directorio de vectores: {PERSIST_DIR}")
    
    # 1. Cargar documentos
    documents = load_documents(DATA_DIR)
    
    if not documents:
        logger.warning("No se encontraron documentos para procesar.")
        logger.info("Coloca archivos .txt, .pdf o .docx en la carpeta data/docs/")
        return
    
    # 2. Fragmentar documentos
    split_docs = split_documents(documents)
    
    # 3. Crear embeddings
    vectordb = create_embeddings(split_docs)
    
    if vectordb:
        # 4. Probar recuperación
        test_retrieval(vectordb)
        
        logger.info("✅ Ingesta completada exitosamente!")
        logger.info(f"📊 Estadísticas:")
        logger.info(f"   - Documentos originales: {len(documents)}")
        logger.info(f"   - Chunks generados: {len(split_docs)}")
        logger.info(f"   - Base de datos: {PERSIST_DIR}")
    else:
        logger.error("❌ Error en la ingesta de documentos")

if __name__ == "__main__":
    main()
