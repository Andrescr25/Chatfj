#!/usr/bin/env python3
"""
Sistema profesional de ingesta de documentos para RAG
Basado en mejores prácticas de OpenAI, Anthropic y Pinecone

Características:
- Chunking semántico inteligente
- Metadata rica y estructurada
- Overlap contextual
- Preservación de jerarquía
- Tamaño óptimo de chunks (512-1024 tokens)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import hashlib
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import (
        TextLoader, 
        PyPDFLoader,
        DirectoryLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Error importando dependencias: {e}")
    logger.error("Ejecuta: pip install langchain langchain-community pypdf chromadb sentence-transformers")
    sys.exit(1)

# Configuración optimizada
DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configuración de chunking profesional
CHUNK_CONFIG = {
    "chunk_size": 800,  # Tamaño óptimo en caracteres (~200 tokens)
    "chunk_overlap": 150,  # 18% de overlap para mantener contexto
    "separators": [
        "\n\n\n",  # Separación entre secciones grandes
        "\n\n",    # Separación entre párrafos
        "\n",      # Saltos de línea
        ". ",      # Fin de oración
        ", ",      # Comas (última opción)
        " ",       # Espacios
        ""         # Caracteres individuales (fallback)
    ]
}


class ProfessionalDocumentProcessor:
    """Procesador profesional de documentos con chunking semántico."""
    
    def __init__(self):
        self.embeddings = None
        self.vectordb = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CONFIG["chunk_size"],
            chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
            separators=CHUNK_CONFIG["separators"],
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_documents(self, data_dir: str) -> List[Document]:
        """Carga documentos desde el directorio especificado."""
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"❌ Directorio {data_dir} no existe")
            return documents
        
        logger.info(f"📂 Cargando documentos desde: {data_dir}")
        
        # Cargar PDFs
        pdf_files = list(data_path.glob("**/*.pdf"))
        for pdf_file in pdf_files:
            try:
                logger.info(f"  📄 Procesando PDF: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # Agregar metadata rica
                for doc in docs:
                    doc.metadata.update({
                        "filename": pdf_file.name,
                        "source": str(pdf_file),
                        "file_type": "pdf",
                        "category": self._detect_category(pdf_file.name),
                        "upload_date": datetime.now().isoformat(),
                    })
                
                documents.extend(docs)
                logger.info(f"    ✅ {len(docs)} páginas cargadas")
            except Exception as e:
                logger.error(f"    ❌ Error procesando {pdf_file.name}: {e}")
        
        # Cargar archivos de texto
        txt_files = list(data_path.glob("**/*.txt")) + list(data_path.glob("**/*.md"))
        for txt_file in txt_files:
            try:
                logger.info(f"  📝 Procesando texto: {txt_file.name}")
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                
                # Agregar metadata rica
                for doc in docs:
                    doc.metadata.update({
                        "filename": txt_file.name,
                        "source": str(txt_file),
                        "file_type": "text",
                        "category": self._detect_category(txt_file.name),
                        "upload_date": datetime.now().isoformat(),
                    })
                
                documents.extend(docs)
                logger.info(f"    ✅ {len(docs)} documentos cargados")
            except Exception as e:
                logger.error(f"    ❌ Error procesando {txt_file.name}: {e}")
        
        logger.info(f"✅ Total de documentos cargados: {len(documents)}")
        return documents
    
    def _detect_category(self, filename: str) -> str:
        """Detecta la categoría del documento basándose en el nombre."""
        filename_lower = filename.lower()
        
        categories = {
            "pension": ["pension", "alimentaria", "alimenticio"],
            "familia": ["familia", "familiar", "matrimonio", "divorcio"],
            "laboral": ["laboral", "trabajo", "despido", "salario"],
            "penal": ["penal", "delito", "condena"],
            "civil": ["civil", "contrato", "propiedad"],
            "procesal": ["procesal", "proceso", "procedimiento"],
        }
        
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return "general"
    
    def create_smart_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Crea chunks inteligentes con contexto preservado.
        Implementa mejores prácticas de chunking para RAG.
        """
        logger.info("🔪 Creando chunks semánticos inteligentes...")
        
        # Paso 1: Crear todos los chunks
        all_chunks = []
        
        for doc in documents:
            # Extraer título del documento si está disponible
            doc_title = self._extract_title(doc.page_content)
            
            # Crear chunks con el splitter recursivo
            chunks = self.text_splitter.split_documents([doc])
            
            # Enriquecer metadata de cada chunk (sin chunk_index ni total_chunks aún)
            for chunk in chunks:
                # Preservar metadata original
                chunk.metadata.update(doc.metadata)
                
                # Agregar metadata básica del chunk
                chunk.metadata.update({
                    "document_title": doc_title,
                    "char_count": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split()),
                })
                
                all_chunks.append(chunk)
        
        # Paso 2: Agrupar chunks por archivo y asignar índices correctos
        chunks_by_file = {}
        for chunk in all_chunks:
            filename = chunk.metadata.get('filename', 'unknown')
            if filename not in chunks_by_file:
                chunks_by_file[filename] = []
            chunks_by_file[filename].append(chunk)
        
        # Paso 3: Asignar chunk_index y total_chunks correctos por archivo
        final_chunks = []
        for filename, file_chunks in chunks_by_file.items():
            total_chunks_for_file = len(file_chunks)
            
            for i, chunk in enumerate(file_chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": total_chunks_for_file,
                    "chunk_id": hashlib.md5(f"{filename}_{i}".encode()).hexdigest()[:16],
                })
                
                # Agregar contexto de documento al inicio del primer chunk
                doc_title = chunk.metadata.get("document_title", "")
                if doc_title and i == 0:
                    chunk.page_content = f"[{doc_title}]\n\n{chunk.page_content}"
                
                final_chunks.append(chunk)
        
        logger.info(f"✅ Total de chunks creados: {len(final_chunks)}")
        logger.info(f"📊 Promedio de chunks por archivo: {len(final_chunks) / len(chunks_by_file):.1f}")
        logger.info(f"📁 Archivos únicos procesados: {len(chunks_by_file)}")
        
        return final_chunks
    
    def _extract_title(self, content: str) -> str:
        """Extrae el título del contenido del documento."""
        lines = content.split('\n')
        for line in lines[:5]:  # Buscar en las primeras 5 líneas
            clean_line = line.strip()
            if len(clean_line) > 10 and len(clean_line) < 150:
                return clean_line
        return "Documento Legal"
    
    def create_vectorstore(self, chunks: List[Document], persist_dir: str) -> Chroma:
        """Crea la base de datos vectorial con los chunks."""
        logger.info("🔢 Generando embeddings y creando vectorstore...")
        
        # Inicializar embeddings si no existen
        if not self.embeddings:
            logger.info(f"📊 Cargando modelo de embeddings: {EMBEDDING_MODEL}")
            self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Crear directorio si no existe
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Crear vectorstore
        logger.info(f"💾 Creando Chroma DB en: {persist_dir}")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
            collection_name="legal_documents"
        )
        
        logger.info("✅ Vectorstore creado exitosamente")
        return vectordb
    
    def get_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """Genera estadísticas del procesamiento."""
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(len(c.page_content) for c in chunks),
            "avg_chunk_size": sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0,
            "categories": {},
            "file_types": {},
            "documents": set()
        }
        
        for chunk in chunks:
            category = chunk.metadata.get("category", "unknown")
            file_type = chunk.metadata.get("file_type", "unknown")
            filename = chunk.metadata.get("filename", "unknown")
            
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["documents"].add(filename)
        
        stats["unique_documents"] = len(stats["documents"])
        del stats["documents"]  # No necesitamos la lista completa
        
        return stats


def main():
    """Función principal de ingesta profesional."""
    logger.info("=" * 60)
    logger.info("🚀 SISTEMA PROFESIONAL DE INGESTA DE DOCUMENTOS")
    logger.info("=" * 60)
    
    # Inicializar procesador
    processor = ProfessionalDocumentProcessor()
    
    # Paso 1: Cargar documentos
    logger.info("\n📥 PASO 1: Carga de documentos")
    documents = processor.load_documents(DATA_DIR)
    
    if not documents:
        logger.error("❌ No se encontraron documentos para procesar")
        return 1
    
    # Paso 2: Crear chunks inteligentes
    logger.info("\n🔪 PASO 2: Chunking semántico")
    chunks = processor.create_smart_chunks(documents)
    
    # Paso 3: Generar estadísticas
    logger.info("\n📊 PASO 3: Estadísticas del procesamiento")
    stats = processor.get_statistics(chunks)
    
    logger.info(f"  • Total de chunks: {stats['total_chunks']}")
    logger.info(f"  • Documentos únicos: {stats['unique_documents']}")
    logger.info(f"  • Tamaño promedio de chunk: {stats['avg_chunk_size']:.0f} caracteres")
    logger.info(f"  • Total de caracteres: {stats['total_characters']:,}")
    
    logger.info("\n  Distribución por categoría:")
    for category, count in stats['categories'].items():
        logger.info(f"    - {category}: {count} chunks")
    
    logger.info("\n  Distribución por tipo de archivo:")
    for file_type, count in stats['file_types'].items():
        logger.info(f"    - {file_type}: {count} chunks")
    
    # Paso 4: Crear vectorstore
    logger.info("\n💾 PASO 4: Creación de base de datos vectorial")
    
    # Limpiar base de datos anterior si existe
    if os.path.exists(CHROMA_DIR):
        import shutil
        logger.warning(f"⚠️  Eliminando base de datos anterior: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)
    
    vectordb = processor.create_vectorstore(chunks, CHROMA_DIR)
    
    # Verificar
    collection_count = vectordb._collection.count()
    logger.info(f"✅ Verificación: {collection_count} documentos en la base de datos")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ INGESTA COMPLETADA EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"\n📍 Base de datos creada en: {CHROMA_DIR}")
    logger.info(f"📊 Total de chunks indexados: {collection_count}")
    logger.info("\n🎯 El sistema está listo para responder preguntas\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

