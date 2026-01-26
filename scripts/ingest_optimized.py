#!/usr/bin/env python3
"""
Sistema OPTIMIZADO de ingesta de documentos para RAG
Mejoras sobre ingest_professional.py:
- Filtrado de contenido no útil (índices, presentaciones, etc.)
- Chunking semántico mejorado con contexto de sección
- Eliminación de chunks muy cortos
- Mejor preservación de estructura legal (artículos, capítulos)
"""

import os
import sys
import re
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
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
except ImportError as e:
    logger.error(f"Error importando dependencias: {e}")
    logger.error("Ejecuta: pip install langchain langchain-community pypdf chromadb sentence-transformers")
    sys.exit(1)

# Configuración
DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Configuración de chunking OPTIMIZADA
CHUNK_CONFIG = {
    "chunk_size": 1200,  # Aumentado para chunks más sustanciales
    "chunk_overlap": 250,  # Mayor overlap para mejor continuidad
    "separators": [
        "\n\n\n",      # Separación entre secciones grandes
        "\n\n",        # Separación entre párrafos
        "\nArtículo ", # Separar artículos legales
        "\nCapítulo ", # Separar capítulos
        "\n",          # Saltos de línea
        ". ",          # Fin de oración
        ", ",          # Comas
        " ",           # Espacios
        ""             # Fallback
    ]
}

# Patrones para detectar contenido no útil
USELESS_PATTERNS = [
    r'^[\s\.]+$',  # Solo espacios y puntos (índices)
    r'^PRESENTACI[OÓ]N[\s\.]+$',
    r'^CONTENIDO[\s\.]+$',
    r'^TABLA DE CONTENIDO[\s\.]+$',
    r'^[ÍI]NDICE[\s\.]+$',
    r'^\d+\s*$',  # Solo números de página
]

class OptimizedDocumentProcessor:
    """Procesador OPTIMIZADO de documentos con filtrado inteligente."""

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
        self.stats = {
            "total_pages": 0,
            "filtered_pages": 0,
            "total_chunks": 0,
            "filtered_chunks": 0,
        }

    def is_useless_content(self, text: str) -> bool:
        """Detecta si el contenido es inútil (índice, presentación, etc.)."""
        text_clean = text.strip()

        # Muy corto
        if len(text_clean) < 50:
            return True

        # Coincide con patrones inútiles
        for pattern in USELESS_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True

        # Mayoría de caracteres son puntos (índices)
        if text_clean.count('.') > len(text_clean) * 0.3:
            return True

        # Demasiados números (tabla de contenidos)
        digits = sum(c.isdigit() for c in text_clean)
        if digits > len(text_clean) * 0.5:
            return True

        return False

    def extract_section_context(self, text: str) -> Dict[str, str]:
        """Extrae contexto de sección (artículo, capítulo, etc.)."""
        context = {}

        # Buscar artículo
        article_match = re.search(r'Art[ií]culo\s+(\d+[a-z]*)', text, re.IGNORECASE)
        if article_match:
            context['article'] = article_match.group(1)

        # Buscar capítulo
        chapter_match = re.search(r'Cap[ií]tulo\s+([IVXLCDM]+|\d+)', text, re.IGNORECASE)
        if chapter_match:
            context['chapter'] = chapter_match.group(1)

        # Buscar sección
        section_match = re.search(r'Secci[oó]n\s+([IVXLCDM]+|\d+)', text, re.IGNORECASE)
        if section_match:
            context['section'] = section_match.group(1)

        return context

    def load_documents(self, data_dir: str) -> List[Document]:
        """Carga documentos filtrando contenido inútil."""
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

                # Filtrar páginas inútiles
                filtered_docs = []
                for doc in docs:
                    self.stats["total_pages"] += 1

                    if self.is_useless_content(doc.page_content):
                        self.stats["filtered_pages"] += 1
                        continue

                    # Agregar metadata rica
                    doc.metadata.update({
                        "filename": pdf_file.name,
                        "source": str(pdf_file),
                        "file_type": "pdf",
                        "category": self._detect_category(pdf_file.name),
                        "upload_date": datetime.now().isoformat(),
                    })

                    # Agregar contexto de sección
                    section_context = self.extract_section_context(doc.page_content)
                    doc.metadata.update(section_context)

                    filtered_docs.append(doc)

                documents.extend(filtered_docs)
                logger.info(f"    ✅ {len(filtered_docs)}/{len(docs)} páginas útiles")

            except Exception as e:
                logger.error(f"    ❌ Error procesando {pdf_file.name}: {e}")

        # Cargar archivos de texto
        txt_files = list(data_path.glob("**/*.txt")) + list(data_path.glob("**/*.md"))
        for txt_file in txt_files:
            try:
                logger.info(f"  📝 Procesando texto: {txt_file.name}")
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()

                for doc in docs:
                    if not self.is_useless_content(doc.page_content):
                        doc.metadata.update({
                            "filename": txt_file.name,
                            "source": str(txt_file),
                            "file_type": "text",
                            "category": self._detect_category(txt_file.name),
                            "upload_date": datetime.now().isoformat(),
                        })
                        documents.append(doc)

                logger.info(f"    ✅ {len(docs)} documentos cargados")
            except Exception as e:
                logger.error(f"    ❌ Error procesando {txt_file.name}: {e}")

        logger.info(f"✅ Total de documentos útiles cargados: {len(documents)}")
        logger.info(f"🗑️  Páginas filtradas: {self.stats['filtered_pages']}/{self.stats['total_pages']}")
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
            "constitucional": ["constitucion", "constitucional"],
            "agrario": ["agrario", "pesca", "agricultura"],
        }

        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category

        return "general"

    def create_smart_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Crea chunks inteligentes con:
        - Filtrado de chunks muy cortos
        - Contexto de sección preservado
        - Metadata enriquecida
        """
        logger.info("🔪 Creando chunks semánticos optimizados...")

        all_chunks = []

        for doc in documents:
            # Extraer título del documento
            doc_title = self._extract_title(doc.page_content)

            # Crear chunks
            chunks = self.text_splitter.split_documents([doc])

            for chunk in chunks:
                self.stats["total_chunks"] += 1

                # Filtrar chunks muy cortos o inútiles
                if len(chunk.page_content) < 200 or self.is_useless_content(chunk.page_content):
                    self.stats["filtered_chunks"] += 1
                    continue

                # Preservar metadata original
                chunk.metadata.update(doc.metadata)

                # Agregar metadata del chunk
                chunk.metadata.update({
                    "document_title": doc_title,
                    "char_count": len(chunk.page_content),
                    "word_count": len(chunk.page_content.split()),
                })

                # Extraer contexto adicional del chunk
                section_context = self.extract_section_context(chunk.page_content)
                chunk.metadata.update(section_context)

                all_chunks.append(chunk)

        # Agrupar por archivo y asignar índices
        chunks_by_file = {}
        for chunk in all_chunks:
            filename = chunk.metadata.get('filename', 'unknown')
            if filename not in chunks_by_file:
                chunks_by_file[filename] = []
            chunks_by_file[filename].append(chunk)

        # Asignar chunk_index y total_chunks
        final_chunks = []
        for filename, file_chunks in chunks_by_file.items():
            total_chunks_for_file = len(file_chunks)

            for i, chunk in enumerate(file_chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": total_chunks_for_file,
                    "chunk_id": hashlib.md5(f"{filename}_{i}".encode()).hexdigest()[:16],
                })

                # Agregar contexto al primer chunk
                doc_title = chunk.metadata.get("document_title", "")
                if doc_title and i == 0:
                    chunk.page_content = f"[Documento: {doc_title}]\n\n{chunk.page_content}"

                final_chunks.append(chunk)

        logger.info(f"✅ Chunks útiles creados: {len(final_chunks)}")
        logger.info(f"🗑️  Chunks filtrados: {self.stats['filtered_chunks']}/{self.stats['total_chunks']}")
        logger.info(f"📊 Promedio de chunks por archivo: {len(final_chunks) / len(chunks_by_file):.1f}")
        logger.info(f"📁 Archivos únicos procesados: {len(chunks_by_file)}")

        return final_chunks

    def _extract_title(self, content: str) -> str:
        """Extrae el título del contenido del documento."""
        lines = content.split('\n')
        for line in lines[:10]:  # Buscar en las primeras 10 líneas
            clean_line = line.strip()
            if len(clean_line) > 10 and len(clean_line) < 150:
                # No incluir líneas que son solo puntos o números
                if not re.match(r'^[\s\.]+$', clean_line) and not clean_line.isdigit():
                    return clean_line
        return "Documento Legal"

    def create_vectorstore(self, chunks: List[Document], persist_dir: str) -> Chroma:
        """Crea la base de datos vectorial con los chunks."""
        logger.info("🔢 Generando embeddings y creando vectorstore...")

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
            "min_chunk_size": min(len(c.page_content) for c in chunks) if chunks else 0,
            "max_chunk_size": max(len(c.page_content) for c in chunks) if chunks else 0,
            "categories": {},
            "file_types": {},
            "documents": set(),
            "chunks_with_article": 0,
            "chunks_with_chapter": 0,
        }

        for chunk in chunks:
            category = chunk.metadata.get("category", "unknown")
            file_type = chunk.metadata.get("file_type", "unknown")
            filename = chunk.metadata.get("filename", "unknown")

            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["documents"].add(filename)

            if chunk.metadata.get("article"):
                stats["chunks_with_article"] += 1
            if chunk.metadata.get("chapter"):
                stats["chunks_with_chapter"] += 1

        stats["unique_documents"] = len(stats["documents"])
        del stats["documents"]

        return stats


def main():
    """Función principal de ingesta optimizada."""
    logger.info("=" * 60)
    logger.info("🚀 SISTEMA OPTIMIZADO DE INGESTA DE DOCUMENTOS")
    logger.info("=" * 60)

    # Inicializar procesador
    processor = OptimizedDocumentProcessor()

    # Paso 1: Cargar documentos
    logger.info("\n📥 PASO 1: Carga y filtrado de documentos")
    documents = processor.load_documents(DATA_DIR)

    if not documents:
        logger.error("❌ No se encontraron documentos útiles para procesar")
        return 1

    # Paso 2: Crear chunks inteligentes
    logger.info("\n🔪 PASO 2: Chunking semántico optimizado")
    chunks = processor.create_smart_chunks(documents)

    if not chunks:
        logger.error("❌ No se crearon chunks útiles")
        return 1

    # Paso 3: Generar estadísticas
    logger.info("\n📊 PASO 3: Estadísticas del procesamiento")
    stats = processor.get_statistics(chunks)

    logger.info(f"  • Total de chunks útiles: {stats['total_chunks']}")
    logger.info(f"  • Documentos únicos: {stats['unique_documents']}")
    logger.info(f"  • Tamaño promedio: {stats['avg_chunk_size']:.0f} caracteres")
    logger.info(f"  • Tamaño mínimo: {stats['min_chunk_size']:.0f} caracteres")
    logger.info(f"  • Tamaño máximo: {stats['max_chunk_size']:.0f} caracteres")
    logger.info(f"  • Total de caracteres: {stats['total_characters']:,}")
    logger.info(f"  • Chunks con artículo: {stats['chunks_with_article']}")
    logger.info(f"  • Chunks con capítulo: {stats['chunks_with_chapter']}")

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
    logger.info("✅ INGESTA OPTIMIZADA COMPLETADA EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info(f"\n📍 Base de datos creada en: {CHROMA_DIR}")
    logger.info(f"📊 Total de chunks indexados: {collection_count}")
    logger.info(f"🎯 Mejoras aplicadas:")
    logger.info(f"   • Filtrado de {processor.stats['filtered_pages']} páginas inútiles")
    logger.info(f"   • Filtrado de {processor.stats['filtered_chunks']} chunks muy cortos")
    logger.info(f"   • Contexto de artículos y capítulos preservado")
    logger.info(f"   • Chunks de tamaño óptimo (min {stats['min_chunk_size']:.0f}, promedio {stats['avg_chunk_size']:.0f})")
    logger.info("\n🎯 El sistema está listo para respuestas más precisas\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
