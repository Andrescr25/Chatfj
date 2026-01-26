#!/usr/bin/env python3
"""
Sistema PREMIUM de ingesta de documentos para RAG
MODO: MÁXIMA CALIDAD - Sin compromisos en velocidad

Configuración optimizada para respuestas perfectas:
- Chunks GRANDES (3500 caracteres) con contexto completo
- Overlap MASIVO (800 caracteres) para continuidad perfecta
- Mínimo filtrado: preservar toda información relevante
- Metadata RICA con contexto legal completo
- Detección avanzada de estructura legal
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
import hashlib
from datetime import datetime

# Configurar logging detallado
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

# Configuración de chunking PREMIUM - Máxima Calidad
CHUNK_CONFIG = {
    "chunk_size": 3500,  # Chunks MUY grandes para contexto completo
    "chunk_overlap": 800,  # Overlap MASIVO para no perder continuidad
    "min_chunk_size": 100,  # Mínimo muy bajo para preservar toda info relevante
    "separators": [
        "\n\n\n\n",    # Separación entre secciones muy grandes
        "\n\n\n",      # Separación entre secciones grandes
        "\n\n",        # Separación entre párrafos
        "\nArtículo ", # Separar artículos legales (CRÍTICO)
        "\nARTÍCULO ", # Variante mayúsculas
        "\nCapítulo ", # Separar capítulos
        "\nCAPÍTULO ", # Variante mayúsculas
        "\nSección ",  # Separar secciones
        "\nTítulo ",   # Separar títulos
        "\n",          # Saltos de línea
        ". ",          # Fin de oración
        "; ",          # Punto y coma
        ", ",          # Comas
        " ",           # Espacios
        ""             # Fallback
    ]
}

# Patrones MÍNIMOS para detectar contenido totalmente inútil
# Solo filtramos lo absolutamente necesario
USELESS_PATTERNS = [
    r'^[\s\.]+$',  # Solo espacios y puntos vacíos
    r'^PRESENTACI[OÓ]N\s*$',  # Título de presentación vacío
    r'^CONTENIDO\s*$',  # Título de contenido vacío
    r'^[ÍI]NDICE\s*$',  # Título de índice vacío
    r'^\d+\s*$',  # Solo un número de página aislado
]


class PremiumDocumentProcessor:
    """Procesador PREMIUM de documentos - Máxima calidad sin compromisos."""

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
            "preserved_chunks": 0,
            "chunks_with_legal_context": 0,
            "total_chars": 0,
        }

    def is_useless_content(self, text: str) -> bool:
        """
        Detecta SOLO contenido completamente inútil.
        MODO PREMIUM: Preservar casi todo.
        """
        text_clean = text.strip()

        # Solo muy muy corto (casi vacío)
        if len(text_clean) < 20:
            return True

        # Coincide con patrones totalmente inútiles
        for pattern in USELESS_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                return True

        # Casi todo son puntos (índice de puntos suspensivos)
        if text_clean.count('.') > len(text_clean) * 0.5:
            return True

        return False

    def extract_advanced_legal_context(self, text: str) -> Dict[str, Any]:
        """
        Extrae contexto legal AVANZADO con máximo detalle.
        """
        context = {
            'legal_references': [],
            'has_legal_content': False,
        }

        # Buscar artículos (múltiples variantes)
        article_patterns = [
            r'Art[ií]culo\s+(\d+[a-z]*\.?\s*(?:bis|ter)?)',
            r'Art\.\s*(\d+[a-z]*)',
            r'ARTÍCULO\s+(\d+)',
        ]
        for pattern in article_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                context['article'] = matches[0].strip()
                context['all_articles'] = list(set(matches))
                context['legal_references'].extend([f"Art. {m}" for m in matches])
                context['has_legal_content'] = True
                break

        # Buscar capítulos
        chapter_patterns = [
            r'Cap[ií]tulo\s+([IVXLCDM]+|\d+)',
            r'CAPÍTULO\s+([IVXLCDM]+|\d+)',
        ]
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                context['chapter'] = match.group(1)
                context['has_legal_content'] = True
                break

        # Buscar secciones
        section_match = re.search(r'Secci[oó]n\s+([IVXLCDM]+|\d+)', text, re.IGNORECASE)
        if section_match:
            context['section'] = section_match.group(1)
            context['has_legal_content'] = True

        # Buscar títulos
        title_match = re.search(r'T[ií]tulo\s+([IVXLCDM]+|\d+)', text, re.IGNORECASE)
        if title_match:
            context['title'] = title_match.group(1)
            context['has_legal_content'] = True

        # Buscar referencias a leyes
        law_patterns = [
            r'Ley\s+N[oº°]?\s*(\d+)',
            r'Ley\s+(\d+)',
            r'C[óo]digo\s+(Civil|Penal|Familia|Trabajo|Procesal)',
        ]
        for pattern in law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                context['legal_references'].extend([f"Ley {m}" if isinstance(m, str) and m.isdigit() else m for m in matches])
                context['has_legal_content'] = True

        # Detectar tipo de contenido legal
        if 'procedimiento' in text.lower() or 'proceso' in text.lower():
            context['content_type'] = 'procesal'
        elif 'sanción' in text.lower() or 'multa' in text.lower() or 'pena' in text.lower():
            context['content_type'] = 'sancionatorio'
        elif 'derecho' in text.lower() or 'obligación' in text.lower():
            context['content_type'] = 'sustantivo'

        # Contar términos legales importantes
        legal_terms = [
            'artículo', 'ley', 'código', 'reglamento', 'decreto',
            'derecho', 'obligación', 'procedimiento', 'proceso',
            'sentencia', 'resolución', 'dictamen'
        ]
        legal_term_count = sum(text.lower().count(term) for term in legal_terms)
        context['legal_density'] = legal_term_count / max(len(text.split()), 1)

        return context

    def load_documents(self, data_dir: str) -> List[Document]:
        """Carga documentos con filtrado MÍNIMO - preservar casi todo."""
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.error(f"❌ Directorio {data_dir} no existe")
            return documents

        logger.info(f"📂 Cargando documentos en MODO PREMIUM desde: {data_dir}")
        logger.info(f"⚙️  Configuración: chunks={CHUNK_CONFIG['chunk_size']}, overlap={CHUNK_CONFIG['chunk_overlap']}")

        # Cargar PDFs
        pdf_files = list(data_path.glob("**/*.pdf"))
        logger.info(f"📄 Encontrados {len(pdf_files)} archivos PDF")

        for pdf_file in pdf_files:
            try:
                logger.info(f"  📄 Procesando PDF: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()

                # Filtrado MÍNIMO - solo páginas totalmente vacías
                filtered_docs = []
                for doc in docs:
                    self.stats["total_pages"] += 1

                    # Solo filtrar lo absolutamente inútil
                    if self.is_useless_content(doc.page_content):
                        self.stats["filtered_pages"] += 1
                        continue

                    # Agregar metadata RICA
                    doc.metadata.update({
                        "filename": pdf_file.name,
                        "source": str(pdf_file),
                        "file_type": "pdf",
                        "category": self._detect_category(pdf_file.name),
                        "upload_date": datetime.now().isoformat(),
                        "page_length": len(doc.page_content),
                    })

                    # Agregar contexto legal AVANZADO
                    legal_context = self.extract_advanced_legal_context(doc.page_content)

                    # Convertir listas a strings para ChromaDB
                    for key, value in legal_context.items():
                        if isinstance(value, list):
                            legal_context[key] = ", ".join(str(v) for v in value)

                    doc.metadata.update(legal_context)

                    filtered_docs.append(doc)

                documents.extend(filtered_docs)
                logger.info(f"    ✅ {len(filtered_docs)}/{len(docs)} páginas preservadas")

            except Exception as e:
                logger.error(f"    ❌ Error procesando {pdf_file.name}: {e}")

        # Cargar archivos de texto
        txt_files = list(data_path.glob("**/*.txt")) + list(data_path.glob("**/*.md"))
        if txt_files:
            logger.info(f"📝 Encontrados {len(txt_files)} archivos de texto")

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

                        legal_context = self.extract_advanced_legal_context(doc.page_content)

                        # Convertir listas a strings para ChromaDB
                        for key, value in legal_context.items():
                            if isinstance(value, list):
                                legal_context[key] = ", ".join(str(v) for v in value)

                        doc.metadata.update(legal_context)

                        documents.append(doc)

                logger.info(f"    ✅ {len(docs)} documentos cargados")
            except Exception as e:
                logger.error(f"    ❌ Error procesando {txt_file.name}: {e}")

        logger.info(f"✅ Total de documentos cargados: {len(documents)}")
        logger.info(f"🗑️  Páginas filtradas (solo completamente vacías): {self.stats['filtered_pages']}/{self.stats['total_pages']}")
        return documents

    def _detect_category(self, filename: str) -> str:
        """Detecta categorías con más precisión."""
        filename_lower = filename.lower()

        categories = {
            "pension_alimentaria": ["pension", "alimentaria", "alimenticio", "alimentos"],
            "familia": ["familia", "familiar", "matrimonio", "divorcio", "tutela", "curatela"],
            "laboral": ["laboral", "trabajo", "despido", "salario", "codigo_trabajo"],
            "penal": ["penal", "delito", "condena", "codigo_penal"],
            "civil": ["civil", "contrato", "propiedad", "codigo_civil"],
            "procesal": ["procesal", "proceso", "procedimiento"],
            "constitucional": ["constitucion", "constitucional"],
            "agrario": ["agrario", "pesca", "agricultura"],
            "transito": ["transito", "vehiculo", "conductor"],
            "violencia_domestica": ["violencia", "domestica", "proteccion"],
        }

        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category

        return "general"

    def create_premium_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Crea chunks PREMIUM con máxima preservación de información.
        """
        logger.info("🔪 Creando chunks PREMIUM (grandes, con contexto completo)...")

        all_chunks = []
        chunk_sizes = []

        for doc in documents:
            # Extraer título del documento
            doc_title = self._extract_title(doc.page_content)

            # Crear chunks GRANDES
            chunks = self.text_splitter.split_documents([doc])

            for i, chunk in enumerate(chunks):
                self.stats["total_chunks"] += 1

                # Preservar chunks incluso si son pequeños (mínimo 100 chars)
                if len(chunk.page_content.strip()) < CHUNK_CONFIG["min_chunk_size"]:
                    continue

                self.stats["preserved_chunks"] += 1
                chunk_sizes.append(len(chunk.page_content))
                self.stats["total_chars"] += len(chunk.page_content)

                # Enriquecer metadata del chunk
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata.get('filename', 'unknown')}_{i}",
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "document_title": doc_title,
                })

                # Re-extraer contexto legal por chunk (puede haber cambiado)
                chunk_legal_context = self.extract_advanced_legal_context(chunk.page_content)

                # Convertir listas a strings para ChromaDB
                for key, value in chunk_legal_context.items():
                    if isinstance(value, list):
                        chunk_legal_context[key] = ", ".join(str(v) for v in value)

                chunk.metadata.update(chunk_legal_context)

                if chunk_legal_context.get('has_legal_content'):
                    self.stats["chunks_with_legal_context"] += 1

                # Crear un ID único basado en contenido
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
                chunk.metadata["content_hash"] = content_hash

                all_chunks.append(chunk)

        # Estadísticas detalladas
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)

            logger.info(f"✅ Chunks creados: {len(all_chunks)}")
            logger.info(f"📊 Tamaño promedio: {avg_size:.0f} caracteres")
            logger.info(f"📊 Tamaño mínimo: {min_size} caracteres")
            logger.info(f"📊 Tamaño máximo: {max_size} caracteres")
            logger.info(f"⚖️  Chunks con contexto legal: {self.stats['chunks_with_legal_context']}")
            logger.info(f"📈 Densidad legal: {self.stats['chunks_with_legal_context']/len(all_chunks)*100:.1f}%")

        return all_chunks

    def _extract_title(self, text: str) -> str:
        """Extrae el título del documento."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            # El título suele estar en las primeras líneas
            title_candidates = lines[:5]
            for candidate in title_candidates:
                if 10 < len(candidate) < 200 and not candidate.isdigit():
                    return candidate
        return "Sin título"

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """Crea la base de datos vectorial con embeddings de alta calidad."""
        logger.info("🧠 Inicializando embeddings (modelo premium)...")

        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

        logger.info(f"💾 Creando ChromaDB en: {CHROMA_DIR}")

        # Eliminar DB existente
        if os.path.exists(CHROMA_DIR):
            import shutil
            logger.info("🗑️  Eliminando base de datos anterior...")
            shutil.rmtree(CHROMA_DIR)

        # Crear nueva DB con el nombre de colección correcto
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="legal_documents",  # Mismo nombre que usa el backend
        )

        logger.info("✅ Base de datos vectorial creada")
        return self.vectordb

    def process_all(self):
        """Ejecuta el pipeline completo de procesamiento PREMIUM."""
        logger.info("="*80)
        logger.info("🏆 MODO PREMIUM: Procesamiento para MÁXIMA CALIDAD")
        logger.info("="*80)

        # 1. Cargar documentos
        documents = self.load_documents(DATA_DIR)
        if not documents:
            logger.error("❌ No se cargaron documentos")
            return

        # 2. Crear chunks premium
        chunks = self.create_premium_chunks(documents)
        if not chunks:
            logger.error("❌ No se crearon chunks")
            return

        # 3. Crear vectorstore
        self.create_vectorstore(chunks)

        # 4. Resumen final
        logger.info("="*80)
        logger.info("✅ PROCESAMIENTO PREMIUM COMPLETADO")
        logger.info("="*80)
        logger.info(f"📄 Páginas procesadas: {self.stats['total_pages']}")
        logger.info(f"🗑️  Páginas filtradas: {self.stats['filtered_pages']}")
        logger.info(f"📦 Chunks creados: {self.stats['preserved_chunks']}")
        logger.info(f"⚖️  Chunks con contexto legal: {self.stats['chunks_with_legal_context']}")
        logger.info(f"📊 Caracteres totales: {self.stats['total_chars']:,}")
        if self.stats['preserved_chunks'] > 0:
            avg_chunk_size = self.stats['total_chars'] / self.stats['preserved_chunks']
            logger.info(f"📊 Tamaño promedio por chunk: {avg_chunk_size:.0f} caracteres")
        logger.info("="*80)


def main():
    """Función principal."""
    processor = PremiumDocumentProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()
