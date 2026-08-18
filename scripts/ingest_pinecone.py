#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv("config/config.env")
load_dotenv(find_dotenv())

DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatfj-legal-index")
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
from langchain_huggingface import HuggingFaceEndpointEmbeddings
hf_embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model=EMBEDDING_MODEL
)
def read_pdf(file_path):
    text = ""
    try:
        # Use str(Path) directly, avoid decoding issues in Mac APFS
        safe_path = Path(file_path).resolve()
        with open(safe_path, 'rb') as f:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader = PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

import time
def get_embeddings(texts):
    for attempt in range(5):
        try:
            res = hf_embeddings.embed_documents(texts)
            time.sleep(0.5) # Respirar entre llamadas para evitar Rate Limits
            return res
        except Exception as e:
            logger.warning(f"API Error (Attempt {(attempt+1)}): {e}")
            time.sleep(10) # Mayor espera en caso de Timeout 504
    return None

def main():
    if not PINECONE_API_KEY:
        logger.error("No PINECONE_API_KEY")
        sys.exit(1)
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    path = Path(DATA_DIR)
    pdf_files = list(path.glob("**/*.pdf"))
    logger.info(f"Encontrados {len(pdf_files)} PDFs.")
    
    total_chunks = 0
    
    for pdf in pdf_files:
        logger.info(f"Procesando: {pdf.name}")
        text = read_pdf(str(pdf))
        if not text:
            continue
            
        chunks = chunk_text(text)
        logger.info(f" - {len(chunks)} fragmentos extraidos.")
        
        # Procesar en lotes de 20 (Limita carga a HF Cloud para evitar 504 Timeout)
        batch_size = 20 
        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i:i+batch_size]
            batch_ids = [f"{pdf.name}_chunk_{i+j}" for j in range(len(batch_texts))]
            batch_metadata = [{"source": str(pdf), "text": txt} for txt in batch_texts]
            
            vectors = get_embeddings(batch_texts)
            if vectors:
                upsert_data = list(zip(batch_ids, vectors, batch_metadata))
                index.upsert(vectors=upsert_data, namespace="")
                total_chunks += len(vectors)
                logger.info(f" - Subidos {len(vectors)} fragmentos (Total: {total_chunks})")

    logger.info(f"✅ Ingesta completa. {total_chunks} fragmentos subidos en total.")

if __name__ == "__main__":
    main()
