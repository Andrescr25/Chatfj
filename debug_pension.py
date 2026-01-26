
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock environment variables
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
os.environ["CHROMA_PERSIST_DIRECTORY"] = "data/chroma_db"

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY")

async def test_retrieval(query):
    print(f"🔎 Querying: {query}")
    
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    
    embedder = await loop.run_in_executor(
        executor, 
        lambda: SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )
    
    vectordb = await loop.run_in_executor(
        executor,
        lambda: Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedder,
            collection_name="legal_documents"
        )
    )
    
    results = vectordb.similarity_search_with_score(query, k=5)
    
    print("\n--- RESULTS ---")
    for doc, score in results:
        meta = doc.metadata
        filename = meta.get('filename') or meta.get('source', 'Unknown')
        print(f"\n📄 File: {filename}")
        print(f"📊 Score: {score}")
        print(f"📝 Content Snippet: {doc.page_content[:400]}...")
        print("-" * 40)

if __name__ == "__main__":
    q = "apremio corporal pago pension alimentaria carcel"
    asyncio.run(test_retrieval(q))
