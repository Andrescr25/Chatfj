
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

async def inspect_chunks(query):
    print(f"🔎 Inspecting chunks for query: '{query}'")
    
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
    
    # Get top 3 chunks
    results = vectordb.similarity_search_with_score(query, k=3)
    
    print("\n--- RETRIEVED CHUNKS ---")
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        print(f"\n🧩 CHUNK #{i+1} (Score: {score:.4f})")
        print(f"📄 Source: {meta.get('filename', 'Unknown')}")
        print(f"📌 Section: {meta.get('article', 'N/A')} | Chapter: {meta.get('chapter', 'N/A')}")
        print(f"📝 Content:\n{'-'*20}\n{doc.page_content}\n{'-'*20}")
        print("-" * 40)

if __name__ == "__main__":
    # Query designed to hit the problematic concepts mentioned by user
    q = "apremio corporal embargo salario nuevo codigo familia beneficio buscar trabajo"
    asyncio.run(inspect_chunks(q))
