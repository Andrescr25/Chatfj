import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load config
load_dotenv("config/config.env")
load_dotenv()

def diagnose():
    print("🔍 Diagnóstico de RAG y Pinecone")
    print("================================")
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    env = os.getenv("PINECONE_ENV")
    
    if not api_key:
        print("❌ PINECONE_API_KEY no encontrada")
        return

    try:
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        index_names = [i.name for i in indexes]
        
        print(f"Indices encontrados: {index_names}")
        
        if index_name not in index_names:
            print(f"❌ Índice '{index_name}' no encontrado en Pinecone")
            return
            
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        print(f"✅ Estadísticas del índice '{index_name}':")
        print(stats)
        
        # Check explicit dimension from describe_index_stats or list_indexes?
        # describe_index_stats includes 'dimension' usually
        
        info = pc.describe_index(index_name)
        print(f"📏 Dimensión del índice: {info.dimension}")
        print(f"📦 Modelo configurado en env: {os.getenv('EMBEDDING_MODEL_NAME')}")
        
        # Test Embeddings
        print("\n🧪 Probando Embeddings...")
        try:
            from src.app.core.rag.embeddings import EmbeddingService
            svc = EmbeddingService()
            print(f"   Cliente de embeddings: {type(svc.client)}")
            print(f"   Modelo: {getattr(svc.client, 'model_name', 'Unknown')}")
            
            start = time.time()
            vec = svc.embed_query_sync("Test query check")
            elapsed = time.time() - start
            
            if vec:
                print(f"✅ Embedding generado correctamente. Dimensión: {len(vec)}")
                print(f"⏱️ Tiempo de generación: {elapsed:.2f}s")
                
                if len(vec) != info.dimension:
                    print(f"❌ MISMATCH: Vector ({len(vec)}) != Índice ({info.dimension})")
                    print("⚠️ Esto causará que la búsqueda falle siempre.")
                else:
                    print("✅ Dimensiones coinciden.")
            else:
                print("❌ Falló la generación de embedding (retornó vacío)")
                
        except Exception as e:
            print(f"❌ Error probando embeddings: {e}")

    except Exception as e:
        print(f"❌ Error conectando a Pinecone: {e}")

if __name__ == "__main__":
    diagnose()
