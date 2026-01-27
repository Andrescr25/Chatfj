import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Chat Facilitadores Judiciales"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    
    # Paths
    PERSIST_DIR: str = "data/chroma_db"
    
    # LLM Configuration
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "moonshotai/kimi-k2-instruct-0905"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "openai/gpt-4-turbo"
    
    # RAG / Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "chatfj-legal-index"
    
    # Search
    SEARCH_TOP_K: int = 4
    
    # Firebase
    FIREBASE_CREDENTIALS_PATH: str = "config/firebase-adminsdk.json"
    
    model_config = {
        "env_file": "config/config.env",
        "case_sensitive": True,
        "extra": "ignore"  # Allow extra env vars
    }

settings = Settings()
