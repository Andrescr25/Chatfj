import os
from typing import List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Chat Facilitadores Judiciales"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    
    # Paths
    
    # LLM Configuration
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "openai/gpt-4-turbo"
    
    # RAG / Embeddings
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
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
