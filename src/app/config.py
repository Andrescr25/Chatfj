from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Chat Facilitadores Judiciales"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    
    # Paths
    
    # LLM Configuration
    # Proveedor principal. Se conserva por compatibilidad: si LLM_CHAIN está
    # vacío, la cascada es solo este.
    LLM_PROVIDER: str = "groq"
    # Cascada de respaldo, en orden y separada por comas (ej.: "gemini,omniroute,groq").
    # Si un proveedor se queda sin cupo, la respuesta la da el siguiente.
    LLM_CHAIN: str = ""
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "openai/gpt-4-turbo"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # OmniRoute: pasarela compatible con OpenAI que enruta entre muchos
    # proveedores y hace su propia cascada del lado del servidor.
    OMNIROUTE_API_KEY: Optional[str] = None
    OMNIROUTE_BASE_URL: str = "https://cloud.omniroute.online/v1"
    OMNIROUTE_MODEL: str = "openai/gpt-oss-120b"

    
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
    FIREBASE_STORAGE_BUCKET: Optional[str] = None

    # Administración
    # Correos con acceso de administración aunque no tengan el custom claim.
    # Sirve para el arranque inicial y para no quedarse nunca fuera del sistema.
    ADMIN_EMAILS: str = ""
    # Nombre anterior de la misma variable; se sigue leyendo por compatibilidad.
    SUPERADMIN_EMAILS: str = ""

    # Ingesta de documentos
    UPLOAD_DIR: str = "data/uploads"
    MAX_UPLOAD_MB: int = 25
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBED_BATCH_SIZE: int = 20

    # CORS: dominios adicionales separados por coma
    EXTRA_CORS_ORIGINS: str = ""

    model_config = {
        "env_file": "config/config.env",
        "case_sensitive": True,
        "extra": "ignore"  # Allow extra env vars
    }

    @property
    def llm_chain(self) -> List[str]:
        """Proveedores en el orden en que se deben intentar."""
        crudo = self.LLM_CHAIN.strip() or self.LLM_PROVIDER
        vistos, orden = set(), []
        for nombre in crudo.split(","):
            nombre = nombre.strip().lower()
            if nombre and nombre not in vistos:
                vistos.add(nombre)
                orden.append(nombre)
        return orden

    @property
    def admin_emails(self) -> List[str]:
        crudo = self.ADMIN_EMAILS or self.SUPERADMIN_EMAILS
        return [e.strip().lower() for e in crudo.split(",") if e.strip()]

    @property
    def cors_origins(self) -> List[str]:
        base = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://chatfj.web.app",
            "https://chatfj-26458.web.app",
            "https://chatfj-26458.firebaseapp.com",
        ]
        extra = [o.strip() for o in self.EXTRA_CORS_ORIGINS.split(",") if o.strip()]
        return base + extra

settings = Settings()
