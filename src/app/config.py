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
    # "gemini-flash-latest" apunta siempre al flash vigente: gemini-2.5-flash
    # quedó descontinuado para cuentas nuevas y dejó de responder.
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    # HuggingFace: el mismo token que ya se usa para los embeddings sirve para
    # generar respuestas, con API compatible con OpenAI. Respaldo sin registro extra.
    HUGGINGFACE_CHAT_MODEL: str = "zai-org/GLM-5.2"
    HUGGINGFACE_CHAT_BASE_URL: str = "https://router.huggingface.co/v1"

    # SambaNova: API compatible con OpenAI. Ojo: desde agosto de 2026 las
    # cuentas nuevas exigen método de pago; sin él responde 402.
    SAMBANOVA_API_KEY: Optional[str] = None
    SAMBANOVA_BASE_URL: str = "https://api.sambanova.ai/v1"
    SAMBANOVA_MODEL: str = "gpt-oss-120b"

    # Cerebras: capa gratuita propia y API compatible con OpenAI.
    # Sirve de respaldo real cuando el proveedor principal se queda sin cupo.
    CEREBRAS_API_KEY: Optional[str] = None
    CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"
    CEREBRAS_MODEL: str = "gpt-oss-120b"

    # OmniRoute: pasarela compatible con OpenAI. Se deja disponible como opción,
    # pero exige hospedarla y conectarle llaves propias de otros proveedores.
    OMNIROUTE_API_KEY: Optional[str] = None
    OMNIROUTE_BASE_URL: str = "https://cloud.omniroute.online/v1"
    OMNIROUTE_MODEL: str = "openai/gpt-oss-120b"

    
    # RAG / Embeddings
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = None
    # Llaves adicionales de HuggingFace, separadas por coma. El crédito es por
    # cuenta, así que una llave de otra cuenta da cupo propio. Al ser el mismo
    # modelo, sus vectores son comparables y usan el mismo espacio del índice.
    HUGGINGFACE_TOKENS_EXTRA: str = ""

    # Respaldo de embeddings, por API compatible con OpenAI. DEBE servir el
    # mismo modelo: vectores de otro modelo no son comparables con el índice.
    # DeepInfra sirve intfloat/multilingual-e5-large.
    EMBEDDINGS_FALLBACK_API_KEY: Optional[str] = None
    EMBEDDINGS_FALLBACK_BASE_URL: str = "https://api.deepinfra.com/v1/openai"

    # Segundo modelo de embeddings (Gemini), en su propio espacio del índice.
    #
    # Desactivado por defecto: su espacio quedó con 990 de 9.036 fragmentos
    # porque la cuota gratuita de embeddings de Gemini permite ~1 petición por
    # minuto, insuficiente para copiar un corpus. Con el espacio incompleto, la
    # búsqueda de respaldo citaría el 11% del acervo como si fuera todo, que es
    # peor que fallar. Se activa cuando el espacio esté completo.
    EMBEDDINGS_GEMINI_ENABLED: bool = False
    EMBEDDINGS_GEMINI_MODEL: str = "gemini-embedding-001"
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "chatfj-legal-index"
    
    # Search
    SEARCH_TOP_K: int = 4
    # Umbral de relevancia POR MODELO: las escalas de similitud no son
    # comparables entre modelos de embeddings. Con e5 lo relevante ronda 0,83;
    # con Gemini, 0,55. Usar el mismo número descartaría todo en un espacio.
    SEARCH_THRESHOLD_E5: float = 0.75
    SEARCH_THRESHOLD_GEMINI: float = 0.45
    
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
    def huggingface_tokens(self) -> List[str]:
        """Llaves de HuggingFace en orden de uso, sin repetidas."""
        crudas = [self.HUGGINGFACEHUB_API_TOKEN or ""]
        crudas += self.HUGGINGFACE_TOKENS_EXTRA.split(",")
        vistas, orden = set(), []
        for t in crudas:
            t = t.strip().strip('"\'').strip()
            if t and t not in vistas:
                vistas.add(t)
                orden.append(t)
        return orden

    @property
    def llm_chain(self) -> List[str]:
        """
        Proveedores en el orden en que se deben intentar.

        Cada entrada puede ser solo el proveedor ("groq") o el proveedor con un
        modelo concreto ("huggingface:zai-org/GLM-5.2"). Lo segundo permite
        encadenar dos modelos del mismo proveedor, que es útil porque las cuotas
        gratuitas suelen contarse por modelo.
        """
        # Se limpian comillas y espacios: al pegar el valor en el panel de
        # Render es fácil que arrastre comillas o un salto de línea, y entonces
        # el primer proveedor quedaba como '"groq' y se descartaba en silencio.
        crudo = self.LLM_CHAIN.strip().strip('"\'').strip() or self.LLM_PROVIDER
        vistos, orden = set(), []
        for entrada in crudo.split(","):
            entrada = entrada.strip().strip('"\'').strip()
            if not entrada:
                continue
            # Solo se parte en el primer ":": los identificadores de modelo
            # pueden contener otro (por ejemplo "openai/gpt-oss-120b:free").
            proveedor, sep, modelo = entrada.partition(":")
            # El proveedor se normaliza; el modelo se respeta tal cual porque
            # distingue mayúsculas.
            normalizada = proveedor.strip().lower() + (sep + modelo.strip() if sep else "")
            if normalizada not in vistos:
                vistos.add(normalizada)
                orden.append(normalizada)
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
