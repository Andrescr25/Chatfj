import os

import firebase_admin
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from firebase_admin import credentials

from src.app.api.v1.api import api_router
from src.app.config import settings
from src.app.utils.logging import logger

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend para Asistente de Facilitadores Judiciales",
    version="2.0.0"
)

# Startup Checks
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando ChatFJ API...")
    
    # Inicializar Firebase Admin SDK
    try:
        if not firebase_admin._apps:
            cred_path = settings.FIREBASE_CREDENTIALS_PATH
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"✅ Firebase Admin SDK inicializado usando credenciales de: {cred_path}")
            else:
                firebase_admin.initialize_app()
                logger.info("✅ Firebase Admin SDK inicializado con credenciales por defecto de Google Cloud.")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo inicializar Firebase Admin SDK (puede requerir credenciales): {e}")

    if not settings.PINECONE_API_KEY:
        logger.warning("⚠️ PINECONE_API_KEY no encontrada. La búsqueda vectorial fallará.")
    else:
        logger.info(f"✅ PINECONE_API_KEY detectada (Index: {settings.PINECONE_INDEX_NAME})")
        
    if settings.admin_emails:
        logger.info(f"👑 Administradores por configuración: {', '.join(settings.admin_emails)}")
    else:
        logger.warning(
            "⚠️ ADMIN_EMAILS vacío. Si nadie tiene el rol asignado, "
            "ejecute: python scripts/bootstrap_admin.py --email <correo>"
        )

    if settings.FIREBASE_STORAGE_BUCKET:
        logger.info(f"📦 Firebase Storage configurado: {settings.FIREBASE_STORAGE_BUCKET}")
    else:
        logger.info("📦 Sin Firebase Storage: los originales se guardan en disco local.")

    if not settings.HUGGINGFACEHUB_API_TOKEN:
        logger.critical("🚨 HUGGINGFACEHUB_API_TOKEN no encontrada. El sistema intentará usar modelos locales y podría quedarse sin RAM (Error '0').")
    else:
        logger.info("✅ HUGGINGFACEHUB_API_TOKEN detectada.")

# CORS Configuration
# Lista explícita de dominios: con credenciales habilitadas, el comodín "*"
# deja la API abierta a cualquier sitio. Se amplía con EXTRA_CORS_ORIGINS.
origins = settings.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Devuelve los errores de validación como un texto legible.

    Por defecto FastAPI responde con una lista de objetos, que las interfaces
    terminan mostrando como "[object Object]".
    """
    mensajes = []
    for error in exc.errors():
        msg = str(error.get("msg", "")).replace("Value error, ", "")
        campo = error.get("loc", [])[-1] if error.get("loc") else ""
        if msg and campo and campo not in ("body",):
            mensajes.append(f"{campo}: {msg}" if msg[0].islower() else msg)
        elif msg:
            mensajes.append(msg)

    return JSONResponse(
        status_code=422,
        content={"detail": " ".join(mensajes) or "Datos inválidos en la solicitud."},
    )


@app.get("/")
def read_root():
    return {"status": "online", "project": "ChatFJ API", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}

# Static Files (for favicon, or simple serving if needed)
# Se monta de último: un mount en "/" captura toda ruta no declarada antes,
# así que /health y /docs deben registrarse primero.
os.makedirs("frontend/build", exist_ok=True)
# Mounting only if it exists, otherwise it's just an API
if os.path.isdir("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=port, reload=settings.DEBUG)
