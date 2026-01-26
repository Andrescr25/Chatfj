import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.app.config import settings
from src.app.utils.logging import logger
from src.app.api.v1.api import api_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend para Asistente de Facilitadores Judiciales",
    version="2.0.0"
)

# CORS Configuration
origins = [
    "http://localhost:3000",
    "https://chatfj.web.app",
    "https://chatfj-26458.web.app",
    "*" # Permissive for development/testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Static Files (for favicon, or simple serving if needed)
# Ensure directory exists or this might fail. Making optional or creating dir.
os.makedirs("frontend/build", exist_ok=True)
# Mounting only if it exists, otherwise it's just an API
if os.path.isdir("frontend/build"):
    app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=port, reload=settings.DEBUG)
