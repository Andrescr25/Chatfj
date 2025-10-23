# Sistema de Facilitadores Judiciales - Costa Rica

Sistema inteligente de asistencia legal con IA híbrida.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

## 🚀 Inicio Rápido

### Desarrollo Local

```bash
# Opción 1: Script automático
./start.sh

# Opción 2: Manual
# Terminal 1 - Backend
source venv/bin/activate
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm start
```

Accede en:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

### Deploy en Render (Gratis)

Ver **[RENDER.md](RENDER.md)** para instrucciones completas.

## 📋 Requisitos

- Python 3.9+
- Node.js 14+
- API Key de Groq (gratis en [console.groq.com](https://console.groq.com))

## ⚙️ Configuración

### Local
Edita `config/config.env`:
```env
GROQ_API_KEY=tu_api_key_aqui
USE_GROQ_API=true
```

### Render
Configura en el dashboard:
- `GROQ_API_KEY`
- `REACT_APP_API_URL`

## 🎯 Características

- ✨ Interfaz moderna tipo ChatGPT
- ⚡ Respuestas instantáneas (< 1s)
- 🤖 IA híbrida (MockLLM + Groq)
- 📱 Responsive
- 💬 Historial de conversaciones
- 🚀 Deploy gratis en Render

## 🛠️ Tecnologías

**Backend**: FastAPI, Groq API (Llama 3.1), ChromaDB, LangChain  
**Frontend**: React 18, CSS moderno

## 📁 Estructura

```
sistema-facilitadores-judiciales/
├── frontend/              # React App
│   ├── public/           # Archivos públicos
│   └── src/              # Código fuente React
│       ├── App.js        # Componente principal
│       ├── App.css       # Estilos
│       └── index.js      # Entry point
├── src/                  # Backend API
│   └── api.py           # FastAPI
├── config/               # Configuración
│   ├── config.env       # Variables de entorno
│   └── security.py      # Seguridad
├── data/docs/            # PDFs legales (35 documentos)
├── scripts/              # Scripts auxiliares
│   └── ingest.py        # Procesar documentos
├── start.sh             # Inicio rápido
├── build.sh             # Build producción
├── render.yaml          # Config Render
├── Procfile             # Deploy config
└── requirements.txt     # Dependencias Python
```

## 💰 Costos

**$0/mes** con plan gratuito de Render + Groq API

---

**Deploy**: Ver [RENDER.md](RENDER.md)  
Sistema de Facilitadores Judiciales de Costa Rica 🇨🇷