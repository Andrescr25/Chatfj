# 🚀 Guía de Despliegue en Render

Sistema de Facilitadores Judiciales - Costa Rica

## 📋 Pre-requisitos

1. **Cuenta en Render**: Crea una cuenta gratuita en [render.com](https://render.com)
2. **Repositorio Git**: Tu código debe estar en GitHub, GitLab o Bitbucket
3. **API Key de Groq**: Obtén una gratis en [console.groq.com](https://console.groq.com)

## 🎯 Opción 1: Despliegue Automático con render.yaml

### Paso 1: Preparar el Repositorio

```bash
# 1. Asegúrate de estar en la rama main
git checkout main

# 2. Commit de cambios pendientes
git add .
git commit -m "Preparar para despliegue en Render"

# 3. Push al repositorio remoto
git push origin main
```

### Paso 2: Conectar a Render

1. Ve a [dashboard.render.com](https://dashboard.render.com)
2. Click en **"New +"** → **"Blueprint"**
3. Conecta tu repositorio de GitHub/GitLab
4. Render detectará automáticamente `render.yaml`

### Paso 3: Configurar Variables de Entorno

En el dashboard de Render, configura:

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `GROQ_API_KEY` | `gsk_...` | Tu API key de Groq |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Modelo de IA |
| `CHROMA_PERSIST_DIRECTORY` | `/opt/render/project/src/data/chroma_db` | Path de ChromaDB |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Modelo de embeddings |

### Paso 4: Desplegar

1. Click en **"Apply"**
2. Render creará automáticamente:
   - ✅ **Backend API** (chat-fj-api)
   - ✅ **Frontend** (chat-fj-frontend)
   - ✅ **Disco persistente** (1GB para ChromaDB)

### Paso 5: Verificar

- **API**: `https://chat-fj-api.onrender.com/health`
- **Frontend**: `https://chat-fj-frontend.onrender.com`

---

## 🎯 Opción 2: Despliegue Manual

### Backend (API)

1. **New Web Service**
   - Name: `chat-fj-api`
   - Environment: `Python 3.11`
   - Build Command:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     python import_to_langchain_chroma.py
     ```
   - Start Command:
     ```bash
     uvicorn src.api:app --host 0.0.0.0 --port $PORT
     ```

2. **Agregar Disco Persistente**
   - Name: `chroma-data`
   - Mount Path: `/opt/render/project/src/data`
   - Size: 1GB

3. **Variables de Entorno** (ver tabla arriba)

### Frontend (Static Site)

1. **New Static Site**
   - Name: `chat-fj-frontend`
   - Build Command:
     ```bash
     cd frontend && npm install && npm run build
     ```
   - Publish Directory: `frontend/build`

2. **Variable de Entorno**
   - `REACT_APP_API_URL`: URL del backend (ej: `https://chat-fj-api.onrender.com`)

---

## ⚙️ Configuración Adicional

### Actualizar API URL en Frontend

Edita `frontend/src/App.js`:

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'https://chat-fj-api.onrender.com';
```

### Health Check Endpoint

El backend ya incluye un endpoint de health check en `/health`:

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

---

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
# Verifica que requirements.txt esté actualizado
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Error: "ChromaDB not found"
- Verifica que el disco persistente esté montado
- Path correcto: `/opt/render/project/src/data/chroma_db`
- El build command debe ejecutar `python import_to_langchain_chroma.py`

### Error: "Out of memory"
- Plan Free tiene 512MB RAM
- Considera upgradearlo a Starter ($7/mes) con 512MB-1GB
- Modelo de embeddings ya es el más liviano (all-MiniLM-L6-v2)

### Frontend no conecta con Backend
1. Verifica que `REACT_APP_API_URL` esté configurada
2. Habilita CORS en el backend (ya configurado)
3. Usa HTTPS en producción

---

## 📊 Monitoreo

### Logs en Tiempo Real
```bash
# Ver logs del backend
render logs -s chat-fj-api

# Ver logs del frontend
render logs -s chat-fj-frontend
```

### Métricas
- CPU, RAM, y requests en el dashboard de Render
- Logs de errores en la pestaña "Logs"

---

## 💰 Costos Estimados

| Plan | Precio | Specs |
|------|--------|-------|
| **Free** | $0/mes | 512MB RAM, 750 horas/mes |
| **Starter** | $7/mes | 512MB-1GB RAM, ilimitado |
| **Standard** | $25/mes | 2GB RAM, auto-scaling |

**Recomendación**: Starter para producción (Backend + Frontend = $14/mes)

---

## 🔄 Actualizar Despliegue

```bash
# 1. Hacer cambios en el código
git add .
git commit -m "Actualización: descripción"
git push origin main

# 2. Render detectará los cambios y redesplegará automáticamente
```

---

## 🔒 Seguridad

### Secretos y API Keys
- ❌ NUNCA commitear `config/config.env` al repositorio
- ✅ Usar variables de entorno de Render
- ✅ `.gitignore` ya está configurado para excluir secretos

### HTTPS
- ✅ Render proporciona HTTPS automático
- ✅ Certificados SSL gratuitos

---

## 📞 Soporte

- **Documentación Render**: https://render.com/docs
- **Documentación Groq**: https://console.groq.com/docs
- **Issues del proyecto**: [Tu repo]/issues

---

## ✅ Checklist de Despliegue

- [ ] Repositorio pusheado a GitHub/GitLab
- [ ] API Key de Groq obtenida
- [ ] Variables de entorno configuradas en Render
- [ ] Backend desplegado y health check OK
- [ ] Frontend desplegado y conectado al backend
- [ ] ChromaDB inicializado con documentos
- [ ] Prueba de consulta exitosa
- [ ] Logs sin errores críticos

