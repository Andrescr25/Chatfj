# ⚡ Comandos Rápidos para Despliegue

## 🚀 Desplegar en Render (3 pasos)

```bash
# Paso 1: Verificar que todo está listo
./pre_deploy_check.sh

# Paso 2: Commit y push
git add .
git commit -m "✨ Deploy: Sistema FJ con preguntas aclaratorias"
git push origin main

# Paso 3: Ir a Render
open https://dashboard.render.com
# → New → Blueprint → Conectar repo → Configurar GROQ_API_KEY → Apply
```

## 🧪 Testing Local Antes de Desplegar

```bash
# Iniciar servidor local
python3 src/api.py

# En otra terminal, probar endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"como denuncio acoso"}'

# Ejecutar tests completos
python3 test_sistema.py
```

## 📝 Ver Estado del Proyecto

```bash
# Ver archivos modificados
git status

# Ver cambios en archivos
git diff

# Ver últimos commits
git log --oneline -5
```

## 🔧 Mantenimiento Post-Despliegue

```bash
# Ver logs en tiempo real (requiere Render CLI)
render logs -s chat-fj-api --tail

# Reiniciar servicio
# → Desde Render Dashboard → Manual Deploy

# Actualizar después de cambios
git add .
git commit -m "🐛 Fix: descripción del cambio"
git push origin main
# Render redesplegará automáticamente
```

## 📊 Monitoreo Rápido

```bash
# Health check
curl https://chat-fj-api.onrender.com/health

# Probar pregunta
curl -X POST https://chat-fj-api.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Mi esposo me golpea, donde denuncio"}'
```

## 🐛 Troubleshooting

```bash
# Si ChromaDB está vacío, reimportar
python3 import_to_langchain_chroma.py

# Si hay errores de dependencias
pip install -r requirements.txt

# Limpiar cache y reiniciar
rm -f data/cache.db
pkill -9 -f "src/api.py"
python3 src/api.py
```

## 📦 Backup de ChromaDB

```bash
# Hacer backup antes de deploy
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma_db/

# Restaurar si es necesario
tar -xzf chroma_backup_YYYYMMDD.tar.gz
```

## 🔑 Configurar Variables de Entorno en Render

Dashboard → Service → Environment → Add Environment Variable:

```
GROQ_API_KEY = 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
GROQ_MODEL = llama-3.1-8b-instant
CHROMA_PERSIST_DIRECTORY = /opt/render/project/src/data/chroma_db
EMBEDDING_MODEL = sentence-transformers/all-MiniLM-L6-v2
```

## 💡 Tips Rápidos

- **Primera vez**: Tarda 10-15 min (descarga modelos + indexa docs)
- **Upgrades**: Render redespliega automáticamente con cada push
- **Logs**: Siempre revisa logs si algo falla
- **Disco**: 1GB suficiente para 5,000+ documentos
- **RAM**: Starter plan ($7/mes) recomendado para producción

