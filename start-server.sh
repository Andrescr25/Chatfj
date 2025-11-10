#!/bin/bash
# Script para iniciar el servidor completo (backend + proxy)

echo "🚀 Iniciando Sistema de Facilitadores Judiciales..."

# Verificar que el directorio frontend/build existe
if [ ! -d "frontend/build" ]; then
    echo "❌ Error: No se encontró frontend/build"
    echo "   Por favor ejecuta: cd frontend && npm run build"
    exit 1
fi

# Verificar que las dependencias de Node están instaladas
if [ ! -d "node_modules" ]; then
    echo "📦 Instalando dependencias de Node.js..."
    npm install
fi

# Activar entorno virtual de Python (si existe)
if [ -d "venv" ]; then
    echo "🐍 Activando entorno virtual de Python..."
    source venv/bin/activate
fi

# Matar procesos anteriores si existen
echo "🧹 Limpiando procesos anteriores..."
pkill -f "uvicorn src.api:app" 2>/dev/null
pkill -f "proxy_server.js" 2>/dev/null
sleep 2

# Iniciar backend
echo "🔧 Iniciando backend (puerto 8000)..."
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Esperar a que el backend esté listo
echo "⏳ Esperando a que el backend inicie..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend iniciado correctamente"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Error: Backend no inició correctamente"
        echo "   Revisa los logs en logs/backend.log"
        exit 1
    fi
    sleep 1
done

# Iniciar proxy server
echo "🌐 Iniciando proxy server (puerto 4000)..."
node proxy_server.js > logs/proxy.log 2>&1 &
PROXY_PID=$!

# Esperar a que el proxy esté listo
echo "⏳ Esperando a que el proxy inicie..."
sleep 3

if curl -s http://localhost:4000/health > /dev/null 2>&1; then
    echo "✅ Proxy server iniciado correctamente"
else
    echo "❌ Error: Proxy server no inició correctamente"
    echo "   Revisa los logs en logs/proxy.log"
    exit 1
fi

echo ""
echo "🎉 ¡Sistema iniciado exitosamente!"
echo ""
echo "   📱 Frontend:  http://localhost:4000"
echo "   🔌 API:       http://localhost:8000"
echo "   📚 Docs:      http://localhost:8000/docs"
echo ""
echo "   PIDs: Backend=$BACKEND_PID, Proxy=$PROXY_PID"
echo ""
echo "Para detener los servidores:"
echo "   pkill -f uvicorn && pkill -f proxy_server"
echo ""
