#!/bin/bash

echo "⚖️  Chat FJ - Facilitadores Judiciales"
echo "======================================"
echo ""

# Verificar si venv existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar venv
source venv/bin/activate

# Instalar dependencias Python si no están instaladas
if [ ! -f "venv/.installed" ]; then
    echo "📦 Instalando dependencias Python..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# Instalar dependencias Node si no están instaladas
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Instalando dependencias Node..."
    cd frontend
    npm install
    cd ..
fi

echo ""
echo "🚀 Iniciando servicios..."
echo ""

# Iniciar API en background
echo "📡 Iniciando API en http://localhost:8000"
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
API_PID=$!

# Esperar a que API inicie
sleep 5

# Iniciar Frontend
echo "🌐 Iniciando Frontend en http://localhost:3000"
echo ""
cd frontend
npm start

# Cleanup cuando se detenga
kill $API_PID 2>/dev/null

