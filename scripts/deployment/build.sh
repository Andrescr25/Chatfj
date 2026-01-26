#!/bin/bash

echo "🏗️  Building Chat FJ para producción..."
echo "========================================"
echo ""

# Build Frontend
echo "📦 Building React App..."
cd frontend
npm install
npm run build
cd ..

echo ""
echo "✅ Build completado!"
echo ""
echo "Archivos de producción en: frontend/build/"
echo ""
echo "Para servir el frontend:"
echo "  npx serve -s frontend/build -p 3000"
echo ""
echo "Para iniciar API:"
echo "  uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo ""

