#!/bin/bash

echo "🔍 Verificación Pre-Despliegue a Render"
echo "========================================"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_total=0

check() {
    checks_total=$((checks_total + 1))
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        checks_passed=$((checks_passed + 1))
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

echo "📋 Verificando archivos necesarios..."
echo ""

# Check 1: render.yaml existe
[ -f "render.yaml" ]
check $? "render.yaml existe"

# Check 2: requirements.txt existe
[ -f "requirements.txt" ]
check $? "requirements.txt existe"

# Check 3: src/api.py existe
[ -f "src/api.py" ]
check $? "src/api.py existe"

# Check 4: frontend/package.json existe
[ -f "frontend/package.json" ]
check $? "frontend/package.json existe"

# Check 5: .gitignore existe
[ -f ".gitignore" ]
check $? ".gitignore existe"

# Check 6: import_to_langchain_chroma.py existe
[ -f "import_to_langchain_chroma.py" ]
check $? "import_to_langchain_chroma.py existe"

# Check 7: data/bloques_limpios.jsonl existe
[ -f "data/bloques_limpios.jsonl" ]
check $? "data/bloques_limpios.jsonl existe (5058 documentos)"

echo ""
echo "🔐 Verificando seguridad..."
echo ""

# Check 8: config.env no está en git
git ls-files | grep -q "config/config.env"
[ $? -eq 1 ]
check $? "config.env NO está en el repositorio (correcto)"

# Check 9: .gitignore contiene config.env
grep -q "config/config.env" .gitignore
check $? ".gitignore contiene config.env"

# Check 10: .gitignore contiene chroma_db
grep -q "chroma_db" .gitignore
check $? ".gitignore contiene chroma_db"

echo ""
echo "📦 Verificando dependencias..."
echo ""

# Check 11: duckduckgo-search en requirements
grep -q "duckduckgo-search" requirements.txt
check $? "duckduckgo-search en requirements.txt"

# Check 12: groq en requirements
grep -q "groq" requirements.txt
check $? "groq en requirements.txt"

# Check 13: chromadb en requirements
grep -q "chromadb" requirements.txt
check $? "chromadb en requirements.txt"

echo ""
echo "🎯 Verificando configuración de Render..."
echo ""

# Check 14: render.yaml tiene GROQ_API_KEY
grep -q "GROQ_API_KEY" render.yaml
check $? "render.yaml configura GROQ_API_KEY"

# Check 15: render.yaml tiene disco persistente
grep -q "disk:" render.yaml
check $? "render.yaml tiene disco persistente configurado"

# Check 16: render.yaml ejecuta import
grep -q "import_to_langchain_chroma.py" render.yaml
check $? "render.yaml ejecuta import de ChromaDB"

echo ""
echo "========================================"
echo -e "Resultado: ${GREEN}${checks_passed}${NC}/${checks_total} verificaciones pasadas"
echo ""

if [ $checks_passed -eq $checks_total ]; then
    echo -e "${GREEN}🎉 ¡Todo listo para desplegar!${NC}"
    echo ""
    echo "Siguiente paso:"
    echo "  git add ."
    echo "  git commit -m 'Preparar para despliegue en Render'"
    echo "  git push origin main"
    exit 0
else
    echo -e "${RED}⚠️  Hay problemas que resolver antes de desplegar${NC}"
    echo ""
    echo "Revisa los errores arriba y corrígelos."
    exit 1
fi
