#!/bin/bash
# Script para iniciar Ngrok con configuración personalizada

# Crear archivo de configuración temporal de Ngrok
cat > /tmp/ngrok.yml <<EOF
version: "2"
tunnels:
  frontend:
    proto: http
    addr: 3001
  backend:
    proto: http
    addr: 8000
EOF

# Iniciar Ngrok con ambos túneles
echo "Iniciando Ngrok con 2 túneles (frontend y backend)..."
ngrok start --all --config /tmp/ngrok.yml