const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = 4000;

// IMPORTANTE: El orden importa - primero los proxies, luego static, luego catch-all

// 1. Proxy para el API
// Solución: usar filter + router para que el path completo llegue al backend
const apiPaths = ['/ask', '/health', '/documents', '/stats', '/clear-cache', '/training'];

app.use(
  apiPaths,
  createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: (path, req) => {
      // Express elimina el prefijo, así que necesitamos reconstruir el path original
      return req.originalUrl || req.url;
    }
  })
);

// 2. Servir archivos estáticos del frontend
app.use(express.static(path.join(__dirname, 'frontend/build')));

// 3. Catch-all: todas las demás rutas van al frontend (Express 5 compatible)
app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Proxy server running on http://localhost:${PORT}`);
  console.log(`   Frontend: Serving from frontend/build`);
  console.log(`   API: Proxying to http://localhost:8000`);
});