# 🔄 Cambiar a Nuevo Repositorio GitHub

## Repositorio Destino
https://github.com/chatfj2025-cpu/Chatfj.git

## 📋 Pasos Rápidos

### Opción 1: Cambiar el Remote (Recomendado)

```bash
# 1. Ver repositorio actual
git remote -v

# 2. Cambiar al nuevo repositorio
git remote set-url origin https://github.com/chatfj2025-cpu/Chatfj.git

# 3. Verificar que cambió
git remote -v

# 4. Push al nuevo repo
git push -u origin main
```

### Opción 2: Agregar Nuevo Remote

```bash
# 1. Agregar nuevo remote con nombre diferente
git remote add nuevo https://github.com/chatfj2025-cpu/Chatfj.git

# 2. Push al nuevo repo
git push -u nuevo main

# 3. (Opcional) Hacer que "nuevo" sea el principal
git remote rename origin viejo
git remote rename nuevo origin
```

### Opción 3: Empezar Desde Cero

```bash
# 1. Remover el remote actual
git remote remove origin

# 2. Agregar el nuevo
git remote add origin https://github.com/chatfj2025-cpu/Chatfj.git

# 3. Push con -u para configurar tracking
git push -u origin main
```

## 🔐 Autenticación

Si te pide credenciales, tienes 2 opciones:

### A. Personal Access Token (Recomendado)

1. Ve a GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Selecciona scopes: `repo` (completo)
4. Copia el token
5. Úsalo como contraseña cuando hagas `git push`

### B. SSH Key

```bash
# 1. Generar SSH key
ssh-keygen -t ed25519 -C "tu-email@ejemplo.com"

# 2. Agregar a ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 3. Copiar clave pública
cat ~/.ssh/id_ed25519.pub

# 4. Agregar en GitHub → Settings → SSH Keys

# 5. Cambiar URL a SSH
git remote set-url origin git@github.com:chatfj2025-cpu/Chatfj.git
```

## ✅ Verificación

```bash
# Ver que el remote cambió
git remote -v

# Debería mostrar:
# origin  https://github.com/chatfj2025-cpu/Chatfj.git (fetch)
# origin  https://github.com/chatfj2025-cpu/Chatfj.git (push)

# Verificar en GitHub que aparezcan los archivos
```

## 🚨 Problemas Comunes

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/chatfj2025-cpu/Chatfj.git
```

### Error: "Authentication failed"
- Usa Personal Access Token en lugar de contraseña
- O configura SSH keys

### Error: "Repository not found"
- Verifica que la URL sea correcta
- Verifica que tengas acceso al repo
- El repo debe existir en GitHub primero

## 📦 Antes de Push

```bash
# 1. Commit todos los cambios
git add .
git commit -m "✨ Deploy: Sistema FJ con preguntas aclaratorias"

# 2. Verificar estado
git status

# 3. Ver qué se va a pushear
git log --oneline -5
```

## 🎯 Comando Completo Final

```bash
# Todo en uno (copia y pega):
git remote set-url origin https://github.com/chatfj2025-cpu/Chatfj.git && \
git add . && \
git commit -m "✨ Deploy: Sistema FJ completo" && \
git push -u origin main
```

