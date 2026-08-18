.PHONY: ayuda install dev front test test-back test-front lint format build deploy-front admin backfill inventario clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff

ayuda:  ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Crea el entorno y instala dependencias (backend y frontend)
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt -r requirements-dev.txt
	cd frontend && npm install

dev:  ## Levanta el backend en http://127.0.0.1:8000
	$(VENV)/bin/uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000

front:  ## Levanta el frontend en http://localhost:3000
	cd frontend && npm start

test: test-back test-front  ## Corre todas las pruebas

test-back:  ## Pruebas del backend
	$(PY) -m unittest discover -s tests -t .

test-front:  ## Pruebas del frontend
	cd frontend && CI=true npx react-scripts test --watchAll=false

lint:  ## Revisa el código Python
	$(RUFF) check src tests scripts

format:  ## Ordena y formatea el código Python
	$(RUFF) check --fix src tests scripts
	$(RUFF) format src tests scripts

build:  ## Compila el frontend para producción
	@# .env.local tiene prioridad sobre .env.production en Create React App:
	@# si no se aparta, el sitio publicado apuntaría al backend local.
	@if [ -f frontend/.env.local ]; then mv frontend/.env.local frontend/.env.local.bak; fi
	cd frontend && npm run build || (if [ -f .env.local.bak ]; then mv .env.local.bak .env.local; fi; exit 1)
	@if [ -f frontend/.env.local.bak ]; then mv frontend/.env.local.bak frontend/.env.local; fi
	@echo "Verificando que el bundle no apunte al backend local..."
	@! grep -q "127.0.0.1:8" frontend/build/static/js/main.*.js || (echo "ERROR: el bundle quedó apuntando a localhost"; exit 1)
	@echo "OK: bundle listo en frontend/build"

deploy-front: build  ## Compila y publica el frontend en Firebase Hosting
	cd frontend && npx firebase deploy --only hosting --project chatfj-26458

admin:  ## Da acceso de administración. Uso: make admin EMAIL=persona@correo.cr
	$(PY) scripts/bootstrap_admin.py --email $(EMAIL)

backfill:  ## Reconcilia el catálogo de documentos con Pinecone
	$(PY) scripts/backfill_documents_registry.py

inventario:  ## Regenera docs/corpus.md con los documentos indexados
	$(PY) scripts/export_corpus_inventory.py

clean:  ## Borra archivos temporales
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .ruff_cache frontend/build
