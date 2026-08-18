# Chat FJ

Asistente de orientación legal para el **Sistema Nacional de Facilitadoras y
Facilitadores Judiciales** de Costa Rica.

Responde preguntas sobre normativa y trámites costarricenses apoyándose en un
corpus de documentos oficiales cargados por la oficina. No es un abogado: da
orientación, no asesoría legal.

- **Sitio público**: https://chatfj-26458.web.app
- **Panel de administración**: https://chatfj-26458.web.app/admin
- **API**: https://chatfj-m63k.onrender.com

## Cómo funciona

El modelo de lenguaje no "se sabe" las leyes: se le entrega el texto legal en el
momento de responder. A eso se le llama RAG (generación aumentada por
recuperación).

```
Pregunta
   ├─ correcciones aprendidas   (Pinecone, namespace "corrections")
   ├─ documentos oficiales      (Pinecone, namespace por defecto)
   └─ búsqueda web              (solo para contactos y ubicaciones)
                    ↓
        contexto + instrucciones de la oficina
                    ↓
              modelo de lenguaje
                    ↓
           respuesta con fuentes
```

Las correcciones que registran las personas entrenadoras tienen prioridad sobre
los documentos, y los documentos sobre el conocimiento general del modelo.

## Herramientas

| Capa | Tecnología |
|---|---|
| Interfaz | React 18, alojada en Firebase Hosting |
| Servidor | Python 3.11, FastAPI + Uvicorn, desplegado en Render |
| Base vectorial | Pinecone (índice `chatfj-legal-index`) |
| Embeddings | `intfloat/multilingual-e5-large` vía HuggingFace |
| Modelo de lenguaje | Groq (`openai/gpt-oss-120b`) o Google Gemini, intercambiables |
| Autenticación | Firebase Authentication (custom claims) |
| Catálogo y bitácora | Firestore |
| Archivos originales | Firebase Storage |

## Estructura

```
src/app/               Backend
  api/v1/              Rutas HTTP y dependencias compartidas (deps.py)
  core/                Seguridad, catálogo, almacenamiento, caché, auditoría
    llm/               Clientes de los modelos de lenguaje
    rag/               Embeddings, base vectorial, lectores, búsqueda web
    prompts/           Instrucciones del asistente
  services/            Lógica de negocio (chat, documentos, administradores)
  schemas/             Contratos de entrada y salida

frontend/src/          Interfaz
  api/                 Cliente HTTP único
  features/            chat/, admin/, training/
  components/          Piezas reutilizables
  hooks/  config/  styles/

scripts/               Tareas operativas (ver más abajo)
tests/                 Pruebas del backend
docs/                  Documentación de la oficina
data/legacy/           Artefactos de etapas anteriores, ya sin uso
```

## Levantarlo en local

```bash
make install                  # entorno de Python y dependencias de Node
make dev                      # backend en http://127.0.0.1:8000
make front                    # interfaz en http://localhost:3000
```

Hace falta un archivo `config/config.env` (no se versiona) con:

| Variable | Para qué |
|---|---|
| `LLM_PROVIDER` | `groq` o `gemini` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | Llave del proveedor elegido |
| `PINECONE_API_KEY` | Base vectorial |
| `HUGGINGFACEHUB_API_TOKEN` | Embeddings |
| `ADMIN_EMAILS` | Correos con acceso garantizado al panel |
| `FIREBASE_STORAGE_BUCKET` | `chatfj-26458.firebasestorage.app` |

Y `config/firebase-adminsdk.json` con las credenciales de servicio de Firebase.

## Pruebas y calidad

```bash
make test          # backend (unittest) y frontend (jest)
make lint          # ruff
make format        # ruff --fix y formateo
```

Se ejecutan solas en cada push (ver `.github/workflows/tests.yml`).

## Despliegue

El **backend** se despliega solo: Render observa la rama `main` de GitHub.

El **frontend** se publica a mano:

```bash
make deploy-front
```

> ⚠️ **No compile el frontend con `npm run build` directo.** En Create React App,
> `frontend/.env.local` tiene prioridad sobre `.env.production`, así que el sitio
> publicado quedaría apuntando a `localhost`. El objetivo `make build` aparta ese
> archivo y verifica el resultado antes de publicar.

## Tareas operativas

```bash
make admin EMAIL=persona@correo.cr        # dar acceso de administración
make backfill                             # reconciliar el catálogo con Pinecone
python scripts/ingest_documents.py data/docs --dry-run
python scripts/diagnose_rag.py            # diagnóstico de la búsqueda vectorial
python scripts/maintenance/admin_corrections.py --help
```

## Documentación

- [Panel de administración](docs/panel-administracion.md): cómo se gestiona el
  acceso, cómo se suben y eliminan documentos, y cómo se entrena el asistente.

## Transparencia

El código es público, junto con la lista de documentos usados, las instrucciones
que guían al asistente, los umbrales de relevancia y los proveedores de IA. No se
publican las llaves de acceso ni las credenciales.
