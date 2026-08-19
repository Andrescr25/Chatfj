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
| Modelo de lenguaje | Cascada configurable: Groq, HuggingFace, Google Gemini |
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
| `LLM_CHAIN` | Cascada de modelos, en orden (ver abajo) |
| `GROQ_API_KEY` / `CEREBRAS_API_KEY` / `GEMINI_API_KEY` | Llaves de los proveedores |
| `PINECONE_API_KEY` | Base vectorial |
| `HUGGINGFACEHUB_API_TOKEN` | Embeddings |
| `ADMIN_EMAILS` | Correos con acceso garantizado al panel |
| `FIREBASE_STORAGE_BUCKET` | `chatfj-26458.firebasestorage.app` |

Y `config/firebase-adminsdk.json` con las credenciales de servicio de Firebase.

## Cascada de modelos

Un solo proveedor es un punto único de falla: cuando a Gemini se le agotaron los
créditos, el chat dejó de responder por completo aunque las otras llaves
funcionaban. Por eso los proveedores se declaran en orden y el primero que
responda gana:

```
LLM_CHAIN=groq,omniroute,gemini
```

Si un proveedor devuelve 429, se queda sin cupo, se satura o tarda demasiado, la
petición pasa al siguiente sin que la persona usuaria note nada. Los proveedores
sin llave configurada se omiten con un aviso en el arranque, no tumban el
sistema.

Proveedores admitidos: `groq`, `gemini`, `omniroute`, `openrouter`.

### Respaldo actual: HuggingFace

El token de HuggingFace que ya se usa para los embeddings (`HUGGINGFACEHUB_API_TOKEN`)
sirve también para generar respuestas: su router es compatible con OpenAI. No hace
falta registrarse en ningún servicio adicional.

```
LLM_CHAIN=groq,huggingface:zai-org/GLM-5.2,huggingface:meta-llama/Llama-3.3-70B-Instruct,gemini
```

Cada entrada puede llevar el modelo después de dos puntos. Como las cuotas
gratuitas se cuentan **por modelo**, encadenar dos modelos del mismo proveedor
agrega un respaldo sin abrir cuentas nuevas: si GLM-5.2 agota su cupo, entra
Llama 3.3 70B, que además consume menos crédito.

Sin modelo explícito se usa el de la configuración: `GROQ_MODEL`,
`HUGGINGFACE_CHAT_MODEL`, `GEMINI_MODEL`.

### Sobre las capas gratuitas

Se probaron varias alternativas antes de elegir esta:

| Proveedor | Estado (agosto 2026) |
|---|---|
| Groq | Capa gratuita recurrente. Es el proveedor principal |
| HuggingFace | Funciona con el token que ya se tenía. Respaldo actual |
| Google Gemini | 20 solicitudes por día en `gemini-2.5-flash`; se agota enseguida |
| Cerebras | Su capa gratuita son créditos que caducan a los 30 días |
| GitHub Models | En proceso de retiro |
| OpenRouter | Los modelos gratuitos exigen recarga previa |

`cerebras`, `openrouter` y `omniroute` quedan implementados por si sus condiciones
cambian: los cuatro comparten el mismo cliente compatible con OpenAI.

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
make inventario                           # regenerar docs/corpus.md
python scripts/diagnose_rag.py            # diagnóstico de la búsqueda vectorial
python scripts/maintenance/admin_corrections.py --help
```

## Documentación

- [Panel de administración](docs/panel-administracion.md): cómo se gestiona el
  acceso, cómo se suben y eliminan documentos, y cómo se entrena el asistente.
- [Inventario del corpus](docs/corpus.md): con qué documentos responde el
  asistente. Se regenera con `make inventario`.

Los archivos del corpus no se versionan (suman decenas de MB); el inventario es
el registro público de qué información se usó.

## Transparencia

El código es público, junto con la lista de documentos usados, las instrucciones
que guían al asistente, los umbrales de relevancia y los proveedores de IA. No se
publican las llaves de acceso ni las credenciales.
