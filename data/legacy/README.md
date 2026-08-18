# Artefactos de etapas anteriores

Ninguno de estos archivos lo usa el sistema actual. Se conservan por si hace
falta consultar el histórico, pero se pueden borrar sin consecuencias.

| Archivo | Qué era | Por qué ya no se usa |
|---|---|---|
| `cache.db` | Caché de respuestas en SQLite | El caché vive en Firestore (`src/app/core/cache.py`) |
| `training.db` | Correcciones de entrenamiento en SQLite | Las correcciones viven en Pinecone, namespace `corrections` |
| `bloques_limpios.jsonl` | Volcado intermedio de la primera ingesta | La ingesta va directo de los documentos a Pinecone |
| `verified_contacts.json` | Directorio de contactos verificados | Los contactos se indexan desde el Excel del corpus |
| `leyes_vigentes.json` | Índice manual de leyes | La normativa se consulta desde el índice vectorial |
