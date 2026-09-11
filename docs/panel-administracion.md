# Panel de administración de Chat FJ

Permite a las personas administradoras controlar **quién tiene acceso** y **qué
información usa el asistente**, sin depender de la línea de comandos.

Se abre en la ruta `/admin` del sitio, iniciando sesión con Firebase, y luego con
el botón **Panel de administración**.

---

## 1. Acceso

Hay **un solo rol: administrador**. Quien lo tiene puede entrenar el asistente,
gestionar los documentos indexados y dar o quitar acceso a otras personas.

El rol se guarda como *custom claim* en Firebase Auth y viaja dentro del token de
sesión. Por eso, **cuando se le quita el acceso a alguien el sistema revoca sus
sesiones activas**, y quien reciba acceso debe iniciar sesión de nuevo.

Una cuenta de Firebase sin rol asignado **no** puede administrar ni entrenar:
antes bastaba con existir en el proyecto, ahora se requiere rol explícito.

### Respaldo ante bloqueos

La variable `ADMIN_EMAILS` (separada por comas) define correos que siempre tienen
acceso, aunque pierdan el claim. Esas cuentas aparecen como *protegidas* y no se
pueden deshabilitar ni revocar desde el panel: primero hay que sacarlas de esa
variable.

### Primer acceso

```bash
python scripts/bootstrap_admin.py --email persona@correo.cr   # crea o da acceso
python scripts/bootstrap_admin.py --list                      # ver quién tiene acceso
```

Si no se indica contraseña, el script imprime un enlace de Firebase para que la
persona defina la suya.

---

## 2. Administradores (pestaña «Administradores»)

- **Agregar**: correo y nombre. Por defecto se genera un enlace para que la
  persona cree su propia contraseña; opcionalmente se puede fijar una temporal
  (mínimo 8 caracteres).
- **Deshabilitar / habilitar**: bloquea el ingreso sin borrar la cuenta.
- **Revocar acceso**: quita el rol, deshabilita la cuenta y cierra sus sesiones.
  No borra el usuario de Firebase, para no perder el rastro de sus correcciones.

Reglas que el sistema impone:

- Nadie puede deshabilitarse ni eliminarse a sí mismo.
- Siempre debe quedar al menos una persona administradora activa.
- Las cuentas protegidas por `ADMIN_EMAILS` no se pueden revocar desde el panel.

---

## 3. Documentos indexados (pestaña «Documentos»)

### Subir

Se arrastra el archivo (PDF, DOCX, XLSX, TXT o MD; hasta 25 MB), se elige materia
y título, y el sistema:

1. Verifica que no esté duplicado (compara el hash SHA-256 del contenido).
2. Guarda el archivo original en Firebase Storage (o en `data/uploads/` si no hay
   bucket configurado).
3. Extrae el texto, lo divide en fragmentos de 1000 caracteres con 200 de
   traslape —los mismos parámetros de la ingesta original— y genera los
   embeddings por lotes de 20.
4. Sube los vectores a Pinecone con IDs `doc::{doc_id}::{n}`.

La indexación corre en segundo plano; la tabla muestra el avance
(`indexando 120/340`) y refresca sola cada 3 segundos.

### Ver el contenido

Al hacer clic en el nombre de un documento (o en el ícono del ojo) se abre el
visor con **el texto tal como quedó indexado**, fragmento por fragmento. Es lo
que el asistente lee al responder, que no siempre coincide con lo que se ve en
el archivo original: un PDF escaneado sin reconocimiento de texto deja
fragmentos vacíos o con errores de lectura, y eso se detecta aquí a simple
vista.

El visor carga de 20 en 20 fragmentos, permite buscar dentro de los que ya
cargó, y ofrece descargar el original o eliminar el documento sin salir de él.
Funciona también con los documentos de la ingesta inicial, porque lee del
índice y no del archivo.

### Eliminar

Se piden dos confirmaciones: el botón y escribir el nombre exacto del archivo.
El sistema borra **solo** los vectores cuyo ID empieza con el prefijo de ese
documento, elimina el archivo original y marca el registro como eliminado
(conserva el historial de quién lo borró y cuándo).

Las correcciones aprendidas viven en otro espacio del índice (`corrections`) y
**no** se borran al eliminar un documento.

### Reindexar

Vuelve a procesar el archivo original. No está disponible para los documentos
heredados de la ingesta inicial, porque de esos no se guardó el original: para
actualizarlos hay que eliminarlos y volver a subirlos.

---

## 4. El catálogo

Pinecone almacena vectores, no archivos: no se le puede preguntar «¿qué
documentos tengo?». Por eso el sistema mantiene un catálogo en Firestore
(colección `documents`) con nombre, materia, cantidad de fragmentos, prefijo de
IDs, quién lo subió y estado. Sin ese catálogo no habría forma confiable de
listar ni de eliminar.

Los documentos indexados antes del panel se registran con:

```bash
python scripts/backfill_documents_registry.py --dry-run   # simulación
python scripts/backfill_documents_registry.py             # aplica
```

El script recorre los IDs del índice, los agrupa por prefijo y crea un registro
por documento. Es seguro ejecutarlo varias veces: omite los que ya existen.

---

## 5. Bitácora

Toda alta o baja de administradores y toda subida o eliminación de documentos se
registra en la colección `audit_log` de Firestore (o en `logs/audit_log.jsonl` si
Firestore no está disponible), con acción, responsable, objetivo y fecha.

---

## 6. Variables de entorno

| Variable | Para qué |
|---|---|
| `ADMIN_EMAILS` | Administradores garantizados, separados por coma |
| `FIREBASE_STORAGE_BUCKET` | Bucket para los archivos originales (opcional) |
| `EXTRA_CORS_ORIGINS` | Dominios adicionales permitidos, separados por coma |
| `MAX_UPLOAD_MB` | Tamaño máximo de subida (25 por defecto) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Parámetros de troceado (1000 / 200) |

> El disco de Render es efímero: sin `FIREBASE_STORAGE_BUCKET`, los archivos
> originales se pierden en cada despliegue y los documentos no se podrán
> reindexar (los vectores sí permanecen en Pinecone).

---

## 7. Endpoints

Todos requieren rol de administrador.

| Método | Ruta |
|---|---|
| GET | `/api/v1/admins/me` |
| GET | `/api/v1/admins` |
| POST | `/api/v1/admins` |
| PATCH | `/api/v1/admins/{uid}` |
| DELETE | `/api/v1/admins/{uid}` |
| GET | `/api/v1/documents` |
| GET | `/api/v1/documents/{id}` |
| GET | `/api/v1/documents/{id}/content` |
| POST | `/api/v1/documents` |
| POST | `/api/v1/documents/{id}/reindex` |
| DELETE | `/api/v1/documents/{id}` |
| GET | `/api/v1/documents/{id}/download` |

---

## 8. Disponibilidad del servicio (mantenerlo despierto)

Render duerme un servicio gratuito tras **15 minutos sin tráfico entrante**, y
quien pregunte después espera el arranque: unos 40 segundos medidos, más la
respuesta.

### Cómo se evita

1. **Latido interno** ([src/app/core/keep_alive.py](../src/app/core/keep_alive.py)).
   El propio servidor consulta su dirección pública (`RENDER_EXTERNAL_URL`, que
   Render define sola) cada 10 minutos. La petición entra por el proxy de Render
   y cuenta como tráfico. Tras un fallo reintenta al minuto, porque esperar otros
   10 dejaría pasar los 15 de Render.
2. **Respaldo en GitHub Actions** ([keep_alive.yml](../.github/workflows/keep_alive.yml)).
   Cubre el único caso que el latido no puede: si Render llega a dormir el
   servicio, el latido se detiene con él. Si encuentra el servicio recién
   arrancado tras una espera larga, el flujo **falla** y GitHub avisa por correo:
   significa que el latido no lo sostuvo.
3. **Despertar al abrir la página.** El frontend consulta `/api/v1/health` apenas
   carga: si el servicio dormía, el arranque transcurre mientras la persona
   escribe su pregunta.

### Por qué no basta con GitHub Actions

Era el mecanismo anterior. Programado cada 10 minutos, entre el 26 de agosto y el
10 de setiembre de 2026 disparó en la práctica **una vez cada dos horas**
(mediana), con 69 huecos de más de 15 minutos en horario diurno. Todas las
ejecuciones figuraban como exitosas, porque cada una despertaba al servicio.

### Configuración (variables de entorno en Render)

| Variable | Valor | Efecto |
|---|---|---|
| `KEEP_ALIVE` | `24h` (predeterminado) | Siempre despierto |
| | `7-22` | Solo de 7:00 a 22:00 de Costa Rica. De noche duerme, y la primera persona de la mañana espera el arranque |
| | `off` | Sin latido |
| `KEEP_ALIVE_INTERVAL_SECONDS` | `600` | Cada cuánto late. Nunca pasa de 14 minutos |

Un valor de `KEEP_ALIVE` que no se entienda se trata como `24h` y deja una
advertencia en los registros.

### Cuidado con las horas gratuitas

Render da **750 horas gratis al mes por espacio de trabajo**, compartidas entre
**todos** sus servicios web gratuitos, y al agotarlas **suspende todos** hasta el
mes siguiente. Mantener este servicio despierto las 24 horas consume 720 horas en
un mes de 30 días y 744 en uno de 31: casi todo el cupo.

Si el mismo espacio de trabajo tiene otro servicio gratuito que también se
mantiene despierto, no alcanza para los dos. En ese caso use `KEEP_ALIVE=7-22`.

### Comprobar que funciona

`/health` informa `activo_desde`. Si ese valor no cambia entre dos consultas
separadas por más de 15 minutos, el servicio no durmió en ese lapso. En los
registros de Render, el latido deja `💓 Latido activo` al arrancar y solo vuelve
a escribir si falla o se recupera.

Detalles a tener presentes:

- GitHub desactiva los flujos programados tras 60 días sin actividad en el
  repositorio; si eso pasa, se reactivan desde la pestaña Actions. El latido
  interno no depende de eso.
- La dirección que consulta el respaldo se puede cambiar sin tocar el código, con
  la variable de repositorio `BACKEND_URL` (Settings → Secrets and variables →
  Actions → Variables).
