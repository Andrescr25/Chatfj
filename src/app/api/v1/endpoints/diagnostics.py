"""Diagnóstico de la cascada de modelos (solo administración)."""
import logging
import time

from fastapi import APIRouter, Depends

from src.app.config import settings
from src.app.core.llm.factory import _crear
from src.app.core.security import CurrentUser, require_admin

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health/llm")
async def diagnostico_modelos(user: CurrentUser = Depends(require_admin)):
    """
    Prueba uno por uno los proveedores de la cascada y dice cuál responde.

    Existe porque cuando el chat falla, desde afuera todos los errores se ven
    iguales: "error técnico momentáneo". Sin esto hay que adivinar si se acabó
    el cupo, si falta una llave o si el proveedor está caído.
    """
    resultados = []

    for declaracion in settings.llm_chain:
        entrada = {"proveedor": declaracion}
        try:
            cliente = _crear(declaracion)
        except Exception as e:
            entrada.update(estado="sin configurar", detalle=str(e)[:200])
            resultados.append(entrada)
            continue

        entrada["modelo"] = getattr(cliente, "nombre", declaracion)
        inicio = time.time()
        try:
            respuesta = await cliente.generate_async("Responda únicamente con la palabra: ok")
            entrada.update(
                estado="funciona",
                segundos=round(time.time() - inicio, 2),
                respuesta=respuesta[:60],
            )
        except Exception as e:
            entrada.update(
                estado="falla",
                segundos=round(time.time() - inicio, 2),
                detalle=str(e)[:300],
            )
        resultados.append(entrada)

    disponibles = [r["proveedor"] for r in resultados if r["estado"] == "funciona"]
    return {
        "cascada_configurada": settings.llm_chain,
        "proveedores": resultados,
        "hay_respaldo": len(disponibles) > 1,
        "resumen": (
            f"{len(disponibles)} de {len(resultados)} proveedores responden"
            if disponibles else "NINGÚN proveedor responde: el chat está caído"
        ),
    }


@router.get("/health/embeddings")
async def diagnostico_embeddings(user: CurrentUser = Depends(require_admin)):
    """
    Prueba una por una las llaves de embeddings y dice cuál tiene crédito.

    Los embeddings son la falla más peligrosa del sistema porque es silenciosa:
    cuando se agotan, el chat sigue contestando, pero sin citar ni un documento.
    Desde afuera parece que funciona. Esto lo saca a la luz sin tener que leer
    los registros del servidor.

    Ojo: cada consulta a este diagnóstico gasta una llamada por llave, así que
    conviene usarlo cuando haga falta, no dejarlo actualizándose solo.
    """
    from src.app.api.v1.deps import get_chat_service

    servicio = get_chat_service().embedding_service
    resultados = []

    for proveedor in servicio.proveedores_e5:
        nombre = getattr(proveedor, "nombre", "principal")
        entrada = {"proveedor": nombre}
        apartado = getattr(proveedor, "penalizado_hasta", 0.0) - time.monotonic()
        if apartado > 0:
            entrada["apartado_minutos"] = round(apartado / 60, 1)

        inicio = time.time()
        try:
            vector = proveedor.embed_query("ok")
            entrada.update(
                estado="funciona",
                segundos=round(time.time() - inicio, 2),
                dimensiones=len(vector),
            )
        except Exception as e:
            estado_http = getattr(e, "status_code", None)
            entrada.update(
                estado="sin crédito" if estado_http == 402 else "falla",
                segundos=round(time.time() - inicio, 2),
                detalle=str(e)[:300],
            )
        resultados.append(entrada)

    con_credito = [r["proveedor"] for r in resultados if r["estado"] == "funciona"]
    return {
        "modelo": settings.EMBEDDING_MODEL_NAME,
        "proveedores": resultados,
        "hay_respaldo": len(con_credito) > 1,
        "resumen": (
            f"{len(con_credito)} de {len(resultados)} llaves con crédito"
            if con_credito
            else "NINGUNA llave tiene crédito: el chat responde sin documentos"
        ),
    }
