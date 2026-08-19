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
