"""
Latido: mantiene despierto el servicio en Render.

Render duerme un servicio gratuito tras 15 minutos sin tráfico entrante, y la
primera persona que pregunta después espera a que el contenedor arranque (40 s
medidos, más la respuesta).

Antes lo evitaba un flujo programado de GitHub Actions, y no funcionaba:
programado cada 10 minutos, entre el 26 de agosto y el 10 de setiembre de 2026
disparó en la práctica una vez cada dos horas (mediana). El servicio pasaba
dormido la mayor parte del día y todas las ejecuciones figuraban como exitosas,
porque cada una lo despertaba.

El latido corre dentro del propio servidor y consulta su dirección PÚBLICA: la
petición sale a internet y entra por el proxy de Render, que la cuenta como
tráfico entrante. Por localhost no serviría, porque Render nunca la vería.
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Costa Rica es UTC-6 todo el año, sin horario de verano.
ZONA_CR = timezone(timedelta(hours=-6))

SIEMPRE = (0, 24)

# Render duerme el servicio a los 15 minutos: el intervalo queda por debajo con
# margen para una consulta lenta.
INTERVALO_MAXIMO_SEGUNDOS = 14 * 60
INTERVALO_MINIMO_SEGUNDOS = 30

# Tras un fallo no se espera el intervalo completo: con latidos cada 10 minutos,
# otros 10 dejarían pasar 20 sin tráfico, y Render duerme a los 15.
REINTENTO_TRAS_FALLO_SEGUNDOS = 60

_FRANJA = re.compile(r"^(\d{1,2})\s*-\s*(\d{1,2})$")


def interpretar_horario(valor: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Traduce KEEP_ALIVE a la franja en que el servicio se mantiene despierto.

    "24h" o vacío -> siempre; "off" -> nunca (None); "7-22" -> de 7:00 a 22:00
    de Costa Rica. Una franja como "22-6" cruza la medianoche.
    """
    texto = (valor or "").strip().strip("\"'").strip().lower()
    if texto in ("", "24h"):
        return SIEMPRE
    if texto == "off":
        return None

    coincidencia = _FRANJA.match(texto)
    if coincidencia:
        inicio, fin = int(coincidencia.group(1)), int(coincidencia.group(2))
        if 0 <= inicio <= 23 and 0 <= fin <= 24 and inicio != fin:
            return (inicio, fin)

    # Un error de tipeo en el panel de Render no debe dejar dormir el servicio
    # sin que nadie lo note: se mantiene despierto y queda la advertencia.
    logger.warning(
        f"⚠️ KEEP_ALIVE={valor!r} no se entiende; se usa 24h. "
        "Valores válidos: 24h, off o una franja como 7-22."
    )
    return SIEMPRE


def toca_mantener_despierto(franja: Optional[Tuple[int, int]], ahora: datetime) -> bool:
    if franja is None:
        return False
    inicio, fin = franja
    hora = ahora.astimezone(ZONA_CR).hour
    if inicio < fin:
        return inicio <= hora < fin
    return hora >= inicio or hora < fin


def describir(franja: Tuple[int, int]) -> str:
    if franja == SIEMPRE:
        return "las 24 horas"
    return f"de {franja[0]}:00 a {franja[1]}:00 (hora de Costa Rica)"


def intervalo_seguro(segundos: int) -> int:
    return max(INTERVALO_MINIMO_SEGUNDOS, min(int(segundos), INTERVALO_MAXIMO_SEGUNDOS))


def _legible(segundos: int) -> str:
    if segundos >= 60 and segundos % 60 == 0:
        return f"{segundos // 60} min"
    return f"{segundos} s"


def consultar_salud(url_base: str) -> int:
    # /health y no /robots.txt: mientras el servicio duerme, Render contesta
    # /robots.txt por su cuenta sin despertarlo, así que un latido ahí parecería
    # funcionar sin evitar nada.
    return requests.get(f"{url_base.rstrip('/')}/health", timeout=30).status_code


def _ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


async def latir(
    url_base: str,
    franja: Tuple[int, int],
    intervalo: int,
    consultar: Callable[[str], int] = consultar_salud,
    ahora: Callable[[], datetime] = _ahora_utc,
    esperar: Callable[[float], Awaitable[None]] = asyncio.sleep,
    vueltas: Optional[int] = None,
) -> None:
    """
    Consulta el servicio a sí mismo mientras viva el proceso.

    Los cuatro últimos parámetros existen para las pruebas: simulan la red, el
    reloj y la espera sin aguardar diez minutos de verdad.
    """
    espera = intervalo
    estado = None  # "latiendo", "falla" o "pausa": solo se registran los cambios
    vuelta = 0
    while vueltas is None or vuelta < vueltas:
        vuelta += 1
        await esperar(espera)
        espera = intervalo

        if not toca_mantener_despierto(franja, ahora()):
            if estado != "pausa":
                logger.info(
                    f"💤 Latido en pausa fuera del horario ({describir(franja)}): "
                    "Render podrá dormir el servicio."
                )
            estado = "pausa"
            continue

        try:
            codigo = await asyncio.to_thread(consultar, url_base)
            problema = "" if codigo == 200 else f"HTTP {codigo}"
        except Exception as e:  # la red falla a veces; el latido no puede morir por eso
            problema = str(e)[:120] or type(e).__name__

        if problema:
            espera = min(REINTENTO_TRAS_FALLO_SEGUNDOS, intervalo)
            logger.warning(f"⚠️ Latido fallido ({problema}); se reintenta en {_legible(espera)}")
            estado = "falla"
        else:
            if estado != "latiendo":
                logger.info(f"💓 Latido: {url_base}/health respondió 200")
            estado = "latiendo"


def iniciar(url_base: Optional[str], horario: Optional[str], intervalo: int) -> Optional[asyncio.Task]:
    """
    Arranca el latido si corresponde y devuelve la tarea.

    Quien llama debe conservar la referencia: asyncio solo guarda referencias
    débiles a sus tareas, y una sin dueño puede recolectarse a mitad de camino,
    con lo que el latido se detendría sin aviso.
    """
    if not url_base:
        logger.info("💓 Latido inactivo: sin RENDER_EXTERNAL_URL, este proceso no corre en Render.")
        return None

    franja = interpretar_horario(horario)
    if franja is None:
        logger.warning(
            "💤 Latido apagado (KEEP_ALIVE=off): Render dormirá el servicio tras "
            "15 minutos sin tráfico."
        )
        return None

    url_base = url_base.rstrip("/")
    intervalo = intervalo_seguro(intervalo)
    logger.info(f"💓 Latido activo: {url_base}/health cada {_legible(intervalo)}, {describir(franja)}")
    return asyncio.create_task(latir(url_base, franja, intervalo), name="latido-render")
