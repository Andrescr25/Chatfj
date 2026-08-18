import asyncio
import logging
from typing import Any, Dict, List, Tuple

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

class WebSearchHelper:
    """Helper encapsulado para búsquedas web con filtrado de Costa Rica."""
    
    @staticmethod
    async def search_web_info(query: str, detected_location: str = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Búsqueda optimizada para instituciones en Costa Rica."""
        try:
            loop = asyncio.get_running_loop()
            
            # Construir queries
            search_queries = [f"{query} Costa Rica oficial"]
            if detected_location:
                search_queries.insert(0, f"{query} {detected_location} Costa Rica teléfono dirección")

            results = []

            for search_query in search_queries[:1]:  # Solo la primera búsqueda por ahora
                # La consulta se liga como argumento: si el bucle llegara a tener
                # más de una vuelta, la función interna usaría siempre la última.
                def _search(consulta=search_query):
                    try:
                        with DDGS() as ddgs:
                            return list(ddgs.text(consulta, max_results=3, region='cr-es'))
                    except Exception as e:
                        logger.warning(f"Error en búsqueda web: {e}")
                        return []

                search_results = await loop.run_in_executor(None, _search)
                if search_results:
                    results.extend(search_results)
                    break

            if not results:
                return "", []

            # Filtrar solo resultados de Costa Rica o dominios oficiales conocidos
            valid_domains = [
                '.cr', '.go.cr', 'poderjudicial.go.cr', 'pani.go.cr',
                'mtss.go.cr', 'oij.go.cr', 'defensapublica.cr',
                'ccss.sa.cr', 'tse.go.cr', 'asamblea.go.cr',
                'poder-judicial.go.cr', 'ministeriopublico.go.cr'
            ]

            filtered_results = []
            for result in results:
                href = result.get('href', '').lower()
                title = result.get('title', '').lower()
                body = result.get('body', '').lower()

                # Verificar que sea de Costa Rica (dominio O contenido)
                is_cr_domain = any(domain in href for domain in valid_domains)
                is_cr_content = 'costa rica' in title or 'costa rica' in body

                if is_cr_domain or is_cr_content:
                    filtered_results.append(result)

            if not filtered_results:
                return "", []

            # Formatear resultados
            web_info = []
            sources = []

            for result in filtered_results[:2]:  # Top 2 resultados válidos
                title = result.get('title', '')
                body = result.get('body', '')
                href = result.get('href', '')

                web_info.append(f"{title}: {body[:200]}")
                sources.append({
                    "title": title,
                    "url": href,
                    "snippet": body[:250]
                })

            return "\n".join(web_info), sources

        except Exception as e:
            logger.error(f"Error en WebSearchHelper: {e}")
            return "", []
