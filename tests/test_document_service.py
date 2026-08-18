"""
Tests de la gestión de documentos indexados (catálogo, subida y borrado).

No tocan Pinecone ni Firebase: usan dobles de prueba.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException

from src.app.core.rag.loaders import chunk_text
from src.app.core.registry import STATUS_DELETED, STATUS_INDEXED, DocumentRegistry
from src.app.core.security import ROLE_ADMIN, CurrentUser
from src.app.services.document_service import DocumentService, slugify


class VectorStoreFalso:
    """Doble de prueba del índice vectorial."""

    def __init__(self, ids_por_prefijo=None, soporta_listado=True):
        self.embedding_service = None
        self.ids_por_prefijo = ids_por_prefijo or {}
        self.soporta_listado = soporta_listado
        self.eliminados = []
        self.subidos = []

    def list_vector_ids(self, prefix, namespace=""):
        if not self.soporta_listado:
            return None
        return self.ids_por_prefijo.get(prefix, [])

    def delete_vectors(self, ids, namespace="", batch_size=500):
        self.eliminados.extend(ids)
        return len(ids)

    def upsert_vectors(self, vectors, namespace=""):
        self.subidos.extend(vectors)
        return True

    def fetch_vector_metadata(self, ids, namespace=""):
        return {vid: {"text": f"contenido de {vid}"} for vid in ids}

    def index_stats(self):
        return {"namespaces": {"": {"vector_count": 100}}}


class AlmacenamientoFalso:
    def __init__(self):
        self.bucket = None
        self.guardados = {}
        self.eliminados = []

    def save(self, doc_id, filename, content):
        self.guardados[doc_id] = content
        return {"storage_backend": "local", "storage_path": f"/tmp/{doc_id}"}

    def load(self, record):
        return self.guardados.get(record["doc_id"])

    def delete(self, record):
        self.eliminados.append(record["doc_id"])
        return True


class BaseDocumentos(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.vector_store = VectorStoreFalso()
        self.service = DocumentService(self.vector_store, embedding_service=None)
        self.service.registry = DocumentRegistry(json_path=self.tmp.name)
        self.service.storage = AlmacenamientoFalso()
        self.actor = CurrentUser(uid="u1", email="admin@ejemplo.cr", role=ROLE_ADMIN)
        self.patcher = patch("src.app.core.audit.log_action")
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        Path(self.tmp.name).unlink(missing_ok=True)


class TestTroceado(unittest.TestCase):
    def test_respeta_tamano_y_traslape(self):
        texto = "a" * 2500
        fragmentos = chunk_text(texto, chunk_size=1000, overlap=200)
        self.assertEqual(len(fragmentos[0]), 1000)
        # Cada fragmento avanza 800 caracteres (1000 - 200 de traslape)
        self.assertEqual(len(fragmentos), 4)

    def test_texto_vacio_no_produce_fragmentos(self):
        self.assertEqual(chunk_text(""), [])

    def test_traslape_invalido_falla(self):
        with self.assertRaises(ValueError):
            chunk_text("texto", chunk_size=100, overlap=100)


class TestIdentificadores(unittest.TestCase):
    def test_slug_limpia_acentos_y_espacios(self):
        self.assertEqual(slugify("Código Procesal de Familia 2024.pdf"), "codigo-procesal-de-familia-2024")

    def test_slug_nunca_queda_vacio(self):
        self.assertEqual(slugify("¿?.pdf"), "documento")


class TestSubida(BaseDocumentos):
    def test_rechaza_extension_no_admitida(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.create_document("virus.exe", b"contenido", self.actor)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rechaza_archivo_vacio(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.create_document("vacio.pdf", b"", self.actor)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rechaza_archivo_demasiado_grande(self):
        with patch("src.app.config.settings.MAX_UPLOAD_MB", 1):
            with self.assertRaises(HTTPException) as ctx:
                self.service.create_document("grande.pdf", b"x" * (2 * 1024 * 1024), self.actor)
        self.assertEqual(ctx.exception.status_code, 413)

    def test_detecta_documento_duplicado(self):
        self.service.create_document("ley.pdf", b"mismo contenido", self.actor)
        with self.assertRaises(HTTPException) as ctx:
            self.service.create_document("ley-copia.pdf", b"mismo contenido", self.actor)
        self.assertEqual(ctx.exception.status_code, 409)

    def test_registra_prefijo_propio_de_vectores(self):
        record = self.service.create_document("Ley RAC.pdf", b"contenido", self.actor)
        self.assertTrue(record["vector_prefix"].startswith("doc::ley-rac-"))
        self.assertTrue(record["vector_prefix"].endswith("::"))
        self.assertEqual(record["uploaded_by"], "admin@ejemplo.cr")


class TestEliminacion(BaseDocumentos):
    def _documento_indexado(self, chunks=3):
        record = self.service.create_document("ley.pdf", b"contenido", self.actor)
        self.service.registry.update(
            record["doc_id"], status=STATUS_INDEXED, chunks=chunks, chunks_total=chunks
        )
        prefix = record["vector_prefix"]
        self.vector_store.ids_por_prefijo[prefix] = [f"{prefix}{i}" for i in range(chunks)]
        return self.service.get_document(record["doc_id"])

    def test_elimina_solo_los_vectores_del_documento(self):
        doc = self._documento_indexado(chunks=3)
        otro = "doc::otro-documento::0"
        self.vector_store.ids_por_prefijo["doc::otro-documento::"] = [otro]

        resultado = self.service.delete_document(doc["doc_id"], self.actor)

        self.assertEqual(resultado["fragmentos_eliminados"], 3)
        self.assertNotIn(otro, self.vector_store.eliminados)
        self.assertEqual(len(self.vector_store.eliminados), 3)

    def test_marca_el_registro_como_eliminado_y_borra_el_original(self):
        doc = self._documento_indexado()
        self.service.delete_document(doc["doc_id"], self.actor)

        registro = self.service.registry.get(doc["doc_id"])
        self.assertEqual(registro["status"], STATUS_DELETED)
        self.assertEqual(registro["deleted_by"], "admin@ejemplo.cr")
        self.assertIn(doc["doc_id"], self.service.storage.eliminados)
        self.assertNotIn(doc["doc_id"], [d["doc_id"] for d in self.service.list_documents()])

    def test_no_permite_eliminar_dos_veces(self):
        doc = self._documento_indexado()
        self.service.delete_document(doc["doc_id"], self.actor)
        with self.assertRaises(HTTPException) as ctx:
            self.service.delete_document(doc["doc_id"], self.actor)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_reconstruye_los_ids_si_el_indice_no_permite_listarlos(self):
        """Respaldo para índices pod-based: los IDs son deterministas."""
        doc = self._documento_indexado(chunks=4)
        self.vector_store.soporta_listado = False

        resultado = self.service.delete_document(doc["doc_id"], self.actor)

        self.assertEqual(resultado["fragmentos_eliminados"], 4)
        self.assertEqual(
            self.vector_store.eliminados,
            [f"{doc['vector_prefix']}{i}" for i in range(4)],
        )

    def test_documento_inexistente_da_404(self):
        with self.assertRaises(HTTPException) as ctx:
            self.service.delete_document("no-existe", self.actor)
        self.assertEqual(ctx.exception.status_code, 404)


class TestVisorDeContenido(BaseDocumentos):
    def _documento_con_fragmentos(self, chunks=25):
        record = self.service.create_document("ley.pdf", b"contenido", self.actor)
        self.service.registry.update(
            record["doc_id"], status=STATUS_INDEXED, chunks=chunks, chunks_total=chunks
        )
        prefix = record["vector_prefix"]
        # Pinecone devuelve los IDs en orden arbitrario, no numérico:
        # se desordenan a propósito para comprobar que el servicio los ordena.
        desordenados = [10, 2, 24, 0, 7]
        indices = [i for i in desordenados if i < chunks]
        indices += [i for i in range(chunks) if i not in indices]
        self.vector_store.ids_por_prefijo[prefix] = [f"{prefix}{i}" for i in indices]
        return self.service.get_document(record["doc_id"])

    def test_devuelve_los_fragmentos_en_orden(self):
        doc = self._documento_con_fragmentos(chunks=25)
        contenido = self.service.get_document_content(doc["doc_id"], offset=0, limit=5)
        numeros = [f["numero"] for f in contenido["fragmentos"]]
        self.assertEqual(numeros, [1, 2, 3, 4, 5])
        self.assertTrue(contenido["fragmentos"][0]["id"].endswith("::0"))
        self.assertTrue(contenido["fragmentos"][4]["id"].endswith("::4"))

    def test_pagina_desde_el_desplazamiento_indicado(self):
        doc = self._documento_con_fragmentos(chunks=25)
        contenido = self.service.get_document_content(doc["doc_id"], offset=20, limit=10)
        self.assertEqual(contenido["total_fragmentos"], 25)
        self.assertEqual([f["numero"] for f in contenido["fragmentos"]], [21, 22, 23, 24, 25])

    def test_limita_el_tamano_de_pagina(self):
        """Un límite enorme no debe traer un documento entero de golpe."""
        doc = self._documento_con_fragmentos(chunks=25)
        contenido = self.service.get_document_content(doc["doc_id"], offset=0, limit=5000)
        self.assertEqual(contenido["limit"], 50)

    def test_documento_eliminado_no_se_puede_leer(self):
        doc = self._documento_con_fragmentos()
        self.service.delete_document(doc["doc_id"], self.actor)
        with self.assertRaises(HTTPException) as ctx:
            self.service.get_document_content(doc["doc_id"])
        self.assertEqual(ctx.exception.status_code, 400)


class TestReindexado(BaseDocumentos):
    def test_no_reindexa_documentos_sin_archivo_original(self):
        """Los documentos de la ingesta inicial no tienen original: reindexar los destruiría."""
        record = self.service.create_document("ley.pdf", b"contenido", self.actor)
        self.service.registry.update(record["doc_id"], storage_path="", legacy=True)

        with self.assertRaises(HTTPException) as ctx:
            self.service.reindex_document(record["doc_id"], self.actor)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.vector_store.eliminados, [])


if __name__ == "__main__":
    unittest.main()
