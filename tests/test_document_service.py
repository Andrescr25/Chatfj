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

    def fetch_vectors_full(self, ids, namespace=""):
        return [(vid, [0.1, 0.2], {"text": f"contenido de {vid}", "source": "viejo.pdf",
                                   "filename": "viejo.pdf"}) for vid in ids]

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
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            self.tmp = tmp
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
        with patch("src.app.config.settings.MAX_UPLOAD_MB", 1), \
             self.assertRaises(HTTPException) as ctx:
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


class TestRenombrado(BaseDocumentos):
    """
    Renombrar cambia el nombre del archivo y el de las citas, sin tocar el
    contenido ni la relación entre catálogo e índice.
    """

    def _documento(self, chunks=3):
        doc = self.service.create_document("Prevencioun_de_la_violencia.pdf", b"contenido", self.actor)
        self.service.registry.update(doc["doc_id"], chunks=chunks, status=STATUS_INDEXED)
        prefix = doc["vector_prefix"]
        self.vector_store.ids_por_prefijo[prefix] = [f"{prefix}{i}" for i in range(chunks)]
        return self.service.get_document(doc["doc_id"])

    def test_cambia_el_nombre_del_archivo(self):
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(
            doc["doc_id"], self.actor, nombre="Prevención de la violencia política"
        )
        self.assertEqual(resultado["filename"], "Prevención de la violencia política.pdf")

    def test_conserva_la_extension_si_no_se_escribe(self):
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Nombre nuevo")
        self.assertTrue(resultado["filename"].endswith(".pdf"))

    def test_no_duplica_la_extension_si_se_escribe(self):
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Nombre nuevo.pdf")
        self.assertEqual(resultado["filename"], "Nombre nuevo.pdf")

    def test_no_cambia_el_identificador_ni_el_prefijo(self):
        """Cambiarlos rompería la relación entre el catálogo y el índice."""
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Otro")
        self.assertEqual(resultado["doc_id"], doc["doc_id"])
        self.assertEqual(resultado["vector_prefix"], doc["vector_prefix"])
        self.assertEqual(resultado["chunks"], doc["chunks"])

    def test_marca_las_citas_como_pendientes(self):
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Otro")
        self.assertTrue(resultado["citas_pendientes"])

    def test_propaga_el_nombre_a_los_fragmentos(self):
        doc = self._documento(chunks=3)
        self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Nombre definitivo")
        resultado = self.service.renombrar_en_indice(doc["doc_id"])

        self.assertEqual(resultado["actualizados"], 3)
        subidos = self.vector_store.subidos
        for _vid, _valores, metadata in subidos:
            self.assertEqual(metadata["source"], "Nombre definitivo.pdf")
            self.assertEqual(metadata["filename"], "Nombre definitivo.pdf")

    def test_no_altera_el_texto_de_los_fragmentos(self):
        doc = self._documento(chunks=2)
        self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Otro nombre")
        self.service.renombrar_en_indice(doc["doc_id"])
        for vid, _valores, metadata in self.vector_store.subidos:
            self.assertEqual(metadata["text"], f"contenido de {vid}")

    def test_al_terminar_deja_de_estar_pendiente(self):
        doc = self._documento()
        self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Otro")
        self.service.renombrar_en_indice(doc["doc_id"])
        self.assertFalse(self.service.get_document(doc["doc_id"])["citas_pendientes"])

    def test_cambia_la_materia(self):
        doc = self._documento()
        resultado = self.service.actualizar_metadatos(doc["doc_id"], self.actor, category="violencia_domestica")
        self.assertEqual(resultado["category"], "violencia_domestica")

    def test_rechaza_nombre_vacio(self):
        doc = self._documento()
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="   ")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rechaza_barras_en_el_nombre(self):
        """Firestore y las rutas de almacenamiento no las admiten."""
        doc = self._documento()
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="carpeta/archivo")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rechaza_un_nombre_desmedido(self):
        doc = self._documento()
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="x" * 250)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_no_se_puede_renombrar_uno_eliminado(self):
        doc = self._documento()
        self.service.delete_document(doc["doc_id"], self.actor)
        with self.assertRaises(HTTPException) as ctx:
            self.service.actualizar_metadatos(doc["doc_id"], self.actor, nombre="Nuevo")
        self.assertEqual(ctx.exception.status_code, 400)
