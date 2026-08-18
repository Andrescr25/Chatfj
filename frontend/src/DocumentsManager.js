import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Upload, Trash2, RefreshCw, Download, FileText, Loader2,
  AlertTriangle, CheckCircle2, Clock, Database, Eye, X, Search
} from 'lucide-react';
import apiService from './services/api';
import { LEGAL_CATEGORIES } from './config/constants';

const CATEGORIES = [{ value: 'general', label: 'General' }, ...LEGAL_CATEGORIES];

// Confirmación deliberada sin obligar a copiar nombres de archivo larguísimos
const PALABRA_CONFIRMACION = 'ELIMINAR';

const ESTADOS = {
  pendiente:  { label: 'En cola',    icon: Clock,        clase: 'ap-badge-espera' },
  indexando:  { label: 'Indexando',  icon: Loader2,      clase: 'ap-badge-proceso' },
  indexado:   { label: 'Indexado',   icon: CheckCircle2, clase: 'ap-badge-ok' },
  error:      { label: 'Error',      icon: AlertTriangle, clase: 'ap-badge-error' },
  eliminado:  { label: 'Eliminado',  icon: Trash2,       clase: 'ap-badge-error' },
};

function formatearTamano(bytes) {
  if (!bytes) return '—';
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatearAutor(autor) {
  if (!autor) return '—';
  if (autor.toLowerCase().startsWith('ingesta inicial')) return 'Ingesta inicial';
  return autor;
}

function formatearFecha(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('es-CR', {
      day: '2-digit', month: '2-digit', year: 'numeric'
    });
  } catch (e) {
    return '—';
  }
}

function DocumentsManager() {
  const [documentos, setDocumentos] = useState([]);
  const [stats, setStats] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');
  const [aviso, setAviso] = useState('');

  const [archivo, setArchivo] = useState(null);
  const [categoria, setCategoria] = useState('general');
  const [titulo, setTitulo] = useState('');
  const [subiendo, setSubiendo] = useState(false);
  const [arrastrando, setArrastrando] = useState(false);

  const [porEliminar, setPorEliminar] = useState(null);
  const [confirmacion, setConfirmacion] = useState('');
  const [eliminando, setEliminando] = useState(false);

  const [viendo, setViendo] = useState(null);
  const [contenido, setContenido] = useState(null);
  const [cargandoContenido, setCargandoContenido] = useState(false);
  const [errorContenido, setErrorContenido] = useState('');
  const [filtro, setFiltro] = useState('');

  const inputRef = useRef(null);

  const TAMANO_PAGINA = 20;

  const confirmado = confirmacion.trim().toUpperCase() === PALABRA_CONFIRMACION;

  const cargar = useCallback(async (silencioso = false) => {
    if (!silencioso) setCargando(true);
    try {
      const data = await apiService.listDocuments();
      setDocumentos(data.documents || []);
      setStats(data.stats || null);
      setError('');
    } catch (e) {
      setError(e.message || 'No se pudieron cargar los documentos.');
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => { cargar(); }, [cargar]);

  // Mientras haya documentos procesándose, refrescar para mostrar el avance
  useEffect(() => {
    const enProceso = documentos.some(
      d => d.status === 'pendiente' || d.status === 'indexando'
    );
    if (!enProceso) return undefined;
    const id = setInterval(() => cargar(true), 3000);
    return () => clearInterval(id);
  }, [documentos, cargar]);

  const seleccionarArchivo = (file) => {
    if (!file) return;
    setArchivo(file);
    setTitulo(file.name.replace(/\.[^.]+$/, ''));
    setError('');
    setAviso('');
  };

  const manejarSoltar = (e) => {
    e.preventDefault();
    setArrastrando(false);
    seleccionarArchivo(e.dataTransfer.files?.[0]);
  };

  const subir = async () => {
    if (!archivo || subiendo) return;
    setSubiendo(true);
    setError('');
    setAviso('');
    try {
      const record = await apiService.uploadDocument(archivo, categoria, titulo);
      setAviso(
        `"${record.filename}" se subió correctamente. La indexación continúa en segundo plano.`
      );
      setArchivo(null);
      setTitulo('');
      if (inputRef.current) inputRef.current.value = '';
      cargar(true);
    } catch (e) {
      setError(e.message || 'No se pudo subir el documento.');
    } finally {
      setSubiendo(false);
    }
  };

  const reindexar = async (doc) => {
    setError('');
    setAviso('');
    try {
      await apiService.reindexDocument(doc.doc_id);
      setAviso(`Reindexando "${doc.filename}".`);
      cargar(true);
    } catch (e) {
      setError(e.message || 'No se pudo reindexar.');
    }
  };

  const descargar = async (doc) => {
    setError('');
    try {
      await apiService.downloadDocument(doc.doc_id, doc.filename);
    } catch (e) {
      setError(e.message || 'No se pudo descargar el archivo.');
    }
  };

  const eliminar = async () => {
    if (!porEliminar || eliminando) return;
    setEliminando(true);
    setError('');
    try {
      const res = await apiService.deleteDocument(porEliminar.doc_id);
      setAviso(
        `"${res.filename}" se eliminó del índice (${res.fragmentos_eliminados} fragmentos).`
      );
      setPorEliminar(null);
      setConfirmacion('');
      cargar(true);
    } catch (e) {
      setError(e.message || 'No se pudo eliminar el documento.');
    } finally {
      setEliminando(false);
    }
  };

  const abrirVisor = async (doc) => {
    setViendo(doc);
    setContenido(null);
    setFiltro('');
    setErrorContenido('');
    setCargandoContenido(true);
    try {
      const datos = await apiService.getDocumentContent(doc.doc_id, 0, TAMANO_PAGINA);
      setContenido(datos);
    } catch (e) {
      setErrorContenido(e.message || 'No se pudo leer el contenido del documento.');
    } finally {
      setCargandoContenido(false);
    }
  };

  const cargarMas = async () => {
    if (!viendo || !contenido || cargandoContenido) return;
    setCargandoContenido(true);
    try {
      const siguiente = await apiService.getDocumentContent(
        viendo.doc_id, contenido.fragmentos.length, TAMANO_PAGINA
      );
      setContenido({
        ...siguiente,
        fragmentos: [...contenido.fragmentos, ...siguiente.fragmentos],
      });
    } catch (e) {
      setErrorContenido(e.message || 'No se pudieron cargar más fragmentos.');
    } finally {
      setCargandoContenido(false);
    }
  };

  const cerrarVisor = () => {
    setViendo(null);
    setContenido(null);
    setFiltro('');
    setErrorContenido('');
  };

  // Búsqueda tolerante a acentos y mayúsculas sobre los fragmentos ya cargados
  const normalizar = (t) => (t || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');

  const fragmentosVisibles = (contenido?.fragmentos || []).filter(
    (f) => !filtro.trim() || normalizar(f.texto).includes(normalizar(filtro))
  );

  const renderEstado = (doc) => {
    const estado = ESTADOS[doc.status] || ESTADOS.pendiente;
    const Icono = estado.icon;
    const enProceso = doc.status === 'indexando';
    return (
      <span className={`ap-badge ${estado.clase}`}>
        <Icono size={13} className={enProceso ? 'ap-girando' : ''} />
        {estado.label}
        {enProceso && doc.chunks_total
          ? ` ${doc.chunks}/${doc.chunks_total}`
          : ''}
      </span>
    );
  };

  return (
    <div className="ap-seccion">
      {stats && (
        <div className="ap-stats">
          <div className="ap-stat">
            <span className="ap-stat-valor">{stats.documentos}</span>
            <span className="ap-stat-label">Documentos</span>
          </div>
          <div className="ap-stat">
            <span className="ap-stat-valor">{stats.fragmentos_catalogados?.toLocaleString('es-CR')}</span>
            <span className="ap-stat-label">Fragmentos</span>
          </div>
          <div className="ap-stat">
            <span className="ap-stat-valor">{stats.vectores_en_indice?.toLocaleString('es-CR')}</span>
            <span className="ap-stat-label">Vectores en el índice</span>
          </div>
          <div className="ap-stat">
            <span className="ap-stat-valor">{stats.correcciones_aprendidas?.toLocaleString('es-CR')}</span>
            <span className="ap-stat-label">Correcciones aprendidas</span>
          </div>
        </div>
      )}

      <div
        className={`ap-dropzone ${arrastrando ? 'ap-dropzone-activa' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setArrastrando(true); }}
        onDragLeave={() => setArrastrando(false)}
        onDrop={manejarSoltar}
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={22} />
        <div>
          <strong>{archivo ? archivo.name : 'Arrastre un documento o haga clic para elegirlo'}</strong>
          <p>PDF, Word (.docx), Excel (.xlsx) o texto. Máximo 25 MB.</p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.xlsx"
          onChange={(e) => seleccionarArchivo(e.target.files?.[0])}
          hidden
        />
      </div>

      {archivo && (
        <div className="ap-form-subida">
          <label>
            Título
            <input
              type="text"
              value={titulo}
              onChange={(e) => setTitulo(e.target.value)}
              placeholder="Nombre con el que se identificará el documento"
            />
          </label>
          <label>
            Materia
            <select value={categoria} onChange={(e) => setCategoria(e.target.value)}>
              {CATEGORIES.map(c => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </label>
          <div className="ap-form-acciones">
            <button className="ap-btn ap-btn-primario" onClick={subir} disabled={subiendo}>
              {subiendo ? <Loader2 size={15} className="ap-girando" /> : <Upload size={15} />}
              {subiendo ? 'Subiendo...' : 'Subir e indexar'}
            </button>
            <button
              className="ap-btn"
              onClick={() => { setArchivo(null); if (inputRef.current) inputRef.current.value = ''; }}
              disabled={subiendo}
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      {error && <div className="ap-alerta ap-alerta-error"><AlertTriangle size={15} />{error}</div>}
      {aviso && <div className="ap-alerta ap-alerta-ok"><CheckCircle2 size={15} />{aviso}</div>}

      {cargando ? (
        <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Cargando documentos...</div>
      ) : (
        <div className="ap-tabla-contenedor">
          <table className="ap-tabla ap-tabla-documentos">
            <colgroup>
              <col className="ap-col-doc" />
              <col className="ap-col-materia" />
              <col className="ap-col-num" />
              <col className="ap-col-estado" />
              <col className="ap-col-autor" />
              <col className="ap-col-fecha" />
              <col className="ap-col-acciones" />
            </colgroup>
            <thead>
              <tr>
                <th>Documento</th>
                <th className="ap-col-oculta-md">Materia</th>
                <th className="ap-num">Fragmentos</th>
                <th>Estado</th>
                <th className="ap-col-oculta-md">Subido por</th>
                <th className="ap-col-oculta-sm">Fecha</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {documentos.length === 0 && (
                <tr><td colSpan={7} className="ap-vacio">
                  <Database size={18} /> No hay documentos registrados.
                </td></tr>
              )}
              {documentos.map(doc => (
                <tr key={doc.doc_id}>
                  <td>
                    <div className="ap-doc-nombre">
                      <FileText size={15} />
                      <div className="ap-doc-textos">
                        <button
                          type="button"
                          className="ap-doc-enlace"
                          onClick={() => abrirVisor(doc)}
                          title={`Ver el contenido de ${doc.filename}`}
                        >
                          {doc.title || doc.filename}
                        </button>
                        <span title={doc.filename}>
                          {doc.filename} · {formatearTamano(doc.size_bytes)}
                        </span>
                        {doc.error && (
                          <span className="ap-error-texto" title={doc.error}>{doc.error}</span>
                        )}
                        {doc.storage_warning && (
                          <span className="ap-aviso-texto" title={doc.storage_warning}>
                            Original no respaldado
                          </span>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="ap-col-oculta-md">
                    {(CATEGORIES.find(c => c.value === doc.category) || {}).label || doc.category}
                  </td>
                  <td className="ap-num">{(doc.chunks || 0).toLocaleString('es-CR')}</td>
                  <td>{renderEstado(doc)}</td>
                  <td className="ap-celda-secundaria ap-col-oculta-md">
                    <span title={doc.uploaded_by}>{formatearAutor(doc.uploaded_by)}</span>
                    {doc.legacy && <span className="ap-etiqueta-legacy">heredado</span>}
                  </td>
                  <td className="ap-celda-secundaria ap-col-oculta-sm">{formatearFecha(doc.uploaded_at)}</td>
                  <td className="ap-celda-acciones">
                    <div className="ap-acciones">
                      <button title="Ver contenido" onClick={() => abrirVisor(doc)}>
                        <Eye size={15} />
                      </button>
                      {doc.storage_path && (
                        <>
                          <button title="Descargar original" onClick={() => descargar(doc)}>
                            <Download size={15} />
                          </button>
                          <button title="Reindexar" onClick={() => reindexar(doc)}>
                            <RefreshCw size={15} />
                          </button>
                        </>
                      )}
                      <button
                        title="Eliminar del índice"
                        className="ap-accion-peligro"
                        onClick={() => { setPorEliminar(doc); setConfirmacion(''); }}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {viendo && (
        <div className="ap-modal-fondo" onClick={cerrarVisor}>
          <div className="ap-visor" onClick={(e) => e.stopPropagation()}>
            <header className="ap-visor-header">
              <div className="ap-visor-titulo">
                <h3>{viendo.title || viendo.filename}</h3>
                <span>
                  {viendo.filename}
                  {contenido ? ` · ${contenido.total_fragmentos.toLocaleString('es-CR')} fragmentos` : ''}
                </span>
              </div>
              <button className="ap-cerrar" onClick={cerrarVisor} title="Cerrar">
                <X size={20} />
              </button>
            </header>

            <p className="ap-visor-nota">
              Este es el texto tal como quedó indexado, que es lo que el asistente lee al
              responder. Puede no coincidir con lo que se ve en el archivo original: un PDF
              escaneado, por ejemplo, deja fragmentos vacíos o con errores de lectura.
            </p>

            <div className="ap-visor-busqueda">
              <Search size={15} />
              <input
                type="text"
                value={filtro}
                onChange={(e) => setFiltro(e.target.value)}
                placeholder="Buscar dentro de los fragmentos cargados"
              />
              {filtro && (
                <button className="ap-btn" onClick={() => setFiltro('')}>Limpiar</button>
              )}
            </div>

            <div className="ap-visor-cuerpo">
              {errorContenido && (
                <div className="ap-alerta ap-alerta-error">
                  <AlertTriangle size={15} />{errorContenido}
                </div>
              )}

              {!contenido && cargandoContenido && (
                <div className="ap-cargando">
                  <Loader2 size={20} className="ap-girando" /> Leyendo el documento...
                </div>
              )}

              {contenido && contenido.total_fragmentos === 0 && (
                <div className="ap-vacio">
                  Este documento todavía no tiene fragmentos indexados.
                </div>
              )}

              {contenido && fragmentosVisibles.length === 0 && contenido.total_fragmentos > 0 && (
                <div className="ap-vacio">
                  Ningún fragmento cargado contiene «{filtro}».
                </div>
              )}

              {fragmentosVisibles.map((f) => (
                <article key={f.id} className="ap-fragmento">
                  <span className="ap-fragmento-numero">Fragmento {f.numero}</span>
                  <p>{f.texto || <em>(sin texto: el lector no pudo extraer contenido aquí)</em>}</p>
                </article>
              ))}
            </div>

            <footer className="ap-visor-footer">
              <span className="ap-celda-secundaria">
                {contenido
                  ? `Mostrando ${contenido.fragmentos.length} de ${contenido.total_fragmentos.toLocaleString('es-CR')}`
                  : ''}
                {filtro && contenido ? ` · ${fragmentosVisibles.length} coinciden` : ''}
              </span>
              <div className="ap-visor-acciones">
                {contenido && contenido.fragmentos.length < contenido.total_fragmentos && (
                  <button className="ap-btn" onClick={cargarMas} disabled={cargandoContenido}>
                    {cargandoContenido
                      ? <Loader2 size={15} className="ap-girando" />
                      : <FileText size={15} />}
                    Cargar más
                  </button>
                )}
                {viendo.storage_path && (
                  <button className="ap-btn" onClick={() => descargar(viendo)}>
                    <Download size={15} /> Descargar original
                  </button>
                )}
                <button
                  className="ap-btn ap-btn-peligro"
                  onClick={() => { setPorEliminar(viendo); setConfirmacion(''); cerrarVisor(); }}
                >
                  <Trash2 size={15} /> Eliminar del índice
                </button>
              </div>
            </footer>
          </div>
        </div>
      )}

      {porEliminar && (
        <div className="ap-modal-fondo" onClick={() => !eliminando && setPorEliminar(null)}>
          <div className="ap-modal" onClick={(e) => e.stopPropagation()}>
            <h3><AlertTriangle size={18} /> Eliminar documento del índice</h3>
            <p>
              Se eliminarán <strong>{(porEliminar.chunks || 0).toLocaleString('es-CR')} fragmentos</strong> de:
            </p>
            <p className="ap-modal-archivo">{porEliminar.filename}</p>
            <p>
              El asistente dejará de usar esa información. La acción no se puede deshacer.
            </p>
            <p className="ap-modal-instruccion">
              Para confirmar, escriba <code>{PALABRA_CONFIRMACION}</code>
            </p>
            <input
              type="text"
              value={confirmacion}
              onChange={(e) => setConfirmacion(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && confirmado) eliminar(); }}
              placeholder={PALABRA_CONFIRMACION}
              aria-label={`Escriba ${PALABRA_CONFIRMACION} para confirmar`}
              autoFocus
            />
            <div className="ap-modal-acciones">
              <button className="ap-btn" onClick={() => setPorEliminar(null)} disabled={eliminando}>
                Cancelar
              </button>
              <button
                className="ap-btn ap-btn-peligro"
                onClick={eliminar}
                disabled={eliminando || !confirmado}
              >
                {eliminando ? <Loader2 size={15} className="ap-girando" /> : <Trash2 size={15} />}
                Eliminar definitivamente
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DocumentsManager;
