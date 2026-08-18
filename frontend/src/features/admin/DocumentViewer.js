import React from 'react';
import {
  X, Search, Loader2, AlertTriangle, FileText, Download, Trash2
} from 'lucide-react';

/**
 * Visor del contenido indexado de un documento.
 *
 * Muestra los fragmentos tal como quedaron en el índice, que es lo que el
 * asistente lee al responder. La paginación y la carga las gobierna
 * DocumentsManager; aquí solo se dibuja.
 */
function DocumentViewer({
  documento,
  contenido,
  cargando,
  error,
  filtro,
  onFiltrar,
  fragmentosVisibles,
  onCargarMas,
  onDescargar,
  onEliminar,
  onCerrar,
}) {
  if (!documento) return null;

  return (
    <div className="ap-modal-fondo" onClick={onCerrar}>
        <div className="ap-modal-fondo" onClick={onCerrar}>
          <div className="ap-visor" onClick={(e) => e.stopPropagation()}>
            <header className="ap-visor-header">
              <div className="ap-visor-titulo">
                <h3>{documento.title || documento.filename}</h3>
                <span>
                  {documento.filename}
                  {contenido ? ` · ${contenido.total_fragmentos.toLocaleString('es-CR')} fragmentos` : ''}
                </span>
              </div>
              <button className="ap-cerrar" onClick={onCerrar} title="Cerrar">
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
                onChange={(e) => onFiltrar(e.target.value)}
                placeholder="Buscar dentro de los fragmentos cargados"
              />
              {filtro && (
                <button className="ap-btn" onClick={() => onFiltrar('')}>Limpiar</button>
              )}
            </div>

            <div className="ap-visor-cuerpo">
              {error && (
                <div className="ap-alerta ap-alerta-error">
                  <AlertTriangle size={15} />{error}
                </div>
              )}

              {!contenido && cargando && (
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
                  <button className="ap-btn" onClick={onCargarMas} disabled={cargando}>
                    {cargando
                      ? <Loader2 size={15} className="ap-girando" />
                      : <FileText size={15} />}
                    Cargar más
                  </button>
                )}
                {documento.storage_path && (
                  <button className="ap-btn" onClick={() => onDescargar(documento)}>
                    <Download size={15} /> Descargar original
                  </button>
                )}
                <button
                  className="ap-btn ap-btn-peligro"
                  onClick={() => onEliminar(documento)}
                >
                  <Trash2 size={15} /> Eliminar del índice
                </button>
              </div>
            </footer>
          </div>
        </div>
    </div>
  );
}

export default DocumentViewer;
