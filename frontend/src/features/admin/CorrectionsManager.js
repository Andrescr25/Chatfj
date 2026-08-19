import React, { useCallback, useEffect, useState } from 'react';
import {
  Loader2, AlertTriangle, CheckCircle2, Pencil, Trash2, GraduationCap, Search
} from 'lucide-react';
import apiService from '../../api/client';

function formatearFecha(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleDateString('es-CR', { day: '2-digit', month: '2-digit', year: 'numeric' });
  } catch (e) {
    return '—';
  }
}

/**
 * Correcciones aprendidas: revisarlas, corregirlas o quitarlas.
 *
 * Estas correcciones tienen prioridad sobre los documentos oficiales, así que
 * una mal escrita se propaga a todas las consultas parecidas. Hasta ahora solo
 * se podían crear.
 */
function CorrectionsManager() {
  const [correcciones, setCorrecciones] = useState([]);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');
  const [aviso, setAviso] = useState('');
  const [filtro, setFiltro] = useState('');

  const [editando, setEditando] = useState(null);
  const [pregunta, setPregunta] = useState('');
  const [texto, setTexto] = useState('');
  const [guardando, setGuardando] = useState(false);

  const [porEliminar, setPorEliminar] = useState(null);
  const [eliminando, setEliminando] = useState(false);

  const cargar = useCallback(async () => {
    setCargando(true);
    try {
      const datos = await apiService.listCorrections();
      setCorrecciones(datos.correcciones || []);
      setError('');
    } catch (e) {
      setError(e.message || 'No se pudieron cargar las correcciones.');
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => { cargar(); }, [cargar]);

  const abrirEdicion = (c) => {
    setEditando(c);
    setPregunta(c.pregunta);
    setTexto(c.correccion);
    setError('');
  };

  const guardar = async () => {
    if (guardando || !editando) return;
    if (!pregunta.trim() || !texto.trim()) {
      setError('La pregunta y la corrección no pueden quedar vacías.');
      return;
    }
    setGuardando(true);
    try {
      await apiService.updateCorrection(editando.id, { pregunta, correccion: texto });
      setAviso('Corrección actualizada. Ya se aplica en las consultas nuevas.');
      setEditando(null);
      cargar();
    } catch (e) {
      setError(e.message || 'No se pudo guardar la corrección.');
    } finally {
      setGuardando(false);
    }
  };

  const eliminar = async () => {
    if (!porEliminar || eliminando) return;
    setEliminando(true);
    try {
      await apiService.deleteCorrection(porEliminar.id);
      setAviso('Corrección eliminada. El asistente vuelve a responder según los documentos.');
      setPorEliminar(null);
      cargar();
    } catch (e) {
      setError(e.message || 'No se pudo eliminar la corrección.');
    } finally {
      setEliminando(false);
    }
  };

  const normalizar = (t) => (t || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const visibles = correcciones.filter(
    (c) => !filtro.trim() ||
      normalizar(c.pregunta).includes(normalizar(filtro)) ||
      normalizar(c.correccion).includes(normalizar(filtro))
  );

  return (
    <div className="ap-seccion">
      <div className="ap-encabezado-seccion">
        <div>
          <h3>Correcciones aprendidas</h3>
          <p>
            Lo que las personas entrenadoras le enseñaron al asistente. Tienen prioridad
            sobre los documentos oficiales, así que una corrección equivocada se nota en
            todas las consultas parecidas.
          </p>
        </div>
      </div>

      <div className="ap-visor-busqueda">
        <Search size={15} />
        <input
          type="text"
          value={filtro}
          onChange={(e) => setFiltro(e.target.value)}
          placeholder="Buscar por pregunta o por contenido"
        />
        {filtro && <button className="ap-btn" onClick={() => setFiltro('')}>Limpiar</button>}
      </div>

      {error && <div className="ap-alerta ap-alerta-error"><AlertTriangle size={15} />{error}</div>}
      {aviso && <div className="ap-alerta ap-alerta-ok"><CheckCircle2 size={15} />{aviso}</div>}

      {cargando ? (
        <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Cargando correcciones...</div>
      ) : visibles.length === 0 ? (
        <div className="ap-vacio">
          {correcciones.length === 0
            ? 'Todavía no hay correcciones. Se crean desde el modo entrenamiento.'
            : `Ninguna corrección coincide con «${filtro}».`}
        </div>
      ) : (
        <>
          <p className="ap-celda-secundaria">
            {visibles.length} de {correcciones.length} correcciones
          </p>
          <div className="ap-lista-correcciones">
            {visibles.map((c) => (
              <article key={c.id} className="ap-correccion">
                <header>
                  <GraduationCap size={15} />
                  <strong>{c.pregunta || '(sin pregunta)'}</strong>
                </header>
                <p>{c.correccion}</p>
                <footer>
                  <span className="ap-celda-secundaria">
                    {c.entrenador || 'anónimo'} · {formatearFecha(c.fecha)}
                    {c.editada_por ? ` · editada por ${c.editada_por}` : ''}
                  </span>
                  <div className="ap-acciones">
                    <button title="Editar" onClick={() => abrirEdicion(c)}>
                      <Pencil size={15} />
                    </button>
                    <button
                      title="Eliminar"
                      className="ap-accion-peligro"
                      onClick={() => setPorEliminar(c)}
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </footer>
              </article>
            ))}
          </div>
        </>
      )}

      {editando && (
        <div className="ap-modal-fondo" onClick={() => !guardando && setEditando(null)}>
          <div className="ap-modal ap-modal-ancho" onClick={(e) => e.stopPropagation()}>
            <h3><Pencil size={18} /> Editar corrección</h3>
            <label className="ap-campo">
              Pregunta que la activa
              <input type="text" value={pregunta} onChange={(e) => setPregunta(e.target.value)} />
            </label>
            <p className="ap-modal-instruccion">
              Si cambia la pregunta, cambia también cuándo se aplica esta corrección.
            </p>
            <label className="ap-campo">
              Respuesta correcta
              <textarea rows={8} value={texto} onChange={(e) => setTexto(e.target.value)} />
            </label>
            <div className="ap-modal-acciones">
              <button className="ap-btn" onClick={() => setEditando(null)} disabled={guardando}>
                Cancelar
              </button>
              <button className="ap-btn ap-btn-primario" onClick={guardar} disabled={guardando}>
                {guardando ? <Loader2 size={15} className="ap-girando" /> : <CheckCircle2 size={15} />}
                Guardar
              </button>
            </div>
          </div>
        </div>
      )}

      {porEliminar && (
        <div className="ap-modal-fondo" onClick={() => !eliminando && setPorEliminar(null)}>
          <div className="ap-modal" onClick={(e) => e.stopPropagation()}>
            <h3><AlertTriangle size={18} /> Eliminar corrección</h3>
            <p>El asistente dejará de aplicarla y volverá a responder según los documentos.</p>
            <p className="ap-modal-archivo">{porEliminar.pregunta}</p>
            <div className="ap-modal-acciones">
              <button className="ap-btn" onClick={() => setPorEliminar(null)} disabled={eliminando}>
                Cancelar
              </button>
              <button className="ap-btn ap-btn-peligro" onClick={eliminar} disabled={eliminando}>
                {eliminando ? <Loader2 size={15} className="ap-girando" /> : <Trash2 size={15} />}
                Eliminar
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CorrectionsManager;
