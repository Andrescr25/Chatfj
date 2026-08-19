import React, { useCallback, useEffect, useState } from 'react';
import { Loader2, AlertTriangle, Clock, RefreshCw, Search, FileText } from 'lucide-react';
import apiService from '../../api/client';

function fechaHora(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('es-CR', {
      day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
    });
  } catch (e) {
    return '—';
  }
}

/**
 * Historial de consultas de los últimos 7 días.
 *
 * Se guardan la pregunta, la respuesta y los documentos usados; nada que
 * identifique a quien preguntó. Lo que pasa de 7 días se borra solo.
 */
function History() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');
  const [filtro, setFiltro] = useState('');
  const [abierta, setAbierta] = useState(null);

  const cargar = useCallback(async () => {
    setCargando(true);
    try {
      setDatos(await apiService.getHistory(7, 200));
      setError('');
    } catch (e) {
      setError(e.message || 'No se pudo cargar el historial.');
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => { cargar(); }, [cargar]);

  const normalizar = (t) => (t || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const visibles = (datos?.consultas || []).filter(
    (c) => !filtro.trim() ||
      normalizar(c.pregunta).includes(normalizar(filtro)) ||
      normalizar(c.respuesta).includes(normalizar(filtro))
  );

  return (
    <div className="ap-seccion">
      <div className="ap-encabezado-seccion">
        <div>
          <h3>Historial de consultas</h3>
          <p>
            Lo que se le preguntó al asistente en los últimos 7 días y qué respondió.
            No se guarda ningún dato de quien pregunta, y lo que pasa de una semana
            se elimina solo.
          </p>
        </div>
        <button className="ap-btn" onClick={cargar} disabled={cargando}>
          {cargando ? <Loader2 size={15} className="ap-girando" /> : <RefreshCw size={15} />}
          Actualizar
        </button>
      </div>

      <div className="ap-visor-busqueda">
        <Search size={15} />
        <input
          type="text"
          value={filtro}
          onChange={(e) => setFiltro(e.target.value)}
          placeholder="Buscar en preguntas y respuestas"
        />
        {filtro && <button className="ap-btn" onClick={() => setFiltro('')}>Limpiar</button>}
      </div>

      {error && <div className="ap-alerta ap-alerta-error"><AlertTriangle size={15} />{error}</div>}

      {cargando && !datos && (
        <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Cargando historial...</div>
      )}

      {datos && !datos.disponible && (
        <div className="ap-vacio">El registro de consultas no está disponible en este momento.</div>
      )}

      {datos?.disponible && (
        visibles.length === 0 ? (
          <div className="ap-vacio">
            {(datos.consultas || []).length === 0
              ? 'No hay consultas registradas en los últimos 7 días.'
              : `Ninguna consulta coincide con «${filtro}».`}
          </div>
        ) : (
          <>
            <p className="ap-celda-secundaria">
              {visibles.length} de {datos.total} consultas · últimos {datos.dias} días
            </p>
            <div className="ap-historial">
              {visibles.map((c) => {
                const activa = abierta === c.id;
                return (
                  <article key={c.id} className="ap-consulta">
                    <button
                      className="ap-consulta-cabecera"
                      onClick={() => setAbierta(activa ? null : c.id)}
                    >
                      <span className="ap-consulta-fecha"><Clock size={13} /> {fechaHora(c.creado_en)}</span>
                      <span className="ap-consulta-pregunta">{c.pregunta}</span>
                    </button>
                    {activa && (
                      <div className="ap-consulta-cuerpo">
                        <p>{c.respuesta}</p>
                        {(c.documentos || []).length > 0 && (
                          <div className="ap-consulta-fuentes">
                            <FileText size={13} />
                            {c.documentos.join(' · ')}
                          </div>
                        )}
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </>
        )
      )}
    </div>
  );
}

export default History;
