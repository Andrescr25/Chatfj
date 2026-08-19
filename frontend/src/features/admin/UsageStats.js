import React, { useCallback, useEffect, useState } from 'react';
import { Loader2, AlertTriangle, RefreshCw } from 'lucide-react';
import apiService from '../../api/client';

/**
 * Qué documentos sostienen las respuestas.
 *
 * Es la contracara del inventario: no basta con tener un documento indexado,
 * importa si el asistente lo usa. Un documento en cero puede significar que
 * nadie pregunta por ese tema, o que quedó mal extraído.
 */
function UsageStats() {
  const [datos, setDatos] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');

  const cargar = useCallback(async () => {
    setCargando(true);
    try {
      setDatos(await apiService.getDocumentStats(30));
      setError('');
    } catch (e) {
      setError(e.message || 'No se pudieron cargar las estadísticas.');
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => { cargar(); }, [cargar]);

  const maximo = datos?.documentos?.[0]?.consultas || 1;

  return (
    <div className="ap-seccion">
      <div className="ap-encabezado-seccion">
        <div>
          <h3>Documentos más consultados</h3>
          <p>
            Cuántas veces cada documento respaldó una respuesta. El conteo empieza desde
            que se activó el registro, así que un documento en cero no significa que esté
            mal: puede ser que nadie haya preguntado por ese tema todavía.
          </p>
        </div>
        <button className="ap-btn" onClick={cargar} disabled={cargando}>
          {cargando ? <Loader2 size={15} className="ap-girando" /> : <RefreshCw size={15} />}
          Actualizar
        </button>
      </div>

      {error && <div className="ap-alerta ap-alerta-error"><AlertTriangle size={15} />{error}</div>}

      {cargando && !datos && (
        <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Cargando...</div>
      )}

      {datos && !datos.disponible && (
        <div className="ap-vacio">El registro de uso no está disponible en este momento.</div>
      )}

      {datos?.disponible && (
        <>
          <div className="ap-tarjetas">
            <div className="ap-tarjeta-dato">
              <strong>{(datos.total_consultas || 0).toLocaleString('es-CR')}</strong>
              <span>Documentos citados</span>
            </div>
            <div className="ap-tarjeta-dato">
              <strong>{datos.documentos_distintos || 0}</strong>
              <span>Documentos distintos usados</span>
            </div>
          </div>

          {datos.documentos.length === 0 ? (
            <div className="ap-vacio">
              Todavía no hay consultas registradas. Haga una pregunta en el chat y vuelva.
            </div>
          ) : (
            <div className="ap-ranking">
              {datos.documentos.map((d, i) => (
                <div key={d.documento} className="ap-ranking-fila">
                  <span className="ap-ranking-puesto">{i + 1}</span>
                  <div className="ap-ranking-datos">
                    <div className="ap-ranking-nombre" title={d.documento}>{d.documento}</div>
                    <div className="ap-ranking-barra">
                      <span style={{ width: `${Math.max((d.consultas / maximo) * 100, 2)}%` }} />
                    </div>
                  </div>
                  <span className="ap-ranking-cifra">
                    {d.consultas} <em>{d.porcentaje}%</em>
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default UsageStats;
