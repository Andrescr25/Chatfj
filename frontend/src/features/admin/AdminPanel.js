import React, { useEffect, useState } from 'react';
import {
  X, FileText, Users, GraduationCap, Loader2, AlertTriangle, ShieldCheck
} from 'lucide-react';
import './admin.css';
import DocumentsManager from './DocumentsManager';
import AdminUsers from './AdminUsers';
import TrainingChat from '../training/TrainingChat';
import apiService from '../../api/client';

function AdminPanel({ onClose }) {
  const [pestana, setPestana] = useState('documentos');
  const [me, setMe] = useState(null);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');
  const [entrenamientoAbierto, setEntrenamientoAbierto] = useState(false);

  useEffect(() => {
    let activo = true;
    (async () => {
      try {
        // forzar refresco: los roles viajan dentro del token
        await apiService.getToken(true);
        const datos = await apiService.whoAmI();
        if (activo) setMe(datos);
      } catch (e) {
        if (activo) setError(e.message || 'No se pudo verificar su sesión.');
      } finally {
        if (activo) setCargando(false);
      }
    })();
    return () => { activo = false; };
  }, []);

  if (entrenamientoAbierto) {
    return <TrainingChat onClose={() => setEntrenamientoAbierto(false)} />;
  }

  return (
    <div className="ap-overlay">
      <div className="ap-panel">
        <header className="ap-header">
          <div className="ap-header-titulo">
            <h2>Panel de administración</h2>
            {me && (
              <span className="ap-identidad">
                <ShieldCheck size={14} />
                {me.name || me.email} · Administrador
              </span>
            )}
          </div>
          <button className="ap-cerrar" onClick={onClose} title="Cerrar panel">
            <X size={20} />
          </button>
        </header>

        <nav className="ap-tabs">
          <button
            className={pestana === 'documentos' ? 'ap-tab ap-tab-activa' : 'ap-tab'}
            onClick={() => setPestana('documentos')}
          >
            <FileText size={16} /> Documentos
          </button>
          <button
            className={pestana === 'entrenamiento' ? 'ap-tab ap-tab-activa' : 'ap-tab'}
            onClick={() => setPestana('entrenamiento')}
          >
            <GraduationCap size={16} /> Entrenamiento
          </button>
          <button
            className={pestana === 'administradores' ? 'ap-tab ap-tab-activa' : 'ap-tab'}
            onClick={() => setPestana('administradores')}
          >
            <Users size={16} /> Administradores
          </button>
        </nav>

        <div className="ap-contenido">
          {cargando && (
            <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Verificando su sesión...</div>
          )}

          {!cargando && error && (
            <div className="ap-alerta ap-alerta-error">
              <AlertTriangle size={15} />
              {error}
            </div>
          )}

          {!cargando && !error && pestana === 'documentos' && <DocumentsManager />}

          {!cargando && !error && pestana === 'entrenamiento' && (
            <div className="ap-seccion">
              <div className="ap-tarjeta">
                <h3><GraduationCap size={18} /> Modo entrenamiento</h3>
                <p>
                  Haga preguntas al asistente y apruebe o corrija sus respuestas. Las correcciones
                  quedan guardadas con su nombre y tienen prioridad sobre los documentos en las
                  consultas siguientes.
                </p>
                <button
                  className="ap-btn ap-btn-primario"
                  onClick={() => setEntrenamientoAbierto(true)}
                >
                  <GraduationCap size={15} /> Abrir modo entrenamiento
                </button>
              </div>
            </div>
          )}

          {!cargando && !error && pestana === 'administradores' && <AdminUsers me={me} />}
        </div>
      </div>
    </div>
  );
}

export default AdminPanel;
