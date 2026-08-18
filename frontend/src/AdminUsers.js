import React, { useCallback, useEffect, useState } from 'react';
import {
  UserPlus, Loader2, AlertTriangle, CheckCircle2, ShieldCheck,
  Copy, Ban, CircleCheck, Trash2
} from 'lucide-react';
import apiService from './services/api';

function formatearFecha(iso) {
  if (!iso) return 'Nunca';
  try {
    return new Date(iso).toLocaleDateString('es-CR', {
      day: '2-digit', month: '2-digit', year: 'numeric'
    });
  } catch (e) {
    return '—';
  }
}

function AdminUsers({ me }) {
  const [admins, setAdmins] = useState([]);
  const [cargando, setCargando] = useState(true);
  const [error, setError] = useState('');
  const [aviso, setAviso] = useState('');
  const [enlace, setEnlace] = useState('');

  const [mostrarForm, setMostrarForm] = useState(false);
  const [email, setEmail] = useState('');
  const [nombre, setNombre] = useState('');
  const [definirClave, setDefinirClave] = useState(false);
  const [clave, setClave] = useState('');
  const [guardando, setGuardando] = useState(false);

  const [porRevocar, setPorRevocar] = useState(null);
  const [revocando, setRevocando] = useState(false);

  const cargar = useCallback(async () => {
    setCargando(true);
    try {
      const data = await apiService.listAdmins();
      setAdmins(data.admins || []);
      setError('');
    } catch (e) {
      setError(e.message || 'No se pudo cargar la lista de administradores.');
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => { cargar(); }, [cargar]);

  const limpiarFormulario = () => {
    setEmail(''); setNombre('');
    setDefinirClave(false); setClave(''); setMostrarForm(false);
  };

  const crear = async (e) => {
    e.preventDefault();
    if (guardando) return;

    // Validación antes de llamar al servidor, para explicar el problema en su sitio
    if (definirClave && clave.trim().length < 8) {
      setError('La contraseña debe tener al menos 8 caracteres.');
      return;
    }

    setGuardando(true);
    setError('');
    setAviso('');
    setEnlace('');
    try {
      const nuevo = await apiService.createAdmin({
        email: email.trim().toLowerCase(),
        name: nombre.trim(),
        password: definirClave ? clave : null,
      });
      setAviso(`${nuevo.email} ya tiene acceso de administración.`);
      if (nuevo.password_reset_link) {
        setEnlace(nuevo.password_reset_link);
      }
      limpiarFormulario();
      cargar();
    } catch (e) {
      setError(e.message || 'No se pudo crear la cuenta.');
    } finally {
      setGuardando(false);
    }
  };

  const alternarEstado = async (admin) => {
    setError(''); setAviso('');
    try {
      await apiService.updateAdmin(admin.uid, { disabled: !admin.disabled });
      setAviso(`${admin.email} quedó ${admin.disabled ? 'habilitado' : 'deshabilitado'}.`);
      cargar();
    } catch (e) {
      setError(e.message || 'No se pudo cambiar el estado de la cuenta.');
    }
  };

  const revocar = async () => {
    if (!porRevocar || revocando) return;
    setRevocando(true);
    setError('');
    try {
      await apiService.removeAdmin(porRevocar.uid);
      setAviso(`Se revocó el acceso de ${porRevocar.email}.`);
      setPorRevocar(null);
      cargar();
    } catch (e) {
      setError(e.message || 'No se pudo revocar el acceso.');
    } finally {
      setRevocando(false);
    }
  };

  const copiarEnlace = () => {
    navigator.clipboard?.writeText(enlace);
    setAviso('Enlace copiado. Envíeselo a la persona por un medio seguro.');
  };

  return (
    <div className="ap-seccion">
      <div className="ap-encabezado-seccion">
        <div>
          <h3>Personas con acceso administrativo</h3>
          <p>
            Quien aparece en esta lista puede entrenar el asistente, gestionar los documentos
            indexados y dar o quitar acceso a otras personas.
          </p>
        </div>
        <button className="ap-btn ap-btn-primario" onClick={() => setMostrarForm(v => !v)}>
          <UserPlus size={15} /> Agregar
        </button>
      </div>

      {mostrarForm && (
        <form className="ap-form-subida" onSubmit={crear}>
          <label>
            Correo electrónico
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="persona@poder-judicial.go.cr"
              required
            />
          </label>
          <label>
            Nombre completo
            <input
              type="text"
              value={nombre}
              onChange={(e) => setNombre(e.target.value)}
              placeholder="Nombre que aparecerá en las correcciones"
            />
          </label>
          <label className="ap-checkbox">
            <input
              type="checkbox"
              checked={definirClave}
              onChange={(e) => setDefinirClave(e.target.checked)}
            />
            Definir una contraseña ahora (si no, se genera un enlace para que la persona cree la suya)
          </label>
          {definirClave && (
            <label>
              Contraseña temporal
              <input
                type="password"
                value={clave}
                onChange={(e) => setClave(e.target.value)}
                minLength={8}
                placeholder="Mínimo 8 caracteres"
                required
              />
            </label>
          )}
          <div className="ap-form-acciones">
            <button type="submit" className="ap-btn ap-btn-primario" disabled={guardando}>
              {guardando ? <Loader2 size={15} className="ap-girando" /> : <UserPlus size={15} />}
              Dar acceso
            </button>
            <button type="button" className="ap-btn" onClick={limpiarFormulario} disabled={guardando}>
              Cancelar
            </button>
          </div>
        </form>
      )}

      {error && <div className="ap-alerta ap-alerta-error"><AlertTriangle size={15} />{error}</div>}
      {aviso && <div className="ap-alerta ap-alerta-ok"><CheckCircle2 size={15} />{aviso}</div>}

      {enlace && (
        <div className="ap-alerta ap-alerta-info ap-enlace-clave">
          <div>
            <strong>Enlace para definir la contraseña</strong>
            <code>{enlace}</code>
          </div>
          <button className="ap-btn" onClick={copiarEnlace}><Copy size={15} /> Copiar</button>
        </div>
      )}

      {cargando ? (
        <div className="ap-cargando"><Loader2 size={20} className="ap-girando" /> Cargando...</div>
      ) : (
        <div className="ap-tabla-contenedor">
          <table className="ap-tabla">
            <colgroup>
              <col className="ap-col-persona" />
              <col className="ap-col-estado-admin" />
              <col className="ap-col-ingreso" />
              <col className="ap-col-invita" />
              <col className="ap-col-acciones" />
            </colgroup>
            <thead>
              <tr>
                <th>Persona</th>
                <th>Estado</th>
                <th className="ap-col-oculta-md">Último ingreso</th>
                <th className="ap-col-oculta-md">Invitada por</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {admins.map(admin => {
                const esYo = me && admin.uid === me.uid;
                const bloqueado = esYo || admin.protected;
                const motivo = admin.protected
                  ? 'Cuenta protegida por la configuración del servidor'
                  : (esYo ? 'No puede modificar su propia cuenta' : '');
                return (
                  <tr key={admin.uid}>
                    <td>
                      <div className="ap-doc-nombre">
                        <ShieldCheck size={15} />
                        <div className="ap-doc-textos">
                          <strong title={admin.name || admin.email}>{admin.name || admin.email}</strong>
                          <span title={admin.email}>
                            {admin.email}
                            {esYo ? ' · usted' : ''}
                            {admin.protected ? ' · protegida' : ''}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`ap-badge ${admin.disabled ? 'ap-badge-error' : 'ap-badge-ok'}`}>
                        {admin.disabled ? 'Deshabilitada' : 'Activa'}
                      </span>
                    </td>
                    <td className="ap-celda-secundaria ap-col-oculta-md">
                      {formatearFecha(admin.last_sign_in)}
                    </td>
                    <td className="ap-celda-secundaria ap-col-oculta-md">
                      <span title={admin.invited_by}>{admin.invited_by || '—'}</span>
                    </td>
                    <td>
                      <div className="ap-acciones">
                        <button
                          title={motivo || (admin.disabled ? 'Habilitar' : 'Deshabilitar')}
                          onClick={() => alternarEstado(admin)}
                          disabled={bloqueado}
                        >
                          {admin.disabled ? <CircleCheck size={15} /> : <Ban size={15} />}
                        </button>
                        <button
                          title={motivo || 'Revocar acceso'}
                          className="ap-accion-peligro"
                          onClick={() => setPorRevocar(admin)}
                          disabled={bloqueado}
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {porRevocar && (
        <div className="ap-modal-fondo" onClick={() => !revocando && setPorRevocar(null)}>
          <div className="ap-modal" onClick={(e) => e.stopPropagation()}>
            <h3><AlertTriangle size={18} /> Revocar acceso</h3>
            <p>
              <strong>{porRevocar.email}</strong> perderá el acceso al panel de inmediato:
              se le quita el rol, se deshabilita la cuenta y se cierran sus sesiones activas.
              Las correcciones que ya hizo se conservan.
            </p>
            <div className="ap-modal-acciones">
              <button className="ap-btn" onClick={() => setPorRevocar(null)} disabled={revocando}>
                Cancelar
              </button>
              <button className="ap-btn ap-btn-peligro" onClick={revocar} disabled={revocando}>
                {revocando ? <Loader2 size={15} className="ap-girando" /> : <Trash2 size={15} />}
                Revocar acceso
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdminUsers;
