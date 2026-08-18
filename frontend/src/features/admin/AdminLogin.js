import React, { useState } from 'react';
import { Lock, Loader2 } from 'lucide-react';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../../config/firebase';

/**
 * Pantalla de ingreso al panel de administración (ruta /admin).
 *
 * Maneja su propio estado y su propio error: App solo decide cuándo mostrarla.
 * La sesión abierta la detecta el listener de Firebase que vive en App.
 */
function AdminLogin({ theme }) {
  const [email, setEmail] = useState('');
  const [clave, setClave] = useState('');
  const [error, setError] = useState('');
  const [enviando, setEnviando] = useState(false);

  const manejarEnvio = async (e) => {
    e.preventDefault();
    if (!email.trim() || !clave.trim() || enviando) return;

    setEnviando(true);
    setError('');
    try {
      const credenciales = await signInWithEmailAndPassword(auth, email.trim(), clave.trim());
      const token = await credenciales.user.getIdToken();
      localStorage.setItem('adminToken', token);
      setEmail('');
      setClave('');
    } catch (err) {
      console.error(err);
      let mensaje = 'Error al iniciar sesión. Verifique sus credenciales.';
      if (
        err.code === 'auth/user-not-found' ||
        err.code === 'auth/wrong-password' ||
        err.code === 'auth/invalid-credential'
      ) {
        mensaje = 'Correo electrónico o contraseña incorrectos.';
      } else if (err.code === 'auth/invalid-email') {
        mensaje = 'Formato de correo electrónico inválido.';
      }
      setError(mensaje);
    } finally {
      setEnviando(false);
    }
  };

  return (
  <div className="admin-login-container" style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      background: theme === 'dark' ? 'radial-gradient(circle, #1e293b 0%, #0f172a 100%)' : 'radial-gradient(circle, #f8fafc 0%, #e2e8f0 100%)',
      fontFamily: "'Outfit', 'Inter', sans-serif",
      padding: '20px',
      color: theme === 'dark' ? '#f8fafc' : '#0f172a'
    }}>
      <div className="admin-login-card" style={{
        background: theme === 'dark' ? 'rgba(30, 41, 59, 0.7)' : 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        border: theme === 'dark' ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.1)',
        borderRadius: '16px',
        padding: '40px 30px',
        width: '100%',
        maxWidth: '400px',
        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        <div style={{
          background: '#1d4ed8',
          color: '#ffffff',
          padding: '16px',
          borderRadius: '50%',
          marginBottom: '20px',
          boxShadow: '0 4px 14px 0 rgba(29, 78, 216, 0.4)'
        }}>
          <Lock size={32} />
        </div>
        <h2 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px', textAlign: 'center' }}>
          Acceso Súper Usuario
        </h2>
        <p style={{ 
          fontSize: '14px', 
          color: theme === 'dark' ? '#94a3b8' : '#64748b', 
          marginBottom: '30px', 
          textAlign: 'center',
          lineHeight: '1.5'
        }}>
          Ingrese sus credenciales de súper usuario para habilitar el Modo Entrenamiento del asistente.
        </p>

        <form onSubmit={manejarEnvio} style={{ width: '100%' }}>
          <div style={{ marginBottom: '20px' }}>
            <label htmlFor="admin-email" style={{
              display: 'block',
              fontSize: '12px',
              fontWeight: '600',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: '8px',
              color: theme === 'dark' ? '#94a3b8' : '#64748b'
            }}>
              Correo Electrónico
            </label>
            <input
              id="admin-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="usuario@poder-judicial.go.cr"
              required
              style={{
                width: '100%',
                padding: '12px 16px',
                borderRadius: '8px',
                border: theme === 'dark' ? '1px solid #475569' : '1px solid #cbd5e1',
                background: theme === 'dark' ? '#0f172a' : '#ffffff',
                color: theme === 'dark' ? '#f8fafc' : '#0f172a',
                fontSize: '16px',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s',
                marginBottom: '15px'
              }}
            />

            <label htmlFor="admin-password" style={{
              display: 'block',
              fontSize: '12px',
              fontWeight: '600',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              marginBottom: '8px',
              color: theme === 'dark' ? '#94a3b8' : '#64748b'
            }}>
              Contraseña
            </label>
            <input
              id="admin-password"
              type="password"
              value={clave}
              onChange={(e) => setClave(e.target.value)}
              placeholder="••••••••"
              required
              style={{
                width: '100%',
                padding: '12px 16px',
                borderRadius: '8px',
                border: theme === 'dark' ? '1px solid #475569' : '1px solid #cbd5e1',
                background: theme === 'dark' ? '#0f172a' : '#ffffff',
                color: theme === 'dark' ? '#f8fafc' : '#0f172a',
                fontSize: '16px',
                outline: 'none',
                boxSizing: 'border-box',
                transition: 'border-color 0.2s'
              }}
            />
          </div>

          {error && (
            <div style={{
              color: '#ef4444',
              fontSize: '13px',
              marginBottom: '20px',
              textAlign: 'center',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              padding: '10px',
              borderRadius: '6px',
              border: '1px solid rgba(239, 68, 68, 0.2)'
            }}>
              ⚠️ {error}
            </div>
          )}

          <button
            type="submit"
            disabled={enviando}
            style={{
              width: '100%',
              padding: '14px',
              background: '#1d4ed8',
              color: '#ffffff',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              fontSize: '15px',
              cursor: enviando ? 'not-allowed' : 'pointer',
              boxShadow: '0 4px 14px 0 rgba(29, 78, 216, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              transition: 'background-color 0.2s'
            }}
          >
            {enviando ? (
              <><Loader2 size={18} className="animate-spin" /> Verificando...</>
            ) : (
              'Iniciar Sesión'
            )}
          </button>
        </form>

        <a 
          href="/"
          style={{
            marginTop: '25px',
            fontSize: '14px',
            color: '#3b82f6',
            textDecoration: 'none',
            fontWeight: '500'
          }}
        >
          ← Volver al Chat Público
        </a>
      </div>
    </div>
  );
}

export default AdminLogin;
