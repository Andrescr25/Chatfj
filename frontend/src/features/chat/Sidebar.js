import React from 'react';
import {
  Menu, X, MessageSquarePlus, MessageSquare, Clock, Trash2,
  AlertTriangle, Settings, Sun, Moon
} from 'lucide-react';

/**
 * Barra lateral: conversaciones, tema y accesos de administración.
 *
 * No tiene estado propio: todo llega por props desde App, que es quien
 * gobierna las conversaciones.
 */
function Sidebar({
  abierta,
  onAbrirCambiar,
  conversaciones,
  conversacionActual,
  conversacionActualId,
  onSeleccionar,
  onNueva,
  onEliminar,
  formatearFecha,
  tema,
  onCambiarTema,
  esAdministrador,
  onAbrirPanel,
  onCerrarSesion,
}) {
  return (
    <>
      {/* Mobile Menu Toggle Button */}
      <button 
        className="mobile-menu-toggle"
        onClick={() => onAbrirCambiar(!abierta)}
        aria-label="Toggle menu"
      >
        {abierta ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Mobile Overlay */}
      {abierta && (
        <div 
          className="mobile-overlay"
          onClick={() => onAbrirCambiar(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`sidebar ${abierta ? 'sidebar-open' : ''}`}>
        <button 
          className="new-chat-btn"
          onClick={() => {
            onNueva();
            onAbrirCambiar(false); // Close sidebar on mobile when creating new conversation
          }}
          disabled={conversacionActual.messages.length === 0}
        >
          <MessageSquarePlus size={16} /> Nueva conversación
        </button>
        
        {conversacionActual.messages.length === 0 && (
          <p className="warning"><AlertTriangle size={12} style={{marginRight: '4px', verticalAlign: 'text-bottom'}}/> Escribe algo primero</p>
        )}

        <div className="conversaciones-list">
          {conversaciones.map(conv => (
            <div key={conv.id} className="conversation-item-wrapper">
              <button
                className={`conversation-item ${conv.id === conversacionActualId ? 'active' : ''}`}
                onClick={() => {
                  onSeleccionar(conv.id);
                  onAbrirCambiar(false); // Close sidebar on mobile when selecting conversation
                }}
              >
                <div className="conversation-title">
                  <MessageSquare size={14} style={{marginTop: '2px'}}/> {conv.title}
                </div>
                <div className="conversation-date">
                  <Clock size={12} /> {formatearFecha(conv.timestamp)}
                </div>
              </button>
              <button
                className="delete-btn"
                onClick={() => onEliminar(conv.id)}
                disabled={conversaciones.length === 1}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          {/* Theme Toggle Switch en sidebar (móvil) */}
          <div className="tema-toggle-container sidebar-tema-toggle">
            <div 
              className="tema-toggle-switch"
              data-tema={tema}
              onClick={onCambiarTema}
              role="button"
              aria-label={tema === 'light' ? 'Cambiar a tema oscuro' : 'Cambiar a tema claro'}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onCambiarTema();
                }
              }}
            >
              <span className="tema-icon sun">
                <Sun size={14} />
              </span>
              <span className="tema-icon moon">
                <Moon size={14} />
              </span>
              <div className="tema-toggle-slider"></div>
            </div>
          </div>
          
          {esAdministrador && (
            <>
              <button
                className="training-mode-btn"
                onClick={() => {
                  onAbrirPanel();
                  onAbrirCambiar(false);
                }}
                title="Panel de administración"
              >
                <Settings size={16} /> Panel de administración
              </button>
              <button
                className="admin-logout-btn"
                onClick={onCerrarSesion}
                title="Cerrar Sesión Súper Usuario"
                style={{
                  marginTop: '8px',
                  backgroundColor: '#dc2626',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  cursor: 'pointer',
                  width: '100%',
                  fontSize: '13px',
                  fontWeight: '500',
                  justifyContent: 'center',
                  transition: 'background-color 0.2s',
                  boxSizing: 'border-box'
                }}
              >
                Cerrar Sesión Admin
              </button>
            </>
          )}
          <p>Chat FJ v2.0</p>
          <p>Poder Judicial CR 🇨🇷</p>
        </div>
      </div>
    </>
  );
}

export default Sidebar;
