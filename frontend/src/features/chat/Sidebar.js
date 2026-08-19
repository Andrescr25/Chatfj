import React from 'react';
import {
  Menu, X, MessageSquarePlus, MessageSquare, Clock, Trash2,
  AlertTriangle, Settings, Sun, Moon, Scale, LogOut
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
      <aside className={`sidebar ${abierta ? 'sidebar-open' : ''}`} aria-label="Conversaciones">
        <div className="sidebar-marca">
          <span className="sidebar-marca-icono"><Scale size={18} /></span>
          <span className="sidebar-marca-textos">
            <strong>Chat FJ</strong>
            <span>Facilitadores Judiciales</span>
          </span>
          <button
            className="sidebar-cerrar"
            onClick={() => onAbrirCambiar(false)}
            aria-label="Cerrar menú"
          >
            <X size={20} />
          </button>
        </div>

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

        <div className="conversations-list">
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
                  <MessageSquare size={14} />
                  <span>{conv.title}</span>
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
          <div className="theme-toggle-container sidebar-theme-toggle">
            <div 
              className="theme-toggle-switch"
              data-theme={tema}
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
              <span className="theme-icon sun">
                <Sun size={14} />
              </span>
              <span className="theme-icon moon">
                <Moon size={14} />
              </span>
              <div className="theme-toggle-slider"></div>
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
                title="Cerrar sesión de administración"
              >
                <LogOut size={15} /> Cerrar sesión
              </button>
            </>
          )}
          <p>Chat FJ v2.0</p>
          <p>Poder Judicial CR 🇨🇷</p>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
