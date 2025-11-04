import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import TrainingMode from './TrainingMode';

function App() {
  const [conversations, setConversations] = useState([
    { id: 1, title: 'Nueva conversación', timestamp: new Date(), messages: [] }
  ]);
  const [currentConvId, setCurrentConvId] = useState(1);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showReferenceModal, setShowReferenceModal] = useState(false);
  const [selectedReference, setSelectedReference] = useState(null);
  const [showTrainingMode, setShowTrainingMode] = useState(false);
  const messagesEndRef = useRef(null);
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const currentConv = conversations.find(c => c.id === currentConvId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentConv?.messages]);

  const createNewConversation = () => {
    if (currentConv.messages.length === 0) return;
    
    const newConv = {
      id: Date.now(),
      title: 'Nueva conversación',
      timestamp: new Date(),
      messages: []
    };
    setConversations([newConv, ...conversations]);
    setCurrentConvId(newConv.id);
    setInput('');
  };

  const deleteConversation = (id) => {
    if (conversations.length === 1) return;
    
    const newConvs = conversations.filter(c => c.id !== id);
    setConversations(newConvs);
    
    if (id === currentConvId) {
      setCurrentConvId(newConvs[0].id);
    }
  };

  const updateConversationTitle = (conv) => {
    const userMessages = conv.messages.filter(m => m.role === 'user');
    if (userMessages.length > 0) {
      const firstMsg = userMessages[0].content;
      conv.title = firstMsg.substring(0, 30) + (firstMsg.length > 30 ? '...' : '');
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    const updatedConv = {
      ...currentConv,
      messages: [...currentConv.messages, userMessage],
      timestamp: new Date()
    };

    setConversations(conversations.map(c => 
      c.id === currentConvId ? updatedConv : c
    ));
    
    updateConversationTitle(updatedConv);
    setInput('');
    setIsLoading(true);

    try {
      const history = currentConv.messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          history: history
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [] // Guardar fuentes para referencias clickeables
      };
      setConversations(conversations.map(c =>
        c.id === currentConvId
          ? { ...c, messages: [...updatedConv.messages, assistantMessage] }
          : c
      ));
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Lo siento, hubo un error al conectar con el servidor. Por favor intenta de nuevo.' 
      };
      setConversations(conversations.map(c => 
        c.id === currentConvId 
          ? { ...c, messages: [...updatedConv.messages, errorMessage] }
          : c
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setInput(example);
  };

  const formatTimestamp = (timestamp) => {
    const now = new Date();
    const diff = Math.floor((now - timestamp) / 1000 / 60 / 60 / 24);

    if (diff === 0) return 'Hoy';
    if (diff === 1) return 'Ayer';
    if (diff < 7) return `Hace ${diff} días`;
    return timestamp.toLocaleDateString('es-CR', { day: '2-digit', month: '2-digit' });
  };

  // Manejar click en referencias [1], [2], [Web]
  const handleReferenceClick = (refNumber, sources) => {
    const source = sources.find(s => s.reference_number === parseInt(refNumber));
    if (!source) return;

    // Mostrar modal para todos los tipos de referencias
    setSelectedReference(source);
    setShowReferenceModal(true);
  };

  // Renderizar mensaje con referencias clickeables
  const renderMessageWithReferences = (content, sources) => {
    if (!sources || sources.length === 0) {
      return content;
    }

    // Dividir en líneas para procesar
    const lines = content.split('\n');
    return lines.map((line, lineIdx) => {
      // Detectar si es una línea de referencia con URL
      const urlRefMatch = line.match(/^\[(\d+)\]\s+(.+?)\s+-\s+(https?:\/\/.+)$/);

      if (urlRefMatch) {
        const [, refNum, title, url] = urlRefMatch;
        return (
          <React.Fragment key={lineIdx}>
            <span>
              <button
                className="reference-link"
                onClick={() => handleReferenceClick(refNum, sources)}
                title="Click para ver fuente"
              >
                [{refNum}]
              </button>
              {' '}
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="reference-url-link"
                title="Abrir en nueva pestaña"
              >
                {title} 🔗
              </a>
            </span>
            {lineIdx < lines.length - 1 && <br />}
          </React.Fragment>
        );
      }

      // Buscar referencias [número] en la línea
      const parts = line.split(/(\[\d+\]|\[Web\])/g);

      return (
        <React.Fragment key={lineIdx}>
          {parts.map((part, partIdx) => {
            // Si es una referencia [número] o [Web]
            const refMatch = part.match(/^\[(\d+|Web)\]$/);
            if (refMatch) {
              const refNumber = refMatch[1];
              return (
                <button
                  key={partIdx}
                  className="reference-link"
                  onClick={() => handleReferenceClick(refNumber, sources)}
                  title="Click para ver fuente"
                >
                  {part}
                </button>
              );
            }
            return <span key={partIdx}>{part}</span>;
          })}
          {lineIdx < lines.length - 1 && <br />}
        </React.Fragment>
      );
    });
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <button 
          className="new-chat-btn"
          onClick={createNewConversation}
          disabled={currentConv.messages.length === 0}
        >
          ➕ Nueva conversación
        </button>
        
        {currentConv.messages.length === 0 && (
          <p className="warning">⚠️ Escribe algo primero</p>
        )}

        <div className="conversations-list">
          {conversations.map(conv => (
            <div key={conv.id} className="conversation-item-wrapper">
              <button
                className={`conversation-item ${conv.id === currentConvId ? 'active' : ''}`}
                onClick={() => setCurrentConvId(conv.id)}
              >
                <div className="conversation-title">
                  💬 {conv.title}
                </div>
                <div className="conversation-date">
                  🕐 {formatTimestamp(conv.timestamp)}
                </div>
              </button>
              <button
                className="delete-btn"
                onClick={() => deleteConversation(conv.id)}
                disabled={conversations.length === 1}
              >
                🗑️
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button
            className="training-mode-btn"
            onClick={() => setShowTrainingMode(true)}
            title="Modo Entrenamiento"
          >
            🎓 Modo Entrenamiento
          </button>
          <p>Chat FJ v2.0</p>
          <p>Poder Judicial CR 🇨🇷</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <h1>Chat FJ</h1>
          <p>Servicio Nacional de Facilitadoras y Facilitadores Judiciales</p>
        </div>

        <div className="chat-container">
          {currentConv.messages.length === 0 ? (
            <div className="welcome-screen">
              <h2>¿En qué puedo ayudarte hoy?</h2>
              <p>Estoy aquí para orientarte sobre temas legales y judiciales en Costa Rica</p>
              
              <div className="examples">
                <p className="examples-title">Ejemplos de consultas</p>
                <div className="example-cards">
                  <button 
                    className="example-card"
                    onClick={() => handleExampleClick('Mi ex no paga pensión, ¿qué hago?')}
                  >
                    <h4>💰 Pensión</h4>
                    <p>Mi ex no paga pensión, ¿qué hago?</p>
                  </button>
                  <button 
                    className="example-card"
                    onClick={() => handleExampleClick('¿Cuánto dura una conciliación?')}
                  >
                    <h4>⚖️ Conciliación</h4>
                    <p>¿Cuánto dura una conciliación?</p>
                  </button>
                  <button 
                    className="example-card"
                    onClick={() => handleExampleClick('Mi jefe no me paga horas extra')}
                  >
                    <h4>👔 Laboral</h4>
                    <p>Mi jefe no me paga horas extra</p>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="messages">
              {currentConv.messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}-message`}>
                  <div className="message-content">
                    {msg.role === 'assistant'
                      ? renderMessageWithReferences(msg.content, msg.sources)
                      : msg.content.split('\n').map((line, i) => (
                          <React.Fragment key={i}>
                            {line}
                            {i < msg.content.split('\n').length - 1 && <br />}
                          </React.Fragment>
                        ))
                    }
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant-message">
                  <div className="message-content loading">
                    ⚖️ Pensando...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <form className="input-section" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Envía un mensaje a Chat FJ... (Presiona Enter para enviar)"
            disabled={isLoading}
          />
        </form>

        <div className="footer">
          Chat FJ puede cometer errores. Verifica la información importante con fuentes oficiales.
        </div>
      </div>

      {/* Modal para mostrar contenido de referencias */}
      {showReferenceModal && selectedReference && (
        <div className="reference-modal-overlay" onClick={() => setShowReferenceModal(false)}>
          <div className="reference-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📄 {selectedReference.filename || 'Fuente'}</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowReferenceModal(false)}
              >
                ✕
              </button>
            </div>
            <div className="modal-content">
              {selectedReference.type === 'web' ? (
                // Si es referencia web, mostrar botón para abrir URL
                <div className="web-reference-content">
                  <p className="web-reference-description">
                    {selectedReference.content || selectedReference.snippet || selectedReference.title}
                  </p>
                  <a
                    href={selectedReference.url || selectedReference.source}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="web-reference-button"
                  >
                    🌐 Abrir sitio web
                  </a>
                </div>
              ) : (
                // Si es documento, mostrar contenido
                (() => {
                  const content = selectedReference.content || selectedReference.snippet;
                  // Si el contenido tiene múltiples fragmentos separados por ---
                  const fragments = content.split('\n\n---\n\n');

                  if (fragments.length > 1) {
                    return fragments.map((fragment, idx) => (
                      <div key={idx} className="content-fragment">
                        {idx > 0 && <div className="fragment-separator">• • •</div>}
                        <p>{fragment.trim()}</p>
                      </div>
                    ));
                  } else {
                    return <p>{content}</p>;
                  }
                })()
              )}
            </div>
            <div className="modal-footer">
              <button
                className="modal-close-footer-btn"
                onClick={() => setShowReferenceModal(false)}
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modo Entrenamiento */}
      {showTrainingMode && (
        <TrainingMode onClose={() => setShowTrainingMode(false)} />
      )}
    </div>
  );
}

export default App;
