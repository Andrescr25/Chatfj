import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import TrainingChat from './TrainingChat';

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
  const [typingMessage, setTypingMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [typingSources, setTypingSources] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    // Cargar tema desde localStorage o usar 'light' por defecto
    return localStorage.getItem('theme') || 'light';
  });
  const messagesEndRef = useRef(null);
  // Usar rutas relativas para que funcione con el proxy y con ngrok
  const API_URL = process.env.REACT_APP_API_URL || '';

  const currentConv = conversations.find(c => c.id === currentConvId);

  // Auto-scroll desactivado para permitir lectura sin interrupciones
  // const scrollToBottom = () => {
  //   messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  // };
  // useEffect(() => {
  //   scrollToBottom();
  // }, [currentConv?.messages]);

  // Aplicar tema al document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Toggle entre tema claro y oscuro
  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'light' ? 'dark' : 'light');
  };

  // Close sidebar when window is resized to desktop size
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768 && sidebarOpen) {
        setSidebarOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sidebarOpen]);

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

  const typeMessage = (fullMessage, sources, updatedConv) => {
    setIsTyping(true);
    setTypingMessage('');
    setTypingSources(sources || []); // Guardar fuentes para la animación

    // Dividir por líneas primero para mantener el formato
    const lines = fullMessage.split('\n');
    let currentText = '';
    let currentLineIndex = 0;
    let currentWordIndex = 0;

    const typeInterval = setInterval(() => {
      if (currentLineIndex < lines.length) {
        const currentLine = lines[currentLineIndex];
        const words = currentLine.split(' ');
        
        if (currentWordIndex < words.length) {
          // Agregar palabra actual
          currentText += (currentWordIndex > 0 ? ' ' : '') + words[currentWordIndex];
          currentWordIndex++;
          setTypingMessage(currentText);
        } else {
          // Pasar a la siguiente línea
          if (currentLineIndex < lines.length - 1) {
            currentText += '\n';
            setTypingMessage(currentText);
          }
          currentLineIndex++;
          currentWordIndex = 0;
        }
      } else {
        clearInterval(typeInterval);
        setIsTyping(false);

        // Una vez terminada la animación, agregar mensaje completo
        // SIN hacer scroll automático
        const assistantMessage = {
          role: 'assistant',
          content: fullMessage,
          sources: sources || []
        };

        setConversations(conversations.map(c =>
          c.id === currentConvId
            ? { ...c, messages: [...updatedConv.messages, assistantMessage] }
            : c
        ));

        setTypingMessage('');
        setTypingSources([]);
      }
    }, 20); // 20ms entre palabras para animación fluida
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

      // Iniciar animación de escritura
      setIsLoading(false);
      typeMessage(data.answer, data.sources, updatedConv);

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

  // Parser de markdown simple (como ChatGPT)
  const parseMarkdownLine = (text) => {
    // No procesar líneas vacías
    if (!text.trim()) return text;

    // Procesar negritas **texto** o __texto__
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Procesar cursiva *texto* o _texto_
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.+?)_/g, '<em>$1</em>');

    return text;
  };

  // Renderizar mensaje con markdown y referencias clickeables
  const renderMessageWithReferences = (content, sources) => {
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

      // Detectar líneas separadoras ---
      if (line.trim() === '---') {
        return (
          <React.Fragment key={lineIdx}>
            <div style={{ borderTop: '1px solid #444', margin: '1rem 0' }} />
          </React.Fragment>
        );
      }

      // Detectar encabezados ## o ### (convertir a emoji + negrita sin mostrar ##)
      const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        const headingText = headingMatch[2];
        const parsedHeading = parseMarkdownLine(headingText);
        return (
          <React.Fragment key={lineIdx}>
            <strong
              className="markdown-heading"
              dangerouslySetInnerHTML={{ __html: '📋 ' + parsedHeading }}
            />
            {lineIdx < lines.length - 1 && <br />}
          </React.Fragment>
        );
      }

      // Detectar listas con - o *
      const listMatch = line.match(/^[-*]\s+(.+)$/);
      if (listMatch) {
        const listText = listMatch[1];
        const parsedList = parseMarkdownLine(listText);
        return (
          <React.Fragment key={lineIdx}>
            <span className="markdown-list-item">
              • <span dangerouslySetInnerHTML={{ __html: parsedList }} />
            </span>
            {lineIdx < lines.length - 1 && <br />}
          </React.Fragment>
        );
      }

      // Procesar referencias [número] en la línea
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
            // Procesar markdown en el texto normal
            const parsedPart = parseMarkdownLine(part);
            return <span key={partIdx} dangerouslySetInnerHTML={{ __html: parsedPart }} />;
          })}
          {lineIdx < lines.length - 1 && <br />}
        </React.Fragment>
      );
    });
  };

  return (
    <div className="app">
      {/* Mobile Menu Toggle Button */}
      <button 
        className="mobile-menu-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle menu"
      >
        {sidebarOpen ? '✕' : '☰'}
      </button>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <button 
          className="new-chat-btn"
          onClick={() => {
            createNewConversation();
            setSidebarOpen(false); // Close sidebar on mobile when creating new conversation
          }}
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
                onClick={() => {
                  setCurrentConvId(conv.id);
                  setSidebarOpen(false); // Close sidebar on mobile when selecting conversation
                }}
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
          {/* Theme Toggle Switch en sidebar (móvil) */}
          <div className="theme-toggle-container sidebar-theme-toggle">
            <div 
              className="theme-toggle-switch"
              data-theme={theme}
              onClick={toggleTheme}
              role="button"
              aria-label={theme === 'light' ? 'Cambiar a tema oscuro' : 'Cambiar a tema claro'}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleTheme();
                }
              }}
            >
              <span className="theme-icon sun">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="5"/>
                  <line x1="12" y1="1" x2="12" y2="3"/>
                  <line x1="12" y1="21" x2="12" y2="23"/>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                  <line x1="1" y1="12" x2="3" y2="12"/>
                  <line x1="21" y1="12" x2="23" y2="12"/>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              </span>
              <span className="theme-icon moon">
                <svg viewBox="0 0 24 24">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              </span>
              <div className="theme-toggle-slider"></div>
            </div>
          </div>
          
          <button
            className="training-mode-btn"
            onClick={() => {
              setShowTrainingMode(true);
              setSidebarOpen(false); // Close sidebar on mobile when opening training mode
            }}
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
          <h1>⚖️ Chat FJ - Servicio Nacional de Facilitadoras y Facilitadores Judiciales</h1>
          
          {/* Theme Toggle Switch */}
          <div className="theme-toggle-container">
            <div 
              className="theme-toggle-switch"
              data-theme={theme}
              onClick={toggleTheme}
              role="button"
              aria-label={theme === 'light' ? 'Cambiar a tema oscuro' : 'Cambiar a tema claro'}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  toggleTheme();
                }
              }}
            >
              <span className="theme-icon sun">
                <svg viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="5"/>
                  <line x1="12" y1="1" x2="12" y2="3"/>
                  <line x1="12" y1="21" x2="12" y2="23"/>
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                  <line x1="1" y1="12" x2="3" y2="12"/>
                  <line x1="21" y1="12" x2="23" y2="12"/>
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              </span>
              <span className="theme-icon moon">
                <svg viewBox="0 0 24 24">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              </span>
              <div className="theme-toggle-slider"></div>
            </div>
          </div>
        </div>

        <div className="chat-container">
          {currentConv.messages.length === 0 ? (
            <div className="welcome-screen">
              <h2>⚖️ ¿En qué puedo ayudarte hoy?</h2>
              <p>Estoy aquí para orientarte sobre temas legales y judiciales en Costa Rica</p>
              
              <div className="example-cards">
                <button 
                  className="example-card"
                  onClick={() => handleExampleClick('Mi ex no paga pensión, ¿qué hago?')}
                >
                  <div className="example-card-icon">💰</div>
                  <div className="example-card-title">Pensión Alimentaria</div>
                  <div className="example-card-text">Mi ex no paga pensión, ¿qué hago?</div>
                </button>
                <button
                  className="example-card"
                  onClick={() => handleExampleClick('¿Que es una conciliación?')}
                >
                  <div className="example-card-icon">⚖️</div>
                  <div className="example-card-title">Conciliación</div>
                  <div className="example-card-text">¿Que es una conciliación?</div>
                </button>
                <button 
                  className="example-card"
                  onClick={() => handleExampleClick('Mi jefe no me paga horas extra')}
                >
                  <div className="example-card-icon">👔</div>
                  <div className="example-card-title">Derecho Laboral</div>
                  <div className="example-card-text">Mi jefe no me paga horas extra</div>
                </button>
              </div>
            </div>
          ) : (
            <div className="messages-container">
              {currentConv.messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? '👤' : '⚖️'}
                  </div>
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
                <div className="typing-indicator">
                  <div className="message-avatar">⚖️</div>
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
              {isTyping && typingMessage && (
                <div className="message assistant">
                  <div className="message-avatar">⚖️</div>
                  <div className="message-content">
                    {renderMessageWithReferences(typingMessage, typingSources)}
                    <span className="typing-cursor">▊</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-section">
          <div className="input-wrapper">
            <div className="input-container">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage(e);
                  }
                }}
                placeholder="Envía un mensaje a Chat FJ..."
                disabled={isLoading}
                rows={1}
                style={{
                  height: 'auto',
                  minHeight: '24px',
                  maxHeight: '200px',
                }}
                onInput={(e) => {
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
              />
              <button
                type="submit"
                className="send-button"
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                title="Enviar mensaje"
              >
                {isLoading ? <div className="spinner"></div> : '↑'}
              </button>
            </div>
            <div className="footer-text">
              Chat FJ puede cometer errores. Verifica la información importante.
            </div>
          </div>
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
        <TrainingChat onClose={() => setShowTrainingMode(false)} />
      )}
    </div>
  );
}

export default App;
