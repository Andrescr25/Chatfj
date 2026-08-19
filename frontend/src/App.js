import React, { useState, useEffect, useRef } from 'react';
import {
  Menu, X, MessageSquarePlus, AlertTriangle, MessageSquare, Clock, Trash2,
  Scale, CircleDollarSign, Briefcase, User, ExternalLink,
  Sun, Moon, Send, Loader2, FileText, Bot, Lock, Settings
} from 'lucide-react';
import './features/chat/chat.css';
import apiService from './api/client';
import AdminPanel from './features/admin/AdminPanel';
import AdminLogin from './features/admin/AdminLogin';
import ReferenceModal from './features/chat/ReferenceModal';
import Sidebar from './features/chat/Sidebar';
import { signOut, onAuthStateChanged } from 'firebase/auth';
import { auth } from './config/firebase';

function App() {
  const [conversations, setConversations] = useState([
    { id: 1, title: 'Nueva conversación', timestamp: new Date(), messages: [] }
  ]);
  const [currentConvId, setCurrentConvId] = useState(1);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showReferenceModal, setShowReferenceModal] = useState(false);
  const [selectedReference, setSelectedReference] = useState(null);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [typingMessage, setTypingMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [typingSources, setTypingSources] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    // Cargar tema desde localStorage o usar 'light' por defecto
    return localStorage.getItem('theme') || 'light';
  });
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const currentConv = conversations.find(c => c.id === currentConvId);

  // El campo de escritura crece con el texto y vuelve a su alto original al
  // enviarlo. Antes el alto se fijaba desde onInput, que no se dispara cuando
  // React limpia el valor: el campo se quedaba enorme y en móvil tapaba media
  // pantalla. El tope lo pone el CSS (max-height), distinto en móvil.
  useEffect(() => {
    const campo = textareaRef.current;
    if (!campo) return;
    campo.style.height = 'auto';
    const tope = parseFloat(getComputedStyle(campo).maxHeight) || 200;
    campo.style.height = `${Math.min(campo.scrollHeight, tope)}px`;
  }, [input]);

  // Estados de autenticación de súper usuario
  const [isAdminRoute, setIsAdminRoute] = useState(false);
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(
    () => !!localStorage.getItem('adminToken')
  );

  // Escuchar cambios de autenticación en Firebase
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        try {
          const idToken = await user.getIdToken();
          localStorage.setItem('adminToken', idToken);
          setIsAdminAuthenticated(true);
        } catch (error) {
          console.error("Error al renovar token de Firebase:", error);
        }
      } else {
        localStorage.removeItem('adminToken');
        setIsAdminAuthenticated(false);
      }
    });
    return () => unsubscribe();
  }, []);

  // Detectar ruta /admin o /#/admin
  useEffect(() => {
    const checkRoute = () => {
      const path = window.location.pathname.toLowerCase();
      const hash = window.location.hash.toLowerCase();
      const params = new URLSearchParams(window.location.search);
      
      const isRoute = 
        path === '/admin' || 
        path === '/admin/' || 
        hash === '#/admin' || 
        hash === '#admin' || 
        params.get('admin') === 'true';
      
      setIsAdminRoute(isRoute);
    };

    checkRoute();
    window.addEventListener('popstate', checkRoute);
    window.addEventListener('hashchange', checkRoute);
    return () => {
      window.removeEventListener('popstate', checkRoute);
      window.removeEventListener('hashchange', checkRoute);
    };
  }, []);

  const handleAdminLogout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error("Error al cerrar sesión:", error);
    }
    localStorage.removeItem('adminToken');
    setIsAdminAuthenticated(false);
    // Redirigir a la raíz para quitar /admin de la URL
    window.location.href = '/';
  };



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

  const typeMessage = (fullMessage, sources, updatedConv, metadata = {}) => {
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
          sources: sources || [],
          matchedQuestion: metadata?.matched_question || '',
          learnedFromFeedback: metadata?.learned_from_feedback || false,
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

      const data = await apiService.ask(input, history);

      // Iniciar animación de escritura
      setIsLoading(false);
      typeMessage(data.answer, data.sources, updatedConv, data);

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

  // Safety check to prevent "Cannot read properties of undefined"
  if (!currentConv) {
    // If somehow we lost the track, reset to first conversation or create one
    if (conversations.length > 0) {
       setCurrentConvId(conversations[0].id);
    }
    return null; // Render nothing while state updates
  }

  if (isAdminRoute && !isAdminAuthenticated) {
    return <AdminLogin theme={theme} />;
  }

  return (
    <div className="app">
      <Sidebar
        abierta={sidebarOpen}
        onAbrirCambiar={setSidebarOpen}
        conversaciones={conversations}
        conversacionActual={currentConv}
        conversacionActualId={currentConvId}
        onSeleccionar={setCurrentConvId}
        onNueva={createNewConversation}
        onEliminar={deleteConversation}
        formatearFecha={formatTimestamp}
        tema={theme}
        onCambiarTema={toggleTheme}
        esAdministrador={isAdminAuthenticated}
        onAbrirPanel={() => setShowAdminPanel(true)}
        onCerrarSesion={handleAdminLogout}
      />

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <h1><Scale size={20} style={{verticalAlign: 'text-bottom', marginRight: '8px'}}/> Chat FJ - Servicio Nacional de Facilitadoras y Facilitadores Judiciales</h1>
        </div>

        <div className="chat-container">
          {currentConv.messages.length === 0 ? (
            <div className="welcome-screen">
              <h2><Scale size={32} style={{verticalAlign: 'middle', marginRight: '10px'}}/> ¿En qué puedo ayudarte hoy?</h2>
              <p>Estoy aquí para orientarte sobre temas legales y judiciales en Costa Rica</p>
              
              <div className="example-cards">
                <button 
                  className="example-card"
                  onClick={() => handleExampleClick('¿Qué pasa si el padre de hijos en común no paga pensión?')}
                >
                  <div className="example-card-icon"><CircleDollarSign size={24}/></div>
                  <div className="example-card-title">Pensión Alimentaria</div>
                  <div className="example-card-text">¿Qué pasa si el padre de hijos en común no paga pensión?</div>
                </button>
                <button
                  className="example-card"
                  onClick={() => handleExampleClick('¿Que es una conciliación?')}
                >
                  <div className="example-card-icon"><Scale size={24}/></div>
                  <div className="example-card-title">Conciliación</div>
                  <div className="example-card-text">¿Que es una conciliación?</div>
                </button>
                <button 
                  className="example-card"
                  onClick={() => handleExampleClick('Mi jefe no me paga horas extra')}
                >
                  <div className="example-card-icon"><Briefcase size={24}/></div>
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
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
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
                  <div className="message-avatar"><Bot size={20} /></div>
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
              {isTyping && typingMessage && (
                <div className="message assistant">
                  <div className="message-avatar"><Bot size={20} /></div>
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
                ref={textareaRef}
              />
              <button
                type="submit"
                className="send-button"
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                title="Enviar mensaje"
              >
                {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
              </button>
            </div>
            <div className="footer-text">
              Chat FJ puede cometer errores. Verifica la información importante.
            </div>
          </div>
        </div>
      </div>

      <ReferenceModal
        reference={showReferenceModal ? selectedReference : null}
        onClose={() => setShowReferenceModal(false)}
      />


      {/* Panel de administración */}
      {showAdminPanel && (
        <AdminPanel onClose={() => setShowAdminPanel(false)} />
      )}
    </div>
  );
}

export default App;
