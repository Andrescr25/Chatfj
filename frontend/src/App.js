import React, { useState, useEffect, useRef } from 'react';
import {
  Menu, X, MessageSquarePlus, AlertTriangle, MessageSquare, Clock, Trash2,
  Scale, CircleDollarSign, Briefcase, User, ExternalLink,
  Sun, Moon, Send, Loader2, FileText, Bot, Lock, Settings
} from 'lucide-react';
import './App.css';
import AdminPanel from './AdminPanel';
import { signInWithEmailAndPassword, signOut, onAuthStateChanged } from 'firebase/auth';
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
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    // Cargar tema desde localStorage o usar 'light' por defecto
    return localStorage.getItem('theme') || 'light';
  });
  const messagesEndRef = useRef(null);

  // Configuración de API URL
  // Para desarrollo local: usa proxy (ruta vacía "")
  // Para producción/Ngrok: usa http://localhost:8000 directo
  const API_URL = process.env.REACT_APP_API_URL || '';

  const currentConv = conversations.find(c => c.id === currentConvId);

  // Estados de autenticación de súper usuario
  const [isAdminRoute, setIsAdminRoute] = useState(false);
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(
    () => !!localStorage.getItem('adminToken')
  );
  const [adminEmailInput, setAdminEmailInput] = useState('');
  const [adminPasswordInput, setAdminPasswordInput] = useState('');
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

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

  const handleAdminLogin = async (e) => {
    e.preventDefault();
    if (!adminEmailInput.trim() || !adminPasswordInput.trim() || isLoggingIn) return;
    setIsLoggingIn(true);
    setLoginError('');
    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        adminEmailInput.trim(),
        adminPasswordInput.trim()
      );
      const idToken = await userCredential.user.getIdToken();
      localStorage.setItem('adminToken', idToken);
      setIsAdminAuthenticated(true);
      setAdminEmailInput('');
      setAdminPasswordInput('');
    } catch (error) {
      console.error(error);
      let errorMsg = 'Error al iniciar sesión. Verifique sus credenciales.';
      if (
        error.code === 'auth/user-not-found' || 
        error.code === 'auth/wrong-password' || 
        error.code === 'auth/invalid-credential'
      ) {
        errorMsg = 'Correo electrónico o contraseña incorrectos.';
      } else if (error.code === 'auth/invalid-email') {
        errorMsg = 'Formato de correo electrónico inválido.';
      }
      setLoginError(errorMsg);
    } finally {
      setIsLoggingIn(false);
    }
  };

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
          correctionUsageId: metadata?.correction_usage_id || null,
          matchedQuestion: metadata?.matched_question || '',
          learnedFromFeedback: metadata?.learned_from_feedback || false,
          feedbackStatus: null
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

  const handleCorrectionFeedback = async (convId, messageIndex, usageId, result) => {
    if (!usageId) return;
    setFeedbackSubmitting(usageId);
    try {
      await fetch(`${API_URL}/training/correction-usage/${usageId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ result, source: 'explicit' })
      });

      setConversations(prevConvs =>
        prevConvs.map(conv => {
          if (conv.id !== convId) return conv;
          const updatedMessages = conv.messages.map((msg, idx) =>
            idx === messageIndex ? { ...msg, feedbackStatus: result } : msg
          );
          return { ...conv, messages: updatedMessages };
        })
      );
    } catch (error) {
      console.error('Error enviando feedback:', error);
      alert('No se pudo guardar tu feedback. Intenta de nuevo.');
    } finally {
      setFeedbackSubmitting(null);
    }
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

          <form onSubmit={handleAdminLogin} style={{ width: '100%' }}>
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
                value={adminEmailInput}
                onChange={(e) => setAdminEmailInput(e.target.value)}
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
                value={adminPasswordInput}
                onChange={(e) => setAdminPasswordInput(e.target.value)}
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

            {loginError && (
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
                ⚠️ {loginError}
              </div>
            )}

            <button
              type="submit"
              disabled={isLoggingIn}
              style={{
                width: '100%',
                padding: '14px',
                background: '#1d4ed8',
                color: '#ffffff',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                fontSize: '15px',
                cursor: isLoggingIn ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 14px 0 rgba(29, 78, 216, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'background-color 0.2s'
              }}
            >
              {isLoggingIn ? (
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

  return (
    <div className="app">
      {/* Mobile Menu Toggle Button */}
      <button 
        className="mobile-menu-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle menu"
      >
        {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
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
          <MessageSquarePlus size={16} /> Nueva conversación
        </button>
        
        {currentConv.messages.length === 0 && (
          <p className="warning"><AlertTriangle size={12} style={{marginRight: '4px', verticalAlign: 'text-bottom'}}/> Escribe algo primero</p>
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
                  <MessageSquare size={14} style={{marginTop: '2px'}}/> {conv.title}
                </div>
                <div className="conversation-date">
                  <Clock size={12} /> {formatTimestamp(conv.timestamp)}
                </div>
              </button>
              <button
                className="delete-btn"
                onClick={() => deleteConversation(conv.id)}
                disabled={conversations.length === 1}
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
                <Sun size={14} />
              </span>
              <span className="theme-icon moon">
                <Moon size={14} />
              </span>
              <div className="theme-toggle-slider"></div>
            </div>
          </div>
          
          {isAdminAuthenticated && (
            <>
              <button
                className="training-mode-btn"
                onClick={() => {
                  setShowAdminPanel(true);
                  setSidebarOpen(false);
                }}
                title="Panel de administración"
              >
                <Settings size={16} /> Panel de administración
              </button>
              <button
                className="admin-logout-btn"
                onClick={handleAdminLogout}
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

      {/* Main Content */}
      <div className="main-content">
        <div className="header">
          <h1><Scale size={20} style={{verticalAlign: 'text-bottom', marginRight: '8px'}}/> Chat FJ - Servicio Nacional de Facilitadoras y Facilitadores Judiciales</h1>
          
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
                <Sun size={14} />
              </span>
              <span className="theme-icon moon">
                <Moon size={14} />
              </span>
              <div className="theme-toggle-slider"></div>
            </div>
          </div>
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
                    {msg.role === 'assistant' && msg.correctionUsageId && (
                      <div className="message-feedback">
                        {msg.feedbackStatus ? (
                          <span className={`feedback-confirmed ${msg.feedbackStatus}`}>
                            {msg.feedbackStatus === 'success'
                              ? '✅ ¡Gracias! Aprenderé de tu confirmación.'
                              : '⚠️ Gracias por avisar, ajustaré esta corrección.'}
                          </span>
                        ) : (
                          <>
                            <span className="feedback-question">
                              ¿Esta corrección aprendida te ayudó?
                            </span>
                            <div className="feedback-buttons">
                              <button
                                className="feedback-button success"
                                disabled={feedbackSubmitting === msg.correctionUsageId}
                                onClick={() =>
                                  handleCorrectionFeedback(
                                    currentConvId,
                                    idx,
                                    msg.correctionUsageId,
                                    'success'
                                  )
                                }
                              >
                                Sí, me ayudó
                              </button>
                              <button
                                className="feedback-button fail"
                                disabled={feedbackSubmitting === msg.correctionUsageId}
                                onClick={() =>
                                  handleCorrectionFeedback(
                                    currentConvId,
                                    idx,
                                    msg.correctionUsageId,
                                    'fail'
                                  )
                                }
                              >
                                No, faltó precisión
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    )}
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
                {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
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
              <h3><FileText size={20} style={{verticalAlign: 'bottom', marginRight: '8px'}}/> {selectedReference.filename || 'Fuente'}</h3>
              <button
                className="modal-close-btn"
                onClick={() => setShowReferenceModal(false)}
              >
                <X size={20} />
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
                    <ExternalLink size={16} style={{marginRight: '6px'}}/> Abrir sitio web
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

      {/* Panel de administración */}
      {showAdminPanel && (
        <AdminPanel onClose={() => setShowAdminPanel(false)} />
      )}
    </div>
  );
}

export default App;
