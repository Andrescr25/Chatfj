import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([
    { id: 1, title: 'Nueva conversación', timestamp: new Date(), messages: [] }
  ]);
  const [currentConvId, setCurrentConvId] = useState(1);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
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
      
      const assistantMessage = { role: 'assistant', content: data.answer };
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
                    {msg.content.split('\n').map((line, i) => (
                      <React.Fragment key={i}>
                        {line}
                        {i < msg.content.split('\n').length - 1 && <br />}
                      </React.Fragment>
                    ))}
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
    </div>
  );
}

export default App;
