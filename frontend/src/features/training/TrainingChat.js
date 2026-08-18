import React, { useState, useRef, useEffect } from 'react';
import {
  X, Send, Check, Pencil, Upload, Loader2,
  GraduationCap, MessageSquare, Clock, BookOpen
} from 'lucide-react';
import './training.css';
import apiService from '../../api/client';

function TrainingChat({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [trainerName, setTrainerName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackState, setFeedbackState] = useState(null);
  // feedbackState: null | 'waiting' | 'correcting'
  const [correctionText, setCorrectionText] = useState('');
  const [pendingFeedback, setPendingFeedback] = useState(null);
  const [stats, setStats] = useState({ approved: 0, corrected: 0 });
  const [submitting, setSubmitting] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [typingMessage, setTypingMessage] = useState('');

  const messagesEndRef = useRef(null);
  const correctionRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, feedbackState, typingMessage, isTyping]);

  useEffect(() => {
    if (feedbackState === 'correcting' && correctionRef.current) {
      correctionRef.current.focus();
      // Position cursor at end
      const len = correctionRef.current.value.length;
      correctionRef.current.setSelectionRange(len, len);
    }
  }, [feedbackState]);

  // === HELPERS ===
  const parseMarkdownLine = (text) => {
    if (!text || !text.trim()) return text;
    // Bold: **text** or __text__
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic: *text* or _text_
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.+?)_/g, '<em>$1</em>');
    return text;
  };

  const renderMessageContent = (content) => {
    if (!content) return null;
    const lines = content.split('\n');
    return lines.map((line, idx) => {
      // Headings (## or ###)
      const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
      if (headingMatch) {
         return (
           <React.Fragment key={idx}>
             <span className="tc-markdown-heading" dangerouslySetInnerHTML={{ __html: parseMarkdownLine(headingMatch[2]) }} />
             {idx < lines.length - 1 && <br />}
           </React.Fragment>
         );
      }
      // Lists (- or *)
      const listMatch = line.match(/^[-*]\s+(.+)$/);
      if (listMatch) {
        return (
          <React.Fragment key={idx}>
            <span className="tc-markdown-list-item">
              • <span dangerouslySetInnerHTML={{ __html: parseMarkdownLine(listMatch[1]) }} />
            </span>
            {idx < lines.length - 1 && <br />}
          </React.Fragment>
        );
      }
      // Check for separator ---
      if (line.trim() === '---') {
        return <hr key={idx} style={{ border: 0, borderTop: '1px solid #334155', margin: '12px 0' }} />;
      }

      // Normal text
      return (
        <React.Fragment key={idx}>
          <span dangerouslySetInnerHTML={{ __html: parseMarkdownLine(line) }} />
          {idx < lines.length - 1 && <br />}
        </React.Fragment>
      );
    });
  };

  const typeMessage = (fullMessage, data) => {
    setIsTyping(true);
    setTypingMessage('');
    
    // Split by lines to preserve structure during typing
    const lines = fullMessage.split('\n');
    let currentText = '';
    let currentLineIndex = 0;
    let currentWordIndex = 0;

    const typeInterval = setInterval(() => {
      if (currentLineIndex < lines.length) {
        const currentLine = lines[currentLineIndex];
        const words = currentLine.split(' '); // Split by space to type word by word

        if (currentWordIndex < words.length) {
          currentText += (currentWordIndex > 0 ? ' ' : '') + words[currentWordIndex];
          currentWordIndex++;
          setTypingMessage(currentText);
        } else {
           // End of line
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
        setTypingMessage('');
        
        // Add final message
        const aiMsg = {
          role: 'assistant',
          content: fullMessage,
          sources: data.sources || [],
          processing_time: data.processing_time || 0,
          learned_from_feedback: data.learned_from_feedback || false
        };
        setMessages(prev => [...prev, aiMsg]);
        setFeedbackState('waiting');
        setPendingFeedback({ question: data.original_question, answer: fullMessage });
      }
    }, 15); // Speed
  };

  // === SEND QUESTION ===
  const sendQuestion = async (e) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input.trim();
    const userMsg = { role: 'user', content: question };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const data = await apiService.ask(question, []);
      // Pass original question for feedback linkage
      data.original_question = question;

      setIsLoading(false);
      // Start typing animation
      typeMessage(data.answer, data);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error al procesar la pregunta. Por favor intente de nuevo.',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // === APPROVE RESPONSE ===
  const approveResponse = async () => {
    if (!pendingFeedback) return;
    setSubmitting(true);

    try {
      await apiService.submitFeedback({
        items: [{
          selected_text: '',
          feedback: 'Respuesta aprobada por entrenador',
          original_question: pendingFeedback.question,
          full_response: pendingFeedback.answer,
          intent: 'approval',
          trainer_name: trainerName || 'anon'
        }]
      });

      setMessages(prev => [...prev, {
        role: 'system',
        content: 'Respuesta aprobada. El sistema seguirá respondiendo de forma similar.',
        type: 'approved'
      }]);

      setStats(prev => ({ ...prev, approved: prev.approved + 1 }));
      resetFeedback();

    } catch (error) {
      console.error('Error aprobando:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: error.message || 'No se pudo registrar la aprobación.',
        type: 'error'
      }]);
    } finally {
      setSubmitting(false);
    }
  };

  // === START CORRECTION ===
  const startCorrection = () => {
    if (!pendingFeedback) return;
    setCorrectionText(pendingFeedback.answer); // PRE-LOAD the AI response
    setFeedbackState('correcting');
  };

  // === SUBMIT CORRECTION ===
  const submitCorrection = async () => {
    if (!correctionText.trim() || !pendingFeedback) return;
    if (correctionText.trim() === pendingFeedback.answer.trim()) {
      // No changes made
      return;
    }
    setSubmitting(true);

    try {
      // El servicio de API adjunta un token fresco y traduce los errores del
      // servidor (401 por sesión vencida, 403 por falta de permisos).
      const data = await apiService.submitFeedback({
        items: [{
          selected_text: pendingFeedback.answer,
          feedback: correctionText,
          original_question: pendingFeedback.question,
          full_response: pendingFeedback.answer,
          intent: 'correction',
          trainer_name: trainerName || 'anon'
        }]
      });

      setMessages(prev => [...prev, {
        role: 'system',
        content: `Corrección guardada (${data.learned_items || 1} aprendida). Se usará en futuras consultas similares.`,
        type: 'corrected'
      }]);

      setStats(prev => ({ ...prev, corrected: prev.corrected + 1 }));
      resetFeedback();

    } catch (error) {
      console.error('Error enviando corrección:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: error.message || 'No se pudo guardar la corrección. Intente de nuevo.',
        type: 'error'
      }]);
    } finally {
      setSubmitting(false);
    }
  };

  const resetFeedback = () => {
    setFeedbackState(null);
    setPendingFeedback(null);
    setCorrectionText('');
  };

  const cancelCorrection = () => {
    setFeedbackState('waiting');
    setCorrectionText('');
  };

  return (
    <div className="tc-overlay">
      <div className="tc-container">

        {/* Header */}
        <div className="tc-header">
          <div className="tc-header-left">
            <GraduationCap size={24} />
            <div>
              <h1>Panel de Entrenamiento</h1>
              <p>Revise y corrija las respuestas de la IA</p>
            </div>
          </div>
          <div className="tc-header-right">
            <div className="tc-stats">
              <span className="tc-stat approved">
                <Check size={14} />
                {stats.approved}
              </span>
              <span className="tc-stat corrected">
                <Pencil size={14} />
                {stats.corrected}
              </span>
            </div>
            <button className="tc-close" onClick={onClose} aria-label="Cerrar">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Trainer Name */}
        <div className="tc-trainer-bar">
          <label htmlFor="trainer-name">Nombre del entrenador:</label>
          <input
            id="trainer-name"
            type="text"
            value={trainerName}
            onChange={(e) => setTrainerName(e.target.value)}
            placeholder="Ej: Lic. María Rodríguez"
          />
        </div>

        {/* Messages Area */}
        <div className="tc-messages">
          {messages.length === 0 && (
            <div className="tc-welcome">
              <MessageSquare size={48} strokeWidth={1.5} />
              <h2>Modo Entrenamiento Activo</h2>
              <p>Haga una pregunta legal y luego podrá:</p>
              <ul>
                <li>
                  <Check size={16} />
                  <span><strong>Aprobar</strong> si la respuesta es correcta</span>
                </li>
                <li>
                  <Pencil size={16} />
                  <span><strong>Corregir</strong> editando la respuesta directamente</span>
                </li>
              </ul>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`tc-msg tc-msg-${msg.role} ${msg.type ? `tc-msg-${msg.type}` : ''} ${msg.error ? 'tc-msg-error' : ''}`}>
              {msg.role === 'user' && (
                <div className="tc-bubble tc-bubble-user">
                  {msg.content}
                </div>
              )}

              {msg.role === 'assistant' && (
                <div className="tc-bubble tc-bubble-ai">
                  <div className="tc-bubble-header">
                    <span className="tc-ai-badge">IA</span>
                  </div>
                  <div className="tc-bubble-content">
                    {renderMessageContent(msg.content)}
                  </div>
                  <div className="tc-sources">
                    <Clock size={12} />
                    <span>{Number(msg.processing_time || 0).toFixed(2)}s</span>
                    {msg.learned_from_feedback && (
                      <span className="tc-correction-badge">
                        <GraduationCap size={12} />
                        Respuesta entrenada
                      </span>
                    )}
                    {msg.sources && msg.sources.length > 0 && (
                      <>
                        <BookOpen size={12} />
                        {msg.sources.length} fuente{msg.sources.length !== 1 ? 's' : ''}
                      </>
                    )}
                  </div>
                </div>
              )}

              {msg.role === 'system' && (
                <div className={`tc-system-msg tc-system-${msg.type || 'info'}`}>
                  {msg.type === 'approved' && <Check size={16} />}
                  {msg.type === 'corrected' && <Pencil size={16} />}
                  {msg.content}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="tc-msg tc-msg-assistant">
              <div className="tc-bubble tc-bubble-ai tc-loading">
                <Loader2 size={18} className="tc-spinner" />
                Procesando...
              </div>
            </div>
          )}

          {isTyping && (
             <div className="tc-msg tc-msg-assistant">
               <div className="tc-bubble tc-bubble-ai">
                 <div className="tc-bubble-header">
                    <span className="tc-ai-badge">IA</span>
                 </div>
                 <div className="tc-bubble-content">
                   {renderMessageContent(typingMessage)}
                   <span className="tc-typing-cursor"></span>
                 </div>
               </div>
             </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="tc-input-area">

          {/* FEEDBACK: Approve / Correct buttons */}
          {feedbackState === 'waiting' && (
            <div className="tc-feedback-bar">
              <span className="tc-feedback-label">¿La respuesta es correcta?</span>
              <div className="tc-feedback-buttons">
                <button
                  className="tc-btn tc-btn-approve"
                  onClick={approveResponse}
                  disabled={submitting}
                >
                  <Check size={18} />
                  Aprobar Respuesta
                </button>
                <button
                  className="tc-btn tc-btn-correct"
                  onClick={startCorrection}
                  disabled={submitting}
                >
                  <Pencil size={18} />
                  Corregir Respuesta
                </button>
              </div>
            </div>
          )}

          {/* CORRECTION MODE: Pre-loaded textarea */}
          {feedbackState === 'correcting' && (
            <div className="tc-correction-panel">
              <div className="tc-correction-header">
                <Pencil size={16} />
                <span>Editando respuesta — modifique lo que necesite corregir</span>
              </div>
              <textarea
                ref={correctionRef}
                className="tc-correction-textarea"
                value={correctionText}
                onChange={(e) => setCorrectionText(e.target.value)}
                rows={6}
              />
              <div className="tc-correction-actions">
                <button
                  className="tc-btn tc-btn-cancel"
                  onClick={cancelCorrection}
                  disabled={submitting}
                >
                  Cancelar
                </button>
                <button
                  className="tc-btn tc-btn-submit"
                  onClick={submitCorrection}
                  disabled={submitting || correctionText.trim() === (pendingFeedback?.answer || '').trim()}
                >
                  {submitting ? (
                    <><Loader2 size={16} className="tc-spinner" /> Guardando...</>
                  ) : (
                    <><Upload size={16} /> Subir Corrección</>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* NORMAL INPUT: Question form */}
          {!feedbackState && (
            <form className="tc-question-form" onSubmit={sendQuestion}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Escriba una pregunta legal para evaluar..."
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="tc-btn tc-btn-send"
              >
                {isLoading ? <Loader2 size={18} className="tc-spinner" /> : <Send size={18} />}
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrainingChat;
