import React, { useState, useRef, useEffect } from 'react';
import './TrainingChat.css';

function TrainingChat({ onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [correctionInput, setCorrectionInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [waitingForFeedback, setWaitingForFeedback] = useState(null);
  const [showCorrectionForm, setShowCorrectionForm] = useState(false);
  const [statistics, setStatistics] = useState({ approved: 0, corrected: 0 });
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' o 'documents'
  const [documents, setDocuments] = useState([]);
  const [documentStats, setDocumentStats] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const API_URL = process.env.REACT_APP_API_URL || '';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cargar estadísticas de documentos cuando se abre el tab
  useEffect(() => {
    if (activeTab === 'documents') {
      loadDocumentStats();
    }
  }, [activeTab]);

  const loadDocumentStats = async () => {
    try {
      const response = await fetch(`${API_URL}/training/document-stats`);
      const data = await response.json();
      if (data.success) {
        setDocumentStats(data);
        setDocuments(data.uploaded_files || []);
      }
    } catch (error) {
      console.error('Error cargando estadísticas de documentos:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', 'general');

    setUploading(true);

    try {
      const response = await fetch(`${API_URL}/training/upload-document`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        alert(`✅ Documento subido: ${data.filename}\n${data.chunks_added} fragmentos procesados`);
        loadDocumentStats(); // Recargar lista
      } else {
        alert('❌ Error al subir documento');
      }
    } catch (error) {
      console.error('Error subiendo documento:', error);
      alert('❌ Error al subir documento');
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const viewDocumentChunks = async (filename) => {
    try {
      const response = await fetch(`${API_URL}/training/document-content/${encodeURIComponent(filename)}`);
      const data = await response.json();
      
      if (data.success) {
        setSelectedDocument({
          filename: data.filename,
          chunks: data.chunks,
          totalChunks: data.total_chunks,
          category: data.category,
          uploadDate: data.upload_date
        });
      } else {
        alert('❌ No se pudo cargar el contenido del documento');
      }
    } catch (error) {
      console.error('Error obteniendo chunks del documento:', error);
      alert('❌ Error al cargar el documento');
    }
  };

  const sendQuestion = async (e) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          history: []
        })
      });

      const data = await response.json();

      const aiMessage = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        processing_time: data.processing_time || 0,
        correctionUsageId: data.correction_usage_id || null
      };

      setMessages(prev => [...prev, aiMessage]);

      // Agregar pregunta de feedback automáticamente
      const feedbackPrompt = {
        role: 'system',
        content: '¿Esta respuesta está bien o hay algo que pueda mejorar?'
      };

      setMessages(prev => [...prev, feedbackPrompt]);
      setWaitingForFeedback({
        question: input,
        originalAnswer: data.answer,
        sources: data.sources || [],
        processing_time: data.processing_time || 0,
        messageIndex: messages.length + 1,
        correctionUsageId: data.correction_usage_id || null
      });

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error al procesar la pregunta. Por favor intenta de nuevo.',
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const approveAnswer = async () => {
    if (!waitingForFeedback) return;

    try {
      // Guardar como aprobada
      await fetch(`${API_URL}/training/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: waitingForFeedback.question,
          answer: waitingForFeedback.originalAnswer,
          sources: waitingForFeedback.sources,
          category_detected: 'general',
          processing_time: waitingForFeedback.processing_time,
          status: 'approved',
          evaluator_notes: 'Usuario aprobó la respuesta en modo entrenamiento',
          feedback_items: []
        })
      });

      if (waitingForFeedback.correctionUsageId) {
        await fetch(`${API_URL}/training/correction-usage/${waitingForFeedback.correctionUsageId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ result: 'success', source: 'training-approve' })
        });
      }

      setMessages(prev => [...prev, {
        role: 'system',
        content: '✅ ¡Perfecto! Respuesta aprobada. Seguiré dando respuestas similares.',
        approved: true
      }]);

      setStatistics(prev => ({ ...prev, approved: prev.approved + 1 }));
      setWaitingForFeedback(null);
      setShowCorrectionForm(false);
      setCorrectionInput('');

    } catch (error) {
      console.error('Error:', error);
    }
  };

  const submitCorrection = async (e) => {
    e.preventDefault();
    if (!correctionInput.trim() || !waitingForFeedback) return;

    try {
      // 1. Guardar el feedback (para estadísticas)
      await fetch(`${API_URL}/training/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: waitingForFeedback.question,
          answer: waitingForFeedback.originalAnswer,
          sources: waitingForFeedback.sources,
          category_detected: 'general',
          processing_time: waitingForFeedback.processing_time,
          status: 'corrected',
          evaluator_notes: `Usuario proporcionó versión mejorada: ${correctionInput}`,
          feedback_items: [{
            feedback_type: 'correction',
            field: 'answer',
            issue: 'Usuario proporcionó versión mejorada',
            correct_value: correctionInput,
            severity: 'high'
          }]
        })
      });

      // 2. CRÍTICO: Guardar la corrección en la tabla de aprendizaje (learned_corrections)
      const learnResponse = await fetch(`${API_URL}/training/learn-correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: waitingForFeedback.question,
          original_answer: waitingForFeedback.originalAnswer,
          corrected_answer: correctionInput,
          correction_type: 'content',
          category: 'general'
        })
      });

      const learnData = await learnResponse.json();
      
      setMessages(prev => [...prev, {
        role: 'user',
        content: correctionInput,
        correction: true
      }]);

      setMessages(prev => [...prev, {
        role: 'system',
        content: `✅ ¡Gracias por la corrección! He guardado tu versión mejorada (ID: ${learnData.correction_id}) y la usaré inmediatamente en futuras consultas similares.`,
        corrected: true
      }]);

      setStatistics(prev => ({ ...prev, corrected: prev.corrected + 1 }));
      setCorrectionInput('');
      setWaitingForFeedback(null);
      setShowCorrectionForm(false);

      if (waitingForFeedback.correctionUsageId) {
        await fetch(`${API_URL}/training/correction-usage/${waitingForFeedback.correctionUsageId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ result: 'fail', source: 'training-correction' })
        });
      }

    } catch (error) {
      console.error('Error:', error);
      alert('❌ Error al guardar la corrección. Por favor intenta de nuevo.');
    }
  };

  const skipFeedback = () => {
    setMessages(prev => [...prev, {
      role: 'system',
      content: 'Feedback omitido. Puedes hacer otra pregunta.',
      skipped: true
    }]);
    setWaitingForFeedback(null);
    setShowCorrectionForm(false);
    setCorrectionInput('');
  };

  return (
    <div className="training-chat-overlay">
      <div className="training-chat-container">
        {/* Header */}
        <div className="training-chat-header">
          <div className="header-info">
            <h1>🎓 Chat de Entrenamiento</h1>
            <p>Ayuda a mejorar las respuestas del sistema</p>
          </div>
          <div className="training-stats-mini">
            <span className="stat-mini approved">✅ {statistics.approved}</span>
            <span className="stat-mini corrected">📝 {statistics.corrected}</span>
          </div>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Tabs */}
        <div className="training-tabs">
          <button 
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat de Entrenamiento
          </button>
          <button 
            className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
            onClick={() => setActiveTab('documents')}
          >
            📄 Documentos ({documentStats?.uploaded_files_count || 0})
          </button>
        </div>

        {/* Chat Tab Content */}
        {activeTab === 'chat' && (
          <>
            <div className="training-messages">
          {messages.length === 0 && (
            <div className="welcome-training">
              <div className="welcome-icon">💬</div>
              <h2>Modo Entrenamiento Activo</h2>
              <p>Haz una pregunta y después podrás:</p>
              <ul>
                <li>✅ Aprobar la respuesta si está correcta</li>
                <li>📝 Dar una versión mejorada si encuentras errores</li>
                <li>🎯 Ayudar al sistema a aprender y mejorar</li>
              </ul>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`training-message ${msg.role}-message`}>
              {msg.role === 'system' && (
                <div className={`system-message ${msg.approved ? 'approved' : ''} ${msg.corrected ? 'corrected' : ''} ${msg.skipped ? 'skipped' : ''}`}>
                  {msg.content}
                </div>
              )}
              {msg.role === 'user' && (
                <div className="user-bubble">
                  {msg.correction && <span className="correction-badge">📝 Tu corrección</span>}
                  {msg.content}
                </div>
              )}
              {msg.role === 'assistant' && (
                <div className="assistant-bubble">
                  <div className="bubble-header">
                    <span className="ai-badge">🤖 IA</span>
                    {msg.processing_time && (
                      <span className="processing-time">⏱️ {msg.processing_time.toFixed(2)}s</span>
                    )}
                  </div>
                  <div className="bubble-content">{msg.content}</div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-count">📚 {msg.sources.length} fuentes</div>
                  )}
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="training-message assistant-message">
              <div className="assistant-bubble loading">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                Pensando...
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="training-input-area">
          {waitingForFeedback ? (
            <div className="feedback-zone">
              {!showCorrectionForm ? (
                <>
                  <div className="feedback-prompt">
                    <p>¿Cómo estuvo esta respuesta?</p>
                  </div>

                  <div className="feedback-actions">
                    <button
                      className="btn-approve-answer"
                      onClick={approveAnswer}
                    >
                      ✅ Está bien
                    </button>

                    <button
                      className="btn-show-correction"
                      onClick={() => setShowCorrectionForm(true)}
                    >
                      📝 Dar versión mejorada
                    </button>

                    <button
                      className="btn-skip"
                      onClick={skipFeedback}
                    >
                      ⏭️ Omitir
                    </button>
                  </div>
                </>
              ) : (
                <form className="correction-form" onSubmit={submitCorrection}>
                  <label>Escribe la versión mejorada de la respuesta:</label>
                  <textarea
                    id="correction-input"
                    value={correctionInput}
                    onChange={(e) => setCorrectionInput(e.target.value)}
                    placeholder="Escribe aquí la versión mejorada..."
                    rows="3"
                    autoFocus
                  />
                  <div className="correction-form-actions">
                    <button
                      type="button"
                      className="btn-cancel-correction"
                      onClick={() => {
                        setShowCorrectionForm(false);
                        setCorrectionInput('');
                      }}
                    >
                      Cancelar
                    </button>
                    <button
                      type="submit"
                      className="btn-submit-correction"
                      disabled={!correctionInput.trim()}
                    >
                      Enviar Corrección
                    </button>
                  </div>
                </form>
              )}
            </div>
          ) : (
            <form className="question-form" onSubmit={sendQuestion}>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Haz una pregunta para entrenar al sistema..."
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
              >
                Enviar
              </button>
            </form>
          )}
        </div>
          </>
        )}

        {/* Documents Tab Content */}
        {activeTab === 'documents' && (
          <div className="documents-panel">
            <div className="documents-header">
              <div className="documents-stats">
                <div className="stat-box">
                  <span className="stat-label">Total de Documentos</span>
                  <span className="stat-value">{documentStats?.total_documents || 0}</span>
                </div>
                <div className="stat-box">
                  <span className="stat-label">Archivos Subidos</span>
                  <span className="stat-value">{documentStats?.uploaded_files_count || 0}</span>
                </div>
              </div>
              
              <div className="upload-section">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <button
                  className="btn-upload"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                >
                  {uploading ? '⏳ Subiendo...' : '📤 Subir Documento'}
                </button>
                <p className="upload-hint">Soporta: PDF, TXT, MD</p>
              </div>
            </div>

            <div className="documents-list">
              <h3>📚 Documentos en la Base de Conocimiento</h3>
              
              {documents.length === 0 ? (
                <div className="no-documents">
                  <p>📭 No hay documentos subidos aún</p>
                  <p className="hint">Sube documentos legales para que la IA pueda consultarlos</p>
                </div>
              ) : (
                <div className="documents-grid">
                  {documents.map((doc, idx) => (
                    <div key={idx} className="document-card">
                      <div className="document-icon">📄</div>
                      <div className="document-info">
                        <h4 title={doc.filename}>{doc.display_name || doc.filename}</h4>
                        <p className="document-meta">
                          <span className="meta-item">
                            🧩 {doc.chunks} fragmentos
                          </span>
                          <span className="meta-item">
                            📁 {doc.category}
                          </span>
                        </p>
                        <p className="document-date">
                          📅 {doc.upload_date === 'Base de datos original' ? doc.upload_date : new Date(doc.upload_date).toLocaleDateString('es-CR')}
                        </p>
                      </div>
                      <button
                        className="btn-view-chunks"
                        onClick={() => viewDocumentChunks(doc.filename)}
                      >
                        👁️ Ver fragmentos
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="documents-explanation">
                <h4>🤔 ¿Cómo entiende la IA esta información?</h4>
                <div className="explanation-content">
                  <div className="explanation-step">
                    <span className="step-number">1</span>
                    <div className="step-info">
                      <h5>Fragmentación</h5>
                      <p>Cada documento se divide en fragmentos (chunks) de ~1000 caracteres con overlap de 200 para mantener contexto.</p>
                    </div>
                  </div>
                  <div className="explanation-step">
                    <span className="step-number">2</span>
                    <div className="step-info">
                      <h5>Vectorización</h5>
                      <p>Cada fragmento se convierte en un vector numérico (embedding) usando el modelo {documentStats?.embedding_model || 'sentence-transformers'}.</p>
                    </div>
                  </div>
                  <div className="explanation-step">
                    <span className="step-number">3</span>
                    <div className="step-info">
                      <h5>Búsqueda Semántica</h5>
                      <p>Cuando haces una pregunta, la IA busca los fragmentos más relevantes por similitud semántica (no solo palabras clave).</p>
                    </div>
                  </div>
                  <div className="explanation-step">
                    <span className="step-number">4</span>
                    <div className="step-info">
                      <h5>Generación de Respuesta</h5>
                      <p>La IA usa los fragmentos encontrados como contexto para generar una respuesta precisa y fundamentada.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Modal para ver chunks de un documento */}
        {selectedDocument && (
          <div className="chunks-modal-overlay" onClick={() => setSelectedDocument(null)}>
            <div className="chunks-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div className="modal-title-section">
                  <h3>📄 {selectedDocument.filename}</h3>
                  <div className="modal-meta">
                    <span className="meta-badge">
                      🧩 {selectedDocument.totalChunks} fragmentos
                    </span>
                    <span className="meta-badge">
                      📁 {selectedDocument.category}
                    </span>
                    <span className="meta-badge">
                      📅 {new Date(selectedDocument.uploadDate).toLocaleDateString('es-CR')}
                    </span>
                  </div>
                </div>
                <button
                  className="modal-close-btn"
                  onClick={() => setSelectedDocument(null)}
                >
                  ✕
                </button>
              </div>
              <div className="modal-content">
                <div className="document-review-hint">
                  <p>
                    <strong>💡 Revisión de Contenido:</strong> Este documento se dividió en {selectedDocument.totalChunks} fragmentos.
                    Revisa cada fragmento para verificar que la información sea correcta y esté bien dividida.
                  </p>
                  <p className="hint-note">
                    <strong>⚠️ Puntos a verificar:</strong> Asegúrate de que no haya cortes en medio de oraciones importantes, 
                    que los números de artículos sean correctos, y que no falte contexto crítico.
                  </p>
                </div>

                <div className="chunks-container">
                  {selectedDocument.chunks && selectedDocument.chunks.map((chunk, idx) => (
                    <div key={chunk.id} className="chunk-review-card">
                      <div className="chunk-review-header">
                        <span className="chunk-number">
                          Fragmento {chunk.chunk_index + 1} de {chunk.total_chunks}
                        </span>
                        <span className="chunk-id-badge" title={`ID: ${chunk.id}`}>
                          ID: {chunk.id.substring(0, 8)}...
                        </span>
                      </div>
                      <div className="chunk-review-content">
                        {chunk.content}
                      </div>
                      <div className="chunk-review-footer">
                        <span className="chunk-length">
                          {chunk.content.length} caracteres
                        </span>
                        {chunk.chunk_index < chunk.total_chunks - 1 && (
                          <span className="chunk-overlap-indicator">
                            ⬇️ Overlap con siguiente fragmento
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="document-review-tips">
                  <h4>🎯 Cómo detectar problemas:</h4>
                  <ul>
                    <li><strong>Fragmentación incorrecta:</strong> Si ves que una oración importante se corta a la mitad entre dos fragmentos.</li>
                    <li><strong>Información incompleta:</strong> Si falta contexto necesario para entender el fragmento.</li>
                    <li><strong>Errores en el texto:</strong> Typos, números de artículos incorrectos, o información desactualizada.</li>
                    <li><strong>Formato confuso:</strong> Si el texto se ve desordenado o difícil de entender.</li>
                  </ul>
                  <p className="tip-action">
                    <strong>💡 Si encuentras errores:</strong> Puedes corregirlos usando el <strong>Chat de Entrenamiento</strong>. 
                    Haz una pregunta relacionada y cuando la IA responda, proporciona la versión correcta.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TrainingChat;
