import React, { useState, useEffect } from 'react';
import './TrainingMode.css';
import DocumentUpload from './DocumentUpload';

function TrainingMode({ onClose }) {
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState(null);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackData, setFeedbackData] = useState({
    status: 'pending',
    evaluatorNotes: '',
    feedbackItems: []
  });
  const [correctedAnswer, setCorrectedAnswer] = useState('');
  const [showCorrectionBox, setShowCorrectionBox] = useState(false);
  const [statistics, setStatistics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Usar rutas relativas para que funcione con el proxy y con ngrok
  const API_URL = process.env.REACT_APP_API_URL || '';

  // Cargar estadísticas al iniciar
  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      const response = await fetch(`${API_URL}/training/statistics`);
      const data = await response.json();
      if (data.success) {
        setStatistics(data.statistics);
      }
    } catch (error) {
      console.error('Error cargando estadísticas:', error);
    }
  };

  const testQuestion = async () => {
    if (!currentQuestion.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentQuestion,
          history: []
        })
      });

      const data = await response.json();
      setCurrentAnswer({
        question: currentQuestion,
        answer: data.answer,
        sources: data.sources || [],
        processing_time: data.processing_time || 0,
        category_detected: 'general' // Esto se puede mejorar agregando categoría en la respuesta
      });
      setShowFeedbackForm(true);
    } catch (error) {
      console.error('Error probando pregunta:', error);
      alert('Error al procesar la pregunta');
    } finally {
      setIsLoading(false);
    }
  };

  const submitFeedback = async (status) => {
    if (!currentAnswer) return;

    try {
      const response = await fetch(`${API_URL}/training/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentAnswer.question,
          answer: currentAnswer.answer,
          sources: currentAnswer.sources,
          category_detected: currentAnswer.category_detected,
          processing_time: currentAnswer.processing_time,
          status: status,
          evaluator_notes: feedbackData.evaluatorNotes,
          feedback_items: feedbackData.feedbackItems
        })
      });

      const data = await response.json();
      if (data.success) {
        alert(`✅ Feedback guardado: ${status === 'approved' ? 'APROBADO' : 'RECHAZADO'}`);

        // Resetear formulario
        setCurrentQuestion('');
        setCurrentAnswer(null);
        setShowFeedbackForm(false);
        setFeedbackData({
          status: 'pending',
          evaluatorNotes: '',
          feedbackItems: []
        });

        // Recargar estadísticas
        loadStatistics();
      }
    } catch (error) {
      console.error('Error guardando feedback:', error);
      alert('Error al guardar feedback');
    }
  };

  const addFeedbackItem = () => {
    setFeedbackData({
      ...feedbackData,
      feedbackItems: [
        ...feedbackData.feedbackItems,
        {
          feedback_type: 'content_error',
          field: 'general',
          issue: '',
          correct_value: '',
          severity: 'medium'
        }
      ]
    });
  };

  const updateFeedbackItem = (index, field, value) => {
    const updatedItems = [...feedbackData.feedbackItems];
    updatedItems[index][field] = value;
    setFeedbackData({
      ...feedbackData,
      feedbackItems: updatedItems
    });
  };

  const removeFeedbackItem = (index) => {
    setFeedbackData({
      ...feedbackData,
      feedbackItems: feedbackData.feedbackItems.filter((_, i) => i !== index)
    });
  };

  const submitCorrection = async () => {
    if (!currentAnswer || !correctedAnswer.trim()) {
      alert('Por favor escribe la respuesta corregida');
      return;
    }

    try {
      // Determinar tipo de corrección basado en feedback items
      let correction_type = 'content';
      if (feedbackData.feedbackItems.length > 0) {
        const firstItem = feedbackData.feedbackItems[0];
        if (firstItem.feedback_type === 'citation_error') correction_type = 'citation';
        else if (firstItem.feedback_type === 'category_error') correction_type = 'category';
        else if (firstItem.feedback_type === 'format_error') correction_type = 'format';
      }

      const response = await fetch(`${API_URL}/training/learn-correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: currentAnswer.question,
          original_answer: currentAnswer.answer,
          corrected_answer: correctedAnswer,
          correction_type: correction_type,
          category: currentAnswer.category_detected
        })
      });

      const data = await response.json();
      if (data.success) {
        alert(`🎓 ¡Corrección guardada! El sistema usará esta respuesta en futuras consultas idénticas.\n\nID: ${data.correction_id}`);

        // También guardar como feedback "corrected"
        await submitFeedback('corrected');

        // Resetear
        setCorrectedAnswer('');
        setShowCorrectionBox(false);
      }
    } catch (error) {
      console.error('Error guardando corrección:', error);
      alert('Error al guardar corrección');
    }
  };

  const exportData = async () => {
    try {
      const response = await fetch(`${API_URL}/training/export?status=approved`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.success) {
        alert(`✅ Exportados ${data.records_exported} registros a ${data.file_path}`);
      }
    } catch (error) {
      console.error('Error exportando datos:', error);
      alert('Error al exportar datos');
    }
  };

  return (
    <div className="training-mode-overlay">
      <div className="training-mode-container">
        <div className="training-header">
          <h1>🎓 Modo Entrenamiento</h1>
          <button className="close-button" onClick={onClose}>✕</button>
        </div>

        {/* Estadísticas */}
        {statistics && (
          <div className="training-stats">
            <div className="stat-card">
              <div className="stat-value">{statistics.total_evaluations}</div>
              <div className="stat-label">Total Evaluaciones</div>
            </div>
            <div className="stat-card success">
              <div className="stat-value">{statistics.approved}</div>
              <div className="stat-label">Aprobadas</div>
            </div>
            <div className="stat-card danger">
              <div className="stat-value">{statistics.rejected}</div>
              <div className="stat-label">Rechazadas</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{statistics.approval_rate.toFixed(1)}%</div>
              <div className="stat-label">Tasa de Aprobación</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{statistics.avg_processing_time.toFixed(2)}s</div>
              <div className="stat-label">Tiempo Promedio</div>
            </div>
          </div>
        )}

        {/* Área de prueba */}
        <div className="training-test-area">
          <h2>Probar Pregunta</h2>
          <div className="input-group">
            <textarea
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              placeholder="Escribe una pregunta de prueba..."
              rows="3"
              disabled={isLoading}
            />
            <button
              onClick={testQuestion}
              disabled={isLoading || !currentQuestion.trim()}
              className="btn-primary"
            >
              {isLoading ? 'Procesando...' : 'Probar'}
            </button>
          </div>
        </div>

        {/* Respuesta y Feedback */}
        {showFeedbackForm && currentAnswer && (
          <div className="training-feedback-section">
            <h2>Respuesta Generada</h2>

            <div className="answer-preview">
              <div className="answer-metadata">
                <span>⏱️ {currentAnswer.processing_time.toFixed(2)}s</span>
                <span>📚 {currentAnswer.sources.length} fuentes</span>
              </div>
              <div className="answer-content">
                {currentAnswer.answer}
              </div>
            </div>

            {/* Botones de evaluación rápida */}
            <div className="evaluation-buttons">
              <button
                className="btn-approve"
                onClick={() => submitFeedback('approved')}
              >
                ✓ Aprobar
              </button>
              <button
                className="btn-reject"
                onClick={() => submitFeedback('rejected')}
              >
                ✗ Rechazar
              </button>
              <button
                className="btn-correct"
                onClick={() => setShowCorrectionBox(!showCorrectionBox)}
                style={{ backgroundColor: '#ff9800', color: 'white' }}
              >
                ✏️ Corregir y Aprender
              </button>
            </div>

            {/* Área de corrección (NUEVO - Aprendizaje en tiempo real) */}
            {showCorrectionBox && (
              <div className="correction-box" style={{
                marginTop: '20px',
                padding: '20px',
                border: '2px solid #ff9800',
                borderRadius: '8px',
                backgroundColor: '#fff8e1'
              }}>
                <h3 style={{ color: '#f57c00', marginTop: 0 }}>🎓 Corrección para Aprendizaje en Tiempo Real</h3>
                <p style={{ fontSize: '14px', color: '#666', marginBottom: '15px' }}>
                  Escribe la respuesta corregida completa. El sistema la guardará y la usará automáticamente
                  la próxima vez que reciba esta misma pregunta.
                </p>
                <textarea
                  value={correctedAnswer}
                  onChange={(e) => setCorrectedAnswer(e.target.value)}
                  placeholder="Escribe aquí la respuesta corregida completa..."
                  rows="8"
                  style={{
                    width: '100%',
                    padding: '12px',
                    borderRadius: '4px',
                    border: '1px solid #ff9800',
                    fontSize: '14px',
                    fontFamily: 'inherit'
                  }}
                />
                <div style={{ marginTop: '15px', display: 'flex', gap: '10px' }}>
                  <button
                    onClick={submitCorrection}
                    disabled={!correctedAnswer.trim()}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: correctedAnswer.trim() ? 'pointer' : 'not-allowed',
                      fontWeight: 'bold',
                      opacity: correctedAnswer.trim() ? 1 : 0.5
                    }}
                  >
                    💾 Guardar Corrección y Aprender
                  </button>
                  <button
                    onClick={() => {
                      setCorrectedAnswer('');
                      setShowCorrectionBox(false);
                    }}
                    style={{
                      padding: '10px 20px',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    ✕ Cancelar
                  </button>
                </div>
              </div>
            )}

            {/* Notas del evaluador */}
            <div className="evaluator-notes">
              <h3>Notas del Evaluador</h3>
              <textarea
                value={feedbackData.evaluatorNotes}
                onChange={(e) => setFeedbackData({...feedbackData, evaluatorNotes: e.target.value})}
                placeholder="Agrega comentarios generales sobre esta respuesta..."
                rows="3"
              />
            </div>

            {/* Feedback detallado */}
            <div className="detailed-feedback">
              <div className="feedback-header">
                <h3>Feedback Detallado</h3>
                <button onClick={addFeedbackItem} className="btn-add">
                  + Agregar Feedback
                </button>
              </div>

              {feedbackData.feedbackItems.map((item, index) => (
                <div key={index} className="feedback-item">
                  <div className="feedback-item-header">
                    <select
                      value={item.feedback_type}
                      onChange={(e) => updateFeedbackItem(index, 'feedback_type', e.target.value)}
                    >
                      <option value="citation_error">Error de Cita Legal</option>
                      <option value="category_error">Error de Categoría</option>
                      <option value="format_error">Error de Formato</option>
                      <option value="content_error">Error de Contenido</option>
                      <option value="suggestion">Sugerencia</option>
                    </select>

                    <select
                      value={item.severity}
                      onChange={(e) => updateFeedbackItem(index, 'severity', e.target.value)}
                      className={`severity-${item.severity}`}
                    >
                      <option value="low">Baja</option>
                      <option value="medium">Media</option>
                      <option value="high">Alta</option>
                      <option value="critical">Crítica</option>
                    </select>

                    <button
                      onClick={() => removeFeedbackItem(index)}
                      className="btn-remove"
                    >
                      ✕
                    </button>
                  </div>

                  <input
                    type="text"
                    value={item.field}
                    onChange={(e) => updateFeedbackItem(index, 'field', e.target.value)}
                    placeholder="Campo afectado (ej: legal_citation, institution)"
                  />

                  <textarea
                    value={item.issue}
                    onChange={(e) => updateFeedbackItem(index, 'issue', e.target.value)}
                    placeholder="Descripción del problema..."
                    rows="2"
                  />

                  <input
                    type="text"
                    value={item.correct_value}
                    onChange={(e) => updateFeedbackItem(index, 'correct_value', e.target.value)}
                    placeholder="Valor correcto (opcional)"
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Componente de subida de documentos */}
        <DocumentUpload API_URL={API_URL} />

        {/* Acciones adicionales */}
        <div className="training-actions">
          <button onClick={exportData} className="btn-export">
            📤 Exportar Datos Aprobados
          </button>
          <button onClick={loadStatistics} className="btn-refresh">
            🔄 Actualizar Estadísticas
          </button>
        </div>
      </div>
    </div>
  );
}

export default TrainingMode;
