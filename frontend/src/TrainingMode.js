import React, { useState, useEffect } from 'react';
import './TrainingMode.css';

function TrainingMode({ onClose }) {
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState(null);
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [feedbackData, setFeedbackData] = useState({
    status: 'pending',
    evaluatorNotes: '',
    feedbackItems: []
  });
  const [statistics, setStatistics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
            </div>

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
