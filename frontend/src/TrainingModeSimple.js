import React, { useState } from 'react';
import './TrainingMode.css';

function TrainingModeSimple({ onClose }) {
  const [selectedIssues, setSelectedIssues] = useState([]);
  const [customFeedback, setCustomFeedback] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || '';

  // Opciones predefinidas más comprensibles para el usuario
  const issueOptions = [
    { id: 'incomplete', label: 'Respuesta incompleta', emoji: '❌', severity: 'medium' },
    { id: 'incorrect', label: 'Información incorrecta', emoji: '⚠️', severity: 'high' },
    { id: 'confusing', label: 'Explicación confusa', emoji: '🤔', severity: 'medium' },
    { id: 'outdated', label: 'Información desactualizada', emoji: '📅', severity: 'high' },
    { id: 'missing_sources', label: 'Falta información de fuentes legales', emoji: '📚', severity: 'medium' },
    { id: 'too_long', label: 'Respuesta muy larga', emoji: '📏', severity: 'low' },
    { id: 'too_short', label: 'Respuesta muy corta', emoji: '📝', severity: 'low' },
    { id: 'other', label: 'Otro problema', emoji: '💬', severity: 'medium' }
  ];

  const toggleIssue = (issueId) => {
    setSelectedIssues(prev =>
      prev.includes(issueId)
        ? prev.filter(id => id !== issueId)
        : [...prev, issueId]
    );
  };

  const submitSimpleFeedback = async (type) => {
    try {
      const feedbackItems = selectedIssues.map(issueId => {
        const issue = issueOptions.find(opt => opt.id === issueId);
        return {
          feedback_type: 'content_error',
          field: 'general',
          issue: issue.label,
          correct_value: '',
          severity: issue.severity
        };
      });

      if (customFeedback.trim()) {
        feedbackItems.push({
          feedback_type: 'suggestion',
          field: 'general',
          issue: customFeedback,
          correct_value: '',
          severity: 'medium'
        });
      }

      const response = await fetch(`${API_URL}/training/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: '',
          answer: '',
          sources: [],
          category_detected: 'general',
          processing_time: 0,
          status: type,
          evaluator_notes: customFeedback || 'Feedback rápido del usuario',
          feedback_items: feedbackItems
        })
      });

      if (response.ok) {
        setSubmitted(true);
        setTimeout(() => onClose(), 2000);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  if (submitted) {
    return (
      <div className="training-mode-overlay">
        <div className="feedback-success">
          <div className="success-icon">✅</div>
          <h2>¡Gracias por tu feedback!</h2>
          <p>Tu opinión nos ayuda a mejorar el sistema</p>
        </div>
      </div>
    );
  }

  return (
    <div className="training-mode-overlay">
      <div className="training-simple-container">
        <div className="training-header">
          <h1>💬 ¿Cómo podemos mejorar?</h1>
          <button className="close-button" onClick={onClose}>✕</button>
        </div>

        <div className="simple-feedback-content">
          <p className="feedback-intro">
            Selecciona los problemas que encontraste en la respuesta:
          </p>

          <div className="issue-grid">
            {issueOptions.map(option => (
              <button
                key={option.id}
                className={`issue-card ${selectedIssues.includes(option.id) ? 'selected' : ''}`}
                onClick={() => toggleIssue(option.id)}
              >
                <span className="issue-emoji">{option.emoji}</span>
                <span className="issue-label">{option.label}</span>
                {selectedIssues.includes(option.id) && (
                  <span className="check-mark">✓</span>
                )}
              </button>
            ))}
          </div>

          <div className="custom-feedback-section">
            <label>¿Algo más que quieras compartirnos? (opcional)</label>
            <textarea
              value={customFeedback}
              onChange={(e) => setCustomFeedback(e.target.value)}
              placeholder="Escribe aquí cualquier comentario adicional..."
              rows="4"
            />
          </div>

          <div className="feedback-actions">
            <button
              className="btn-submit-feedback"
              onClick={() => submitSimpleFeedback('needs_improvement')}
              disabled={selectedIssues.length === 0 && !customFeedback.trim()}
            >
              Enviar Feedback
            </button>
            <button className="btn-cancel" onClick={onClose}>
              Cancelar
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrainingModeSimple;
