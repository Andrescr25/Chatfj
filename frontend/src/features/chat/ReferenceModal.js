import React from 'react';
import { FileText, X, ExternalLink } from 'lucide-react';

/**
 * Modal que muestra el fragmento citado en una respuesta.
 *
 * Recibe la referencia ya resuelta; no sabe nada de conversaciones ni de la API.
 */
function ReferenceModal({ reference, onClose }) {
  if (!reference) return null;

  return (
        <div className="reference-modal-overlay" onClick={() => onClose()}>
          <div className="reference-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3><FileText size={20} style={{verticalAlign: 'bottom', marginRight: '8px'}}/> {reference.filename || 'Fuente'}</h3>
              <button
                className="modal-close-btn"
                onClick={() => onClose()}
              >
                <X size={20} />
              </button>
            </div>
            <div className="modal-content">
              {reference.type === 'web' ? (
                // Si es referencia web, mostrar botón para abrir URL
                <div className="web-reference-content">
                  <p className="web-reference-description">
                    {reference.content || reference.snippet || reference.title}
                  </p>
                  <a
                    href={reference.url || reference.source}
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
                  const content = reference.content || reference.snippet;
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
                onClick={() => onClose()}
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
  );
}

export default ReferenceModal;
