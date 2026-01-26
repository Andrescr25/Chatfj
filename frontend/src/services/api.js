/**
 * Servicio centralizado de API
 * Elimina la duplicación de API_URL en 4 componentes
 */

const API_URL = process.env.REACT_APP_API_URL || '';

class APIService {
  constructor() {
    this.baseURL = API_URL;
  }

  /**
   * Método base para hacer llamadas a la API
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  /**
   * Enviar pregunta al chatbot
   */
  async ask(question, history = []) {
    return this.request('/ask', {
      method: 'POST',
      body: JSON.stringify({ question, history }),
    });
  }

  /**
   * Enviar pregunta con streaming
   */
  async askStream(question, history = []) {
    const url = `${this.baseURL}/ask/stream`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
    });
    return response;
  }

  /**
   * Enviar feedback de entrenamiento
   */
  async submitFeedback(data) {
    return this.request('/training/feedback', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * Enviar corrección
   */
  async submitCorrection(data) {
    return this.request('/training/learn-correction', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  /**
   * Subir documento
   */
  async uploadDocument(file, category) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);

    const url = `${this.baseURL}/training/upload-document`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload error! status: ${response.status}`);
    }

    return await response.json();
  }

  /**
   * Obtener estadísticas de entrenamiento
   */
  async getTrainingStats() {
    return this.request('/training/statistics');
  }

  /**
   * Obtener estadísticas generales
   */
  async getStats() {
    return this.request('/stats');
  }

  /**
   * Obtener estadísticas de caché
   */
  async getCacheStats() {
    return this.request('/cache-stats');
  }

  /**
   * Limpiar caché
   */
  async clearCache() {
    return this.request('/clear-cache', {
      method: 'POST',
    });
  }

  /**
   * Hacer pregunta de entrenamiento
   */
  async askTraining(question, history = []) {
    return this.request('/training/ask', {
      method: 'POST',
      body: JSON.stringify({ question, history }),
    });
  }

  /**
   * Hacer pregunta de entrenamiento con streaming
   */
  async askTrainingStream(question, history = []) {
    const url = `${this.baseURL}/training/ask/stream`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
    });
    return response;
  }
}

export default new APIService();
