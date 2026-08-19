/**
 * Servicio centralizado de API
 * Elimina la duplicación de API_URL en varios componentes
 */

import { auth } from '../config/firebase';

const API_URL = process.env.REACT_APP_API_URL || '';

class APIService {
  constructor() {
    this.baseURL = API_URL;
  }

  /**
   * Token de sesión fresco de Firebase.
   * Importante: los roles viajan dentro del token, así que tras un cambio de
   * rol hay que forzar la renovación (forceRefresh) para que surta efecto.
   */
  async getToken(forceRefresh = false) {
    if (auth.currentUser) {
      try {
        const token = await auth.currentUser.getIdToken(forceRefresh);
        localStorage.setItem('adminToken', token);
        return token;
      } catch (error) {
        console.error('Error al obtener ID Token de Firebase:', error);
      }
    }
    return localStorage.getItem('adminToken');
  }

  /**
   * Extrae el mensaje de error que envía el backend (campo "detail").
   */
  async buildError(response) {
    let detail = '';
    try {
      const data = await response.json();
      detail = this.formatDetail(data.detail ?? data.message);
    } catch (e) {
      // respuesta sin cuerpo JSON
    }
    const error = new Error(detail || `HTTP error! status: ${response.status}`);
    error.status = response.status;
    return error;
  }

  /**
   * FastAPI devuelve "detail" como texto en los errores propios, pero como lista
   * de objetos en los errores de validación (422). Sin esto, la interfaz
   * mostraba "[object Object]".
   */
  formatDetail(detail) {
    if (!detail) return '';
    if (typeof detail === 'string') return detail;

    const limpiar = (msg) => String(msg).replace(/^Value error,\s*/, '');

    if (Array.isArray(detail)) {
      return detail
        .map((item) => (typeof item === 'string' ? item : limpiar(item?.msg || '')))
        .filter(Boolean)
        .join(' ');
    }

    return limpiar(detail.msg || JSON.stringify(detail));
  }

  /**
   * Método base para hacer llamadas a la API
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;

    const adminToken = await this.getToken();

    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (adminToken) {
      headers['Authorization'] = `Bearer ${adminToken}`;
    }

    const config = {
      ...options,
      headers,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw await this.buildError(response);
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
   * Enviar retroalimentación o corrección de entrenamiento
   */
  async submitFeedback(data) {
    return this.request('/feedback', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // ===== Identidad =====

  /**
   * Identidad y rol de la persona autenticada
   */
  async whoAmI() {
    return this.request('/admins/me');
  }

  // ===== Documentos indexados =====

  async listDocuments(includeDeleted = false) {
    return this.request(`/documents?include_deleted=${includeDeleted}`);
  }

  async getDocument(docId) {
    return this.request(`/documents/${encodeURIComponent(docId)}`);
  }

  /**
   * Sube un documento. La indexación ocurre en segundo plano: hay que consultar
   * getDocument() para ver el progreso.
   */
  async uploadDocument(file, category = 'general', title = '') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);
    formData.append('title', title);

    const token = await this.getToken();
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${this.baseURL}/documents`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw await this.buildError(response);
    }

    return await response.json();
  }

  /**
   * Texto indexado del documento, por fragmentos (lo que el asistente lee)
   */
  async getDocumentContent(docId, offset = 0, limit = 20) {
    return this.request(
      `/documents/${encodeURIComponent(docId)}/content?offset=${offset}&limit=${limit}`
    );
  }

  async reindexDocument(docId) {
    return this.request(`/documents/${encodeURIComponent(docId)}/reindex`, {
      method: 'POST',
    });
  }

  async deleteDocument(docId) {
    return this.request(`/documents/${encodeURIComponent(docId)}`, {
      method: 'DELETE',
    });
  }

  /**
   * Descarga el archivo original de un documento indexado
   */
  async downloadDocument(docId, filename) {
    const token = await this.getToken();
    const response = await fetch(
      `${this.baseURL}/documents/${encodeURIComponent(docId)}/download`,
      { headers: token ? { Authorization: `Bearer ${token}` } : {} }
    );

    if (!response.ok) {
      throw await this.buildError(response);
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || 'documento';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }

  // ===== Administradores =====

  // ===== Correcciones aprendidas =====

  async listCorrections() {
    return this.request('/corrections');
  }

  async updateCorrection(id, cambios) {
    return this.request(`/corrections/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(cambios),
    });
  }

  async deleteCorrection(id) {
    return this.request(`/corrections/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  // ===== Uso del asistente =====

  async getDocumentStats(limite = 25) {
    return this.request(`/stats/documents?limite=${limite}`);
  }

  async getHistory(dias = 7, limite = 200) {
    return this.request(`/history?dias=${dias}&limite=${limite}`);
  }

  // ===== Administradores =====

  async listAdmins() {
    return this.request('/admins');
  }

  async createAdmin({ email, name = '', password = null }) {
    return this.request('/admins', {
      method: 'POST',
      body: JSON.stringify({ email, name, password }),
    });
  }

  async updateAdmin(uid, changes) {
    return this.request(`/admins/${encodeURIComponent(uid)}`, {
      method: 'PATCH',
      body: JSON.stringify(changes),
    });
  }

  async removeAdmin(uid) {
    return this.request(`/admins/${encodeURIComponent(uid)}`, {
      method: 'DELETE',
    });
  }
}

export default new APIService();
