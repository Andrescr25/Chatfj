/**
 * Tests para el servicio de API
 */

import APIService from '../api';

// Mock de fetch global
global.fetch = jest.fn();

describe('APIService', () => {
  beforeEach(() => {
    fetch.mockClear();
    localStorage.clear();
  });

  describe('request', () => {
    it('debe hacer una llamada fetch exitosa', async () => {
      const mockResponse = { success: true, data: 'test' };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await APIService.request('/test');

      expect(fetch).toHaveBeenCalledTimes(1);
      expect(result).toEqual(mockResponse);
    });

    it('debe manejar errores HTTP', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => { throw new Error('sin cuerpo'); },
      });

      await expect(APIService.request('/test')).rejects.toThrow('HTTP error! status: 404');
    });

    it('debe leer los errores de validación 422 (detail como lista)', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({
          detail: [
            {
              type: 'value_error',
              loc: ['body', 'password'],
              msg: 'Value error, La contraseña debe tener al menos 8 caracteres.',
            },
          ],
        }),
      });

      await expect(APIService.request('/admins', { method: 'POST' }))
        .rejects.toThrow('La contraseña debe tener al menos 8 caracteres.');
    });

    it('debe usar el mensaje del backend cuando viene en "detail"', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Debe quedar al menos una persona administradora activa.' }),
      });

      await expect(APIService.request('/admins/x', { method: 'DELETE' }))
        .rejects.toThrow('Debe quedar al menos una persona administradora activa.');
    });

    it('debe incluir headers por defecto', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      await APIService.request('/test');

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('debe adjuntar el token de sesión si existe', async () => {
      localStorage.setItem('adminToken', 'token-de-prueba');
      fetch.mockResolvedValueOnce({ ok: true, json: async () => ({}) });

      await APIService.request('/documents');

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer token-de-prueba',
          }),
        })
      );
    });
  });

  describe('ask', () => {
    it('debe enviar pregunta correctamente', async () => {
      const mockResponse = { answer: 'respuesta' };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await APIService.ask('test question', []);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/ask'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ question: 'test question', history: [] }),
        })
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe('submitFeedback', () => {
    it('debe enviar feedback al endpoint vigente', async () => {
      const feedbackData = { items: [{ intent: 'correction' }] };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'success' }),
      });

      await APIService.submitFeedback(feedbackData);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/feedback'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(feedbackData),
        })
      );
    });
  });

  describe('documentos', () => {
    it('debe listar documentos', async () => {
      const mockResponse = { documents: [], stats: { documentos: 0 } };
      fetch.mockResolvedValueOnce({ ok: true, json: async () => mockResponse });

      const result = await APIService.listDocuments();

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/documents?include_deleted=false'),
        expect.any(Object)
      );
      expect(result).toEqual(mockResponse);
    });

    it('debe subir documento como FormData', async () => {
      const mockFile = new File(['content'], 'test.pdf', { type: 'application/pdf' });
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ doc_id: 'test-123', status: 'pendiente' }),
      });

      await APIService.uploadDocument(mockFile, 'laboral', 'Título');

      const [url, config] = fetch.mock.calls[0];
      expect(url).toContain('/documents');
      expect(config.method).toBe('POST');
      expect(config.body).toBeInstanceOf(FormData);
    });

    it('debe propagar el error del backend al subir', async () => {
      const mockFile = new File(['content'], 'test.pdf');
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ detail: 'Ese mismo archivo ya está indexado.' }),
      });

      await expect(APIService.uploadDocument(mockFile, 'general'))
        .rejects.toThrow('Ese mismo archivo ya está indexado.');
    });

    it('debe eliminar un documento por su id', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'success', fragmentos_eliminados: 12 }),
      });

      const result = await APIService.deleteDocument('codigo-civil-abc123');

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/documents/codigo-civil-abc123'),
        expect.objectContaining({ method: 'DELETE' })
      );
      expect(result.fragmentos_eliminados).toBe(12);
    });
  });

  describe('administradores', () => {
    it('debe crear un administrador', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ uid: 'abc', email: 'nueva@persona.cr', role: 'admin' }),
      });

      await APIService.createAdmin({ email: 'nueva@persona.cr', name: 'Nueva' });

      const [url, config] = fetch.mock.calls[0];
      expect(url).toContain('/admins');
      expect(config.method).toBe('POST');
      expect(JSON.parse(config.body)).toEqual({
        email: 'nueva@persona.cr',
        name: 'Nueva',
        password: null,
      });
    });

    it('debe deshabilitar un administrador', async () => {
      fetch.mockResolvedValueOnce({ ok: true, json: async () => ({ disabled: true }) });

      await APIService.updateAdmin('uid-1', { disabled: true });

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/admins/uid-1'),
        expect.objectContaining({ method: 'PATCH' })
      );
    });

    it('debe revocar el acceso de un administrador', async () => {
      fetch.mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'success' }) });

      await APIService.removeAdmin('uid-1');

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/admins/uid-1'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });
});
