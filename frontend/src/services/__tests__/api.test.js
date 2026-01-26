/**
 * Tests para el servicio de API
 */

import APIService from '../api';

// Mock de fetch global
global.fetch = jest.fn();

describe('APIService', () => {
  beforeEach(() => {
    // Limpiar mocks antes de cada test
    fetch.mockClear();
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
      });

      await expect(APIService.request('/test')).rejects.toThrow('HTTP error! status: 404');
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
    it('debe enviar feedback correctamente', async () => {
      const feedbackData = { question: 'test', answer: 'test answer', helpful: true };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await APIService.submitFeedback(feedbackData);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/training/feedback'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(feedbackData),
        })
      );
    });
  });

  describe('submitCorrection', () => {
    it('debe enviar corrección correctamente', async () => {
      const correctionData = { question: 'test', correct_answer: 'correct' };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await APIService.submitCorrection(correctionData);

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/training/learn-correction'),
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });

  describe('uploadDocument', () => {
    it('debe subir documento correctamente', async () => {
      const mockFile = new File(['content'], 'test.pdf', { type: 'application/pdf' });
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await APIService.uploadDocument(mockFile, 'laboral');

      expect(fetch).toHaveBeenCalledTimes(1);
      const callArgs = fetch.mock.calls[0];
      expect(callArgs[0]).toContain('/training/upload-document');
      expect(callArgs[1].method).toBe('POST');
      expect(callArgs[1].body).toBeInstanceOf(FormData);
    });

    it('debe manejar errores de upload', async () => {
      const mockFile = new File(['content'], 'test.pdf');
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      await expect(
        APIService.uploadDocument(mockFile, 'laboral')
      ).rejects.toThrow('Upload error! status: 500');
    });
  });

  describe('getStats', () => {
    it('debe obtener estadísticas', async () => {
      const mockStats = { total: 100, success: 80 };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockStats,
      });

      const result = await APIService.getStats();

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/stats'),
        expect.any(Object)
      );
      expect(result).toEqual(mockStats);
    });
  });

  describe('clearCache', () => {
    it('debe limpiar caché correctamente', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await APIService.clearCache();

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/clear-cache'),
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });
});
