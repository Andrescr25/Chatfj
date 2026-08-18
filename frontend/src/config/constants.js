/**
 * Constantes de la aplicación
 * Elimina la duplicación de categorías y opciones en múltiples archivos
 */

export const LEGAL_CATEGORIES = [
  { value: 'pension_alimentaria', label: 'Pensión Alimentaria' },
  { value: 'laboral', label: 'Laboral' },
  { value: 'penal', label: 'Penal' },
  { value: 'familia', label: 'Familia' },
  { value: 'civil', label: 'Civil' },
  { value: 'transito', label: 'Tránsito' },
  { value: 'violencia_domestica', label: 'Violencia Doméstica' },
];

export const ISSUE_OPTIONS = [
  {
    id: 'incomplete',
    label: 'Respuesta incompleta',
    description: 'Falta información importante'
  },
  {
    id: 'incorrect',
    label: 'Respuesta incorrecta',
    description: 'La información es errónea'
  },
  {
    id: 'unclear',
    label: 'Respuesta poco clara',
    description: 'Es confusa o difícil de entender'
  },
  {
    id: 'outdated',
    label: 'Información desactualizada',
    description: 'Basada en leyes o procedimientos antiguos'
  },
  {
    id: 'irrelevant',
    label: 'Respuesta irrelevante',
    description: 'No responde a lo preguntado'
  }
];

export const UI_TEXT = {
  // Mensajes principales
  WELCOME_MESSAGE: '¿En qué puedo ayudarte hoy?',
  DISCLAIMER: 'Chat FJ puede cometer errores. Verifica información importante.',
  THINKING: 'Pensando...',
  TYPING: 'Escribiendo...',

  // Botones
  SEND: 'Enviar',
  CANCEL: 'Cancelar',
  DELETE: 'Eliminar',
  EDIT: 'Editar',
  SAVE: 'Guardar',
  APPROVE: 'Aprobar',
  CORRECT: 'Corregir',

  // Mensajes de feedback
  FEEDBACK_SUCCESS: 'Gracias por tu feedback',
  CORRECTION_SUCCESS: 'Corrección guardada exitosamente',
  UPLOAD_SUCCESS: 'Documento subido exitosamente',

  // Errores
  ERROR_GENERIC: 'Ocurrió un error. Intenta nuevamente.',
  ERROR_UPLOAD: 'Error al subir el documento',
  ERROR_NETWORK: 'Error de conexión',
};

export const THEME = {
  LIGHT: 'light',
  DARK: 'dark',
};

export const TYPING_SPEED_MS = 20;
export const MESSAGE_MAX_LENGTH = 1000;
