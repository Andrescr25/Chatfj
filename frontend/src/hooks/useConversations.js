/**
 * Hook personalizado para gestión de conversaciones
 * Extrae la lógica de gestión de conversaciones de App.js
 */

import { useState, useEffect } from 'react';

const useConversations = () => {
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);

  // Cargar conversaciones del localStorage al iniciar
  useEffect(() => {
    const savedConversations = localStorage.getItem('conversations');
    if (savedConversations) {
      const parsed = JSON.parse(savedConversations);
      setConversations(parsed);
      if (parsed.length > 0) {
        setActiveConversationId(parsed[0].id);
      }
    } else {
      // Crear conversación inicial
      createNewConversation();
    }
  }, []);

  // Guardar conversaciones en localStorage cuando cambien
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  const createNewConversation = () => {
    const newConv = {
      id: Date.now(),
      title: 'Nueva conversación',
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setConversations(prev => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
    return newConv;
  };

  const deleteConversation = (id) => {
    setConversations(prev => {
      const filtered = prev.filter(c => c.id !== id);

      // Si se eliminó la conversación activa, cambiar a otra
      if (id === activeConversationId) {
        if (filtered.length > 0) {
          setActiveConversationId(filtered[0].id);
        } else {
          // Crear nueva conversación si no quedan
          const newConv = {
            id: Date.now(),
            title: 'Nueva conversación',
            messages: [],
            createdAt: new Date().toISOString(),
          };
          setActiveConversationId(newConv.id);
          return [newConv];
        }
      }

      return filtered;
    });
  };

  const updateConversationTitle = (id, newTitle) => {
    setConversations(prev =>
      prev.map(c => c.id === id ? { ...c, title: newTitle } : c)
    );
  };

  const addMessageToConversation = (id, message) => {
    setConversations(prev =>
      prev.map(c =>
        c.id === id
          ? { ...c, messages: [...c.messages, message] }
          : c
      )
    );
  };

  const updateLastMessage = (id, updatedMessage) => {
    setConversations(prev =>
      prev.map(c => {
        if (c.id === id) {
          const messages = [...c.messages];
          messages[messages.length - 1] = updatedMessage;
          return { ...c, messages };
        }
        return c;
      })
    );
  };

  const getActiveConversation = () => {
    return conversations.find(c => c.id === activeConversationId);
  };

  const clearAllConversations = () => {
    setConversations([]);
    localStorage.removeItem('conversations');
    createNewConversation();
  };

  return {
    conversations,
    activeConversationId,
    setActiveConversationId,
    activeConversation: getActiveConversation(),
    createNewConversation,
    deleteConversation,
    updateConversationTitle,
    addMessageToConversation,
    updateLastMessage,
    clearAllConversations,
  };
};

export default useConversations;
