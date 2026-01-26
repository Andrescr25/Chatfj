/**
 * Hook personalizado para gestión de tema (dark/light)
 */

import { useState, useEffect } from 'react';
import { THEME } from '../config/constants';

const useTheme = () => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved === THEME.DARK;
  });

  useEffect(() => {
    const theme = isDarkMode ? THEME.DARK : THEME.LIGHT;
    localStorage.setItem('theme', theme);
    document.documentElement.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  const toggleTheme = () => {
    setIsDarkMode(prev => !prev);
  };

  return {
    isDarkMode,
    toggleTheme,
    theme: isDarkMode ? THEME.DARK : THEME.LIGHT,
  };
};

export default useTheme;
