"""
Tests para verificar constantes y configuraciones
"""

import unittest
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.constants import (
    POPULAR_CR_INSTITUTIONS,
    POPULAR_CR_INSTITUTIONS_TEXT,
    LEGAL_CATEGORIES,
    CATEGORY_TO_LAWS,
    LLM_CONFIG,
    CACHE_CONFIG,
    SEARCH_CONFIG,
    UI_CONFIG,
    AMBIGUOUS_TERMS,
    CONTACT_PLACEHOLDER,
)


class TestConstants(unittest.TestCase):
    """Tests para verificar que las constantes están bien definidas"""

    def test_popular_institutions_not_empty(self):
        """Test que la lista de instituciones no está vacía"""
        self.assertGreater(len(POPULAR_CR_INSTITUTIONS), 0)
        self.assertIsInstance(POPULAR_CR_INSTITUTIONS, list)

    def test_popular_institutions_text_generated(self):
        """Test que el texto de instituciones se genera correctamente"""
        self.assertIsInstance(POPULAR_CR_INSTITUTIONS_TEXT, str)
        self.assertIn("✅", POPULAR_CR_INSTITUTIONS_TEXT)
        self.assertIn("Poder Judicial", POPULAR_CR_INSTITUTIONS_TEXT)

    def test_legal_categories_structure(self):
        """Test estructura de categorías legales"""
        self.assertIsInstance(LEGAL_CATEGORIES, dict)
        self.assertGreater(len(LEGAL_CATEGORIES), 0)

        # Verificar que cada categoría tiene una lista de variantes
        for category, variants in LEGAL_CATEGORIES.items():
            self.assertIsInstance(variants, list)
            self.assertGreater(len(variants), 0)

    def test_legal_categories_expected_keys(self):
        """Test que existen las categorías esperadas"""
        expected_categories = [
            "pension_alimentaria",
            "laboral",
            "penal",
            "familia",
            "civil",
            "transito",
            "violencia_domestica"
        ]

        for category in expected_categories:
            self.assertIn(category, LEGAL_CATEGORIES)

    def test_category_to_laws_mapping(self):
        """Test mapeo de categorías a leyes"""
        self.assertIsInstance(CATEGORY_TO_LAWS, dict)

        # Cada categoría debe tener al menos una ley
        for category, laws in CATEGORY_TO_LAWS.items():
            self.assertIsInstance(laws, list)
            self.assertGreater(len(laws), 0)

    def test_category_to_laws_matches_legal_categories(self):
        """Test que las claves coinciden entre ambos dicts"""
        categories_keys = set(LEGAL_CATEGORIES.keys())
        laws_keys = set(CATEGORY_TO_LAWS.keys())

        self.assertEqual(categories_keys, laws_keys)

    def test_llm_config_structure(self):
        """Test configuración del LLM"""
        required_keys = ["max_tokens", "temperature", "model"]

        for key in required_keys:
            self.assertIn(key, LLM_CONFIG)

        self.assertIsInstance(LLM_CONFIG["max_tokens"], int)
        self.assertIsInstance(LLM_CONFIG["temperature"], (int, float))
        self.assertIsInstance(LLM_CONFIG["model"], str)

    def test_llm_config_reasonable_values(self):
        """Test que los valores del LLM son razonables"""
        self.assertGreater(LLM_CONFIG["max_tokens"], 0)
        self.assertGreaterEqual(LLM_CONFIG["temperature"], 0)
        self.assertLessEqual(LLM_CONFIG["temperature"], 2)

    def test_cache_config_structure(self):
        """Test configuración de caché"""
        self.assertIn("ttl_seconds", CACHE_CONFIG)
        self.assertIn("max_size", CACHE_CONFIG)

        self.assertIsInstance(CACHE_CONFIG["ttl_seconds"], int)
        self.assertIsInstance(CACHE_CONFIG["max_size"], int)

    def test_search_config_structure(self):
        """Test configuración de búsqueda"""
        required_keys = ["top_k", "confidence_threshold", "similarity_threshold"]

        for key in required_keys:
            self.assertIn(key, SEARCH_CONFIG)

    def test_search_config_reasonable_values(self):
        """Test valores razonables de búsqueda"""
        self.assertGreater(SEARCH_CONFIG["top_k"], 0)
        self.assertGreaterEqual(SEARCH_CONFIG["confidence_threshold"], 0)
        self.assertLessEqual(SEARCH_CONFIG["confidence_threshold"], 100)
        self.assertGreaterEqual(SEARCH_CONFIG["similarity_threshold"], 0)
        self.assertLessEqual(SEARCH_CONFIG["similarity_threshold"], 1)

    def test_ui_config_structure(self):
        """Test configuración de UI"""
        required_keys = ["typing_speed_ms", "max_message_length"]

        for key in required_keys:
            self.assertIn(key, UI_CONFIG)

    def test_ambiguous_terms_structure(self):
        """Test estructura de términos ambiguos"""
        self.assertIsInstance(AMBIGUOUS_TERMS, dict)

        for term, clarifications in AMBIGUOUS_TERMS.items():
            self.assertIsInstance(clarifications, list)
            self.assertGreater(len(clarifications), 1)  # Debe tener al menos 2 opciones

    def test_contact_placeholder_defined(self):
        """Test que el placeholder de contactos está definido"""
        self.assertIsInstance(CONTACT_PLACEHOLDER, str)
        self.assertGreater(len(CONTACT_PLACEHOLDER), 0)


if __name__ == '__main__':
    unittest.main()
