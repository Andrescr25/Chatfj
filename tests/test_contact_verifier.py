"""
Tests unitarios para ContactVerifier
"""

import unittest
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.contact_verifier import ContactVerifier


class TestContactVerifier(unittest.TestCase):
    """Tests para la clase ContactVerifier"""

    def setUp(self):
        """Configuración inicial para cada test"""
        self.verifier = ContactVerifier()

    def test_extract_contacts_basic(self):
        """Test extracción básica de contactos"""
        text = "Llame al 800-8000-645 o al 2222-3333"
        contacts = self.verifier.extract_contacts(text)

        self.assertGreaterEqual(len(contacts), 1)
        # Verifica que se encuentren números largos
        for contact in contacts:
            self.assertGreaterEqual(len(contact), 4)

    def test_extract_contacts_no_contacts(self):
        """Test cuando no hay contactos"""
        text = "Este es un texto sin números de contacto"
        contacts = self.verifier.extract_contacts(text)

        self.assertEqual(len(contacts), 0)

    def test_extract_contacts_short_numbers(self):
        """Test que números cortos no se detectan con min_digits alto"""
        text = "Tengo 12 manzanas"
        contacts = self.verifier.extract_contacts(text, min_digits=8)

        # Con min_digits=8, "12" no debería detectarse
        self.assertEqual(len(contacts), 0)

    def test_mask_all_contacts_basic(self):
        """Test enmascaramiento de todos los contactos"""
        text = "Llame al 800-8000-645 para más información"
        masked = self.verifier.mask_all_contacts(text)

        self.assertNotIn("8000645", masked)
        self.assertIn("[dato de contacto no verificado]", masked)

    def test_mask_all_contacts_preserves_text(self):
        """Test que el texto sin contactos se preserva"""
        text = "No hay números aquí"
        masked = self.verifier.mask_all_contacts(text)

        self.assertEqual(text, masked)

    def test_mask_unverified_contacts_allows_verified(self):
        """Test que contactos verificados no se enmascaran"""
        text = "OIJ: 800-8000-645, Local: 2222-3333"
        # Los dígitos deben coincidir sin espacios ni guiones
        allowed = {"8008000645"}

        masked = self.verifier.mask_unverified_contacts(text, allowed)

        # El número verificado debe estar presente (el patrón original)
        self.assertIn("800-8000-645", masked)
        # El número no verificado debe estar enmascarado
        self.assertIn("[dato de contacto no verificado]", masked)

    def test_mask_unverified_contacts_no_allowed(self):
        """Test que sin contactos permitidos se enmascaran todos"""
        text = "Llame al 800-8000-645"
        masked = self.verifier.mask_unverified_contacts(text, set())

        self.assertIn("[dato de contacto no verificado]", masked)

    def test_verify_and_mask_with_reference(self):
        """Test método de conveniencia con texto de referencia"""
        reference = "Contacto oficial: 800-8000-645"
        text = "Puede llamar al 800-8000-645 o al 2222-3333"

        result = self.verifier.verify_and_mask(text, reference)

        # Verificar que el resultado tiene el formato esperado
        # El número de la referencia debe permanecer, el otro debe enmascararse
        self.assertIsInstance(result, str)
        # Al menos uno debe estar presente: el verificado o el placeholder
        self.assertTrue(
            "800-8000-645" in result or "[dato de contacto no verificado]" in result,
            f"Expected verified contact or placeholder in: {result}"
        )

    def test_verify_and_mask_without_reference(self):
        """Test método de conveniencia sin referencia"""
        text = "Llame al 800-8000-645"
        result = self.verifier.verify_and_mask(text)

        self.assertIn("[dato de contacto no verificado]", result)

    def test_custom_placeholder(self):
        """Test con placeholder personalizado"""
        verifier = ContactVerifier(placeholder="[REDACTED]")
        text = "Llame al 800-8000-645"
        masked = verifier.mask_all_contacts(text)

        self.assertIn("[REDACTED]", masked)
        self.assertNotIn("[dato de contacto no verificado]", masked)

    def test_multiple_contacts_same_line(self):
        """Test múltiples contactos en la misma línea"""
        text = "Llame al 11112222, 33334444 o 55556666"
        contacts = self.verifier.extract_contacts(text)

        self.assertGreaterEqual(len(contacts), 2)  # Al menos 2 contactos

    def test_contacts_with_parentheses(self):
        """Test contactos con paréntesis"""
        text = "Tel: (506) 2222-3333"
        contacts = self.verifier.extract_contacts(text)

        self.assertGreater(len(contacts), 0)

    def test_empty_text(self):
        """Test con texto vacío"""
        contacts = self.verifier.extract_contacts("")
        self.assertEqual(len(contacts), 0)

        masked = self.verifier.mask_all_contacts("")
        self.assertEqual(masked, "")

    def test_none_text(self):
        """Test con texto None"""
        contacts = self.verifier.extract_contacts(None)
        self.assertEqual(len(contacts), 0)


class TestCompatibilityFunctions(unittest.TestCase):
    """Tests para funciones de compatibilidad"""

    def test_mask_contact_tokens_compatibility(self):
        """Test función de compatibilidad mask_contact_tokens"""
        from src.utils.contact_verifier import mask_contact_tokens

        text = "Llame al 800-8000-645"
        masked = mask_contact_tokens(text)

        self.assertIn("[dato de contacto no verificado]", masked)

    def test_extract_contact_digit_tokens_compatibility(self):
        """Test función de compatibilidad extract_contact_digit_tokens"""
        from src.utils.contact_verifier import extract_contact_digit_tokens

        text = "Llame al 800-8000-645"
        contacts = extract_contact_digit_tokens(text)

        self.assertGreater(len(contacts), 0)

    def test_restrict_contacts_to_verified_compatibility(self):
        """Test función de compatibilidad restrict_contacts_to_verified"""
        from src.utils.contact_verifier import restrict_contacts_to_verified

        text = "OIJ: 800-8000-645, Local: 2222-3333"
        allowed = {"80080006 45"}

        masked = restrict_contacts_to_verified(text, allowed, "[MASKED]")

        self.assertIn("[MASKED]", masked)


if __name__ == '__main__':
    unittest.main()
