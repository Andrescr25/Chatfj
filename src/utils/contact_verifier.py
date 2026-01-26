"""
Utilidades para verificación de contactos
Refactoriza las 3 funciones duplicadas de api.py en una clase cohesiva
"""

import re
from typing import Set
from ..config.constants import CONTACT_PLACEHOLDER


class ContactVerifier:
    """Clase para manejar verificación y enmascaramiento de datos de contacto"""

    # Regex para detectar posibles datos de contacto
    CONTACT_TOKEN_REGEX = re.compile(r"[0-9][0-9A-Za-z\s()./-]{2,}")

    def __init__(self, placeholder: str = CONTACT_PLACEHOLDER):
        self.placeholder = placeholder

    def extract_contacts(self, text: str, min_digits: int = 4) -> Set[str]:
        """
        Extrae secuencias numéricas (≥min_digits dígitos) para validar contactos

        Args:
            text: Texto del que extraer contactos
            min_digits: Número mínimo de dígitos para considerar un contacto

        Returns:
            Set de strings con los dígitos de los contactos encontrados
        """
        contacts = set()
        if not text:
            return contacts

        for match in self.CONTACT_TOKEN_REGEX.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) >= min_digits:
                contacts.add(digits)

        return contacts

    def mask_all_contacts(self, text: str, min_digits: int = 4) -> str:
        """
        Reemplaza todos los teléfonos u otros datos de contacto con placeholder

        Args:
            text: Texto a procesar
            min_digits: Número mínimo de dígitos para considerar un contacto

        Returns:
            Texto con contactos enmascarados
        """
        def _replace(match: re.Match) -> str:
            token = match.group(0)
            digits = re.sub(r"\D", "", token)
            if len(digits) >= min_digits:
                return self.placeholder
            return token

        return self.CONTACT_TOKEN_REGEX.sub(_replace, text)

    def mask_unverified_contacts(
        self,
        text: str,
        allowed_contacts: Set[str],
        min_digits: int = 4
    ) -> str:
        """
        Permite solo números presentes en allowed_contacts, reemplaza el resto

        Args:
            text: Texto a procesar
            allowed_contacts: Set de strings con dígitos permitidos
            min_digits: Número mínimo de dígitos para considerar un contacto

        Returns:
            Texto con contactos no verificados enmascarados
        """
        if not allowed_contacts:
            return self.mask_all_contacts(text, min_digits)

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            digits = re.sub(r"\D", "", token)
            if len(digits) >= min_digits and digits not in allowed_contacts:
                return self.placeholder
            return token

        return self.CONTACT_TOKEN_REGEX.sub(_replace, text)

    def verify_and_mask(self, text: str, reference_text: str = None) -> str:
        """
        Método de conveniencia que extrae contactos de referencia y enmascara
        los no verificados en el texto

        Args:
            text: Texto a verificar y enmascarar
            reference_text: Texto de referencia del que extraer contactos permitidos

        Returns:
            Texto con contactos no verificados enmascarados
        """
        if reference_text:
            allowed = self.extract_contacts(reference_text)
            return self.mask_unverified_contacts(text, allowed)
        else:
            return self.mask_all_contacts(text)


# Instancia global para uso conveniente
contact_verifier = ContactVerifier()


# Funciones de compatibilidad con la API anterior
def mask_contact_tokens(text: str, placeholder: str = CONTACT_PLACEHOLDER) -> str:
    """Función de compatibilidad - reemplaza todos los contactos"""
    verifier = ContactVerifier(placeholder)
    return verifier.mask_all_contacts(text)


def extract_contact_digit_tokens(text: str) -> Set[str]:
    """Función de compatibilidad - extrae contactos"""
    return contact_verifier.extract_contacts(text)


def restrict_contacts_to_verified(
    text: str,
    allowed_digits: Set[str],
    placeholder: str = CONTACT_PLACEHOLDER
) -> str:
    """Función de compatibilidad - enmascara no verificados"""
    verifier = ContactVerifier(placeholder)
    return verifier.mask_unverified_contacts(text, allowed_digits)
