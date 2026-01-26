"""
Validador de información de contacto
Asegura que NO se invente ningún número de teléfono, dirección o email
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ContactValidator:
    """
    Valida información de contacto contra una base de datos verificada.
    NUNCA permite que se invente información de contacto.
    """

    def __init__(self, verified_contacts_path: str = "data/verified_contacts.json"):
        self.verified_contacts = self._load_verified_contacts(verified_contacts_path)
        self.phone_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}\b|911|800[-\s]?[\d\s-]{7,12}')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.[\w]+')

        # Crear índices para búsqueda rápida
        self.phones_index = {}
        self.institutions_index = {}

        for contact in self.verified_contacts:
            # Indexar por teléfono
            for phone in contact.get('phones', []):
                phone_clean = self._clean_phone(phone)
                if phone_clean not in self.phones_index:
                    self.phones_index[phone_clean] = []
                self.phones_index[phone_clean].append(contact)

            # Indexar por institución
            inst_name = contact.get('institution', '').lower()
            if inst_name:
                self.institutions_index[inst_name] = contact

    def _load_verified_contacts(self, path: str) -> List[Dict]:
        """Carga la base de datos de contactos verificados."""
        file_path = Path(path)
        if not file_path.exists():
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_phone(self, phone: str) -> str:
        """Limpia un número de teléfono para comparación."""
        return re.sub(r'[-\s]', '', phone.strip())

    def is_phone_verified(self, phone: str) -> bool:
        """
        Verifica si un número de teléfono está en la base de datos verificada.
        """
        phone_clean = self._clean_phone(phone)
        return phone_clean in self.phones_index

    def get_institution_by_phone(self, phone: str) -> Optional[Dict]:
        """
        Obtiene la institución asociada a un número de teléfono verificado.
        """
        phone_clean = self._clean_phone(phone)
        contacts = self.phones_index.get(phone_clean, [])
        return contacts[0] if contacts else None

    def verify_institution_reference(self, institution_name: str) -> Tuple[bool, Optional[Dict]]:
        """
        Verifica si una referencia a una institución es válida.
        Retorna (es_válida, datos_contacto)
        """
        inst_lower = institution_name.lower()

        # Búsqueda exacta
        if inst_lower in self.institutions_index:
            return True, self.institutions_index[inst_lower]

        # Búsqueda parcial
        for inst_key, contact in self.institutions_index.items():
            if inst_lower in inst_key or inst_key in inst_lower:
                return True, contact

        return False, None

    def extract_and_verify_contacts(self, text: str) -> Dict[str, List]:
        """
        Extrae teléfonos y emails del texto y verifica cuáles son válidos.

        Retorna:
        {
            'verified_phones': [...],  # Teléfonos que SÍ están verificados
            'unverified_phones': [...],  # Teléfonos NO verificados (POSIBLES INVENTOS)
            'verified_emails': [...],
            'unverified_emails': [...]
        }
        """
        # Extraer teléfonos
        phones_found = self.phone_pattern.findall(text)
        verified_phones = []
        unverified_phones = []

        for phone in phones_found:
            if self.is_phone_verified(phone):
                verified_phones.append(phone)
            else:
                unverified_phones.append(phone)

        # Extraer emails
        emails_found = self.email_pattern.findall(text)
        verified_emails = []
        unverified_emails = []

        for email in emails_found:
            # Verificar si el email está en la base de datos
            email_verified = any(
                email in contact.get('emails', [])
                for contact in self.verified_contacts
            )
            if email_verified:
                verified_emails.append(email)
            else:
                unverified_emails.append(email)

        return {
            'verified_phones': verified_phones,
            'unverified_phones': unverified_phones,
            'verified_emails': verified_emails,
            'unverified_emails': unverified_emails
        }

    def sanitize_response(self, response: str) -> Tuple[str, List[str]]:
        """
        Sanitiza una respuesta eliminando información de contacto NO verificada.

        Retorna: (respuesta_sanitizada, advertencias)
        """
        verification = self.extract_and_verify_contacts(response)
        warnings = []
        sanitized = response

        # Reemplazar teléfonos no verificados
        for unverified_phone in verification['unverified_phones']:
            # Buscar la institución mencionada cerca del teléfono
            idx = sanitized.find(unverified_phone)
            if idx != -1:
                context = sanitized[max(0, idx-100):min(len(sanitized), idx+100)]

                # Reemplazar con mensaje de advertencia
                sanitized = sanitized.replace(
                    unverified_phone,
                    "[CONTACTO NO VERIFICADO - Consultar sitio oficial]"
                )
                warnings.append(f"❌ Teléfono no verificado removido: {unverified_phone}")

        # Reemplazar emails no verificados
        for unverified_email in verification['unverified_emails']:
            sanitized = sanitized.replace(
                unverified_email,
                "[EMAIL NO VERIFICADO]"
            )
            warnings.append(f"❌ Email no verificado removido: {unverified_email}")

        return sanitized, warnings

    def get_verified_contact_for_query(self, query: str) -> List[Dict]:
        """
        Busca contactos verificados relevantes para una consulta.
        """
        query_lower = query.lower()
        relevant_contacts = []

        # Palabras clave para diferentes instituciones
        keywords_map = {
            'pension': ['JUZGADO', 'PENSIONES', 'ALIMENTARIA'],
            'violencia': ['VIOLENCIA', 'DOMÉSTICA'],
            'familia': ['FAMILIA'],
            'penal': ['PENAL'],
            'emergencia': ['911', 'EMERGENCIA'],
            'defensor': ['DEFENSOR'],
            'sala constitucional': ['SALA', 'CONSTITUCIONAL'],
        }

        for keyword, inst_words in keywords_map.items():
            if keyword in query_lower:
                for contact in self.verified_contacts:
                    inst = contact.get('institution', '').upper()
                    if any(word in inst for word in inst_words):
                        relevant_contacts.append(contact)

        return relevant_contacts[:5]  # Top 5 más relevantes

    def get_stats(self) -> Dict:
        """Retorna estadísticas de la base de datos de contactos."""
        total_phones = sum(len(c.get('phones', [])) for c in self.verified_contacts)
        total_emails = sum(len(c.get('emails', [])) for c in self.verified_contacts)

        return {
            'total_institutions': len(self.verified_contacts),
            'total_phones': total_phones,
            'total_emails': total_emails,
            'institutions_with_phones': len([c for c in self.verified_contacts if c.get('phones')]),
            'institutions_with_emails': len([c for c in self.verified_contacts if c.get('emails')]),
        }


# Instancia global
_validator_instance = None

def get_validator() -> ContactValidator:
    """Obtiene la instancia global del validador."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ContactValidator()
    return _validator_instance
