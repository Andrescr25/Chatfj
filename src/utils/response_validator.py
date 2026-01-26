"""
Validador de respuestas para mejorar precisión
Verifica que las respuestas sean coherentes y confiables
"""

import re
from typing import Dict, List, Tuple


class ResponseValidator:
    """Valida y mejora la calidad de las respuestas de la IA"""

    # Frases que indican incertidumbre
    UNCERTAINTY_PHRASES = [
        "no estoy seguro",
        "no tengo información",
        "no puedo confirmar",
        "posiblemente",
        "tal vez",
        "probablemente",
        "es posible que",
        "podría ser",
        "no encuentro",
        "no dispongo de",
    ]

    # Frases que indican invención
    HALLUCINATION_INDICATORS = [
        "según mis datos",
        "en mi base de datos",
        "he sido entrenado",
        "como IA",
        "como modelo de lenguaje",
    ]

    @classmethod
    def validate_response(
        cls,
        question: str,
        answer: str,
        sources: List[Dict],
        min_confidence: float = 0.7
    ) -> Tuple[bool, str, Dict]:
        """
        Valida una respuesta y retorna si es confiable

        Args:
            question: Pregunta del usuario
            answer: Respuesta generada
            sources: Fuentes usadas
            min_confidence: Confianza mínima requerida (0-1)

        Returns:
            (is_valid, improved_answer, metadata)
        """
        issues = []
        warnings = []
        confidence = 1.0

        # 1. Verificar que haya fuentes
        if not sources or len(sources) == 0:
            issues.append("No hay fuentes para respaldar la respuesta")
            confidence *= 0.5

        # 2. Detectar incertidumbre
        uncertainty_count = sum(
            1 for phrase in cls.UNCERTAINTY_PHRASES
            if phrase in answer.lower()
        )
        if uncertainty_count > 0:
            warnings.append(f"Respuesta con incertidumbre ({uncertainty_count} frases)")
            confidence *= (1 - uncertainty_count * 0.1)

        # 3. Detectar posibles alucinaciones
        hallucination_count = sum(
            1 for phrase in cls.HALLUCINATION_INDICATORS
            if phrase in answer.lower()
        )
        if hallucination_count > 0:
            issues.append("Respuesta podría contener información inventada")
            confidence *= 0.6

        # 4. Verificar longitud razonable
        if len(answer) < 50:
            warnings.append("Respuesta muy corta")
            confidence *= 0.9
        elif len(answer) > 3000:
            warnings.append("Respuesta muy larga")
            confidence *= 0.95

        # 5. Verificar que menciona fuentes legales
        has_legal_reference = any([
            "ley" in answer.lower(),
            "código" in answer.lower(),
            "artículo" in answer.lower(),
            "reglamento" in answer.lower(),
        ])

        if not has_legal_reference and sources:
            # Tiene fuentes pero no las menciona
            warnings.append("No menciona referencias legales específicas")
            confidence *= 0.9

        # 6. Verificar números de contacto
        phone_pattern = r'\d{4}[-\s]?\d{4}'
        phones_in_answer = re.findall(phone_pattern, answer)
        if phones_in_answer and not sources:
            issues.append("Menciona contactos sin fuentes verificadas")
            confidence *= 0.7

        # Decisión final
        is_valid = confidence >= min_confidence and len(issues) == 0

        # Mejorar respuesta si es necesario
        improved_answer = answer
        if not is_valid:
            improved_answer = cls._add_disclaimer(answer, issues, warnings)

        metadata = {
            "confidence": round(confidence, 2),
            "issues": issues,
            "warnings": warnings,
            "sources_count": len(sources),
            "has_legal_reference": has_legal_reference,
        }

        return is_valid, improved_answer, metadata

    @classmethod
    def _add_disclaimer(cls, answer: str, issues: List[str], warnings: List[str]) -> str:
        """Agrega advertencias a respuestas de baja confianza"""

        disclaimer_parts = []

        if issues:
            disclaimer_parts.append(
                "⚠️ **Nota importante**: Esta respuesta tiene limitaciones:\n"
                + "\n".join(f"  • {issue}" for issue in issues)
            )

        if warnings:
            disclaimer_parts.append(
                "\n💡 **Recomendación**: "
                + "; ".join(warnings)
            )

        disclaimer_parts.append(
            "\n\n📞 Para información precisa y actualizada, te recomiendo contactar "
            "directamente con el Poder Judicial o un facilitador judicial certificado."
        )

        return answer + "\n\n" + "\n".join(disclaimer_parts)

    @classmethod
    def check_answer_relevance(cls, question: str, answer: str) -> float:
        """
        Verifica qué tan relevante es la respuesta a la pregunta

        Returns:
            Score de relevancia (0-1)
        """
        # Extraer palabras clave de la pregunta
        question_words = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', question)
            if len(word) > 3
        )

        # Quitar palabras comunes
        stop_words = {
            'para', 'como', 'donde', 'cuando', 'quien', 'cual', 'puedo',
            'debo', 'tengo', 'hacer', 'sobre', 'acerca'
        }
        question_words -= stop_words

        if not question_words:
            return 1.0

        # Contar cuántas palabras clave aparecen en la respuesta
        answer_lower = answer.lower()
        matched_words = sum(1 for word in question_words if word in answer_lower)

        relevance = matched_words / len(question_words)
        return min(relevance, 1.0)

    @classmethod
    def suggest_improvements(cls, question: str, answer: str, sources: List[Dict]) -> List[str]:
        """Sugiere mejoras para la respuesta"""
        suggestions = []

        # Verificar si cita fuentes
        if sources and "[" not in answer:
            suggestions.append(
                "Agregar referencias numéricas a las fuentes (ej: [1], [2])"
            )

        # Verificar estructura
        if "\n" not in answer and len(answer) > 200:
            suggestions.append(
                "Mejorar formato con párrafos o viñetas para mejor lectura"
            )

        # Verificar que responda directamente
        question_lower = question.lower()
        if question_lower.startswith(("cómo", "cuál", "dónde", "cuándo")):
            if not any(word in answer.lower() for word in ["paso", "proceso", "ubicado", "fecha"]):
                suggestions.append(
                    "Asegurar que responda directamente a la pregunta específica"
                )

        return suggestions


# Funciones de conveniencia
def validate_answer(question: str, answer: str, sources: List[Dict]) -> Dict:
    """Función de conveniencia para validar respuestas"""
    is_valid, improved_answer, metadata = ResponseValidator.validate_response(
        question, answer, sources
    )

    return {
        "is_valid": is_valid,
        "answer": improved_answer,
        "original_answer": answer,
        **metadata
    }
