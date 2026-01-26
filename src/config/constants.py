"""
Constantes de configuración del backend
Centraliza valores hardcodeados para facilitar mantenimiento
"""

# Instituciones populares de Costa Rica
POPULAR_CR_INSTITUTIONS = [
    "Poder Judicial",
    "Juzgado de Violencia Doméstica",
    "Juzgado de Familia",
    "Defensoría de los Habitantes",
    "Defensa Pública",
    "INAMU",
    "PANI",
    "Caja Costarricense de Seguro Social (CCSS)",
    "Ministerio de Trabajo y Seguridad Social (MTSS)",
    "Ministerio Público / Fiscalía",
    "Ministerio de Seguridad Pública",
    "Línea 911 de emergencias",
    "Oficinas locales del Poder Judicial"
]

POPULAR_CR_INSTITUTIONS_TEXT = "\n".join(
    f"  ✅ {name}" for name in POPULAR_CR_INSTITUTIONS
)

# Categorías legales y sus variantes
LEGAL_CATEGORIES = {
    "pension_alimentaria": [
        "pensión alimentaria", "pensión", "alimentos", "cuota alimentaria",
        "manutención", "ley 7654", "código de familia"
    ],
    "laboral": [
        "laboral", "trabajo", "despido", "salario", "horas extra",
        "código de trabajo", "prestaciones laborales", "aguinaldo", "cesantía"
    ],
    "penal": [
        "penal", "delito", "crimen", "denuncia penal", "código penal",
        "proceso penal", "imputado", "querella"
    ],
    "familia": [
        "familia", "divorcio", "matrimonio", "tutela", "curatela",
        "guarda", "crianza", "patria potestad", "código de familia"
    ],
    "civil": [
        "civil", "contrato", "obligaciones", "propiedad", "sucesiones",
        "herencia", "código civil", "responsabilidad civil"
    ],
    "transito": [
        "tránsito", "accidente de tránsito", "multa de tránsito",
        "licencia de conducir", "ley de tránsito"
    ],
    "violencia_domestica": [
        "violencia doméstica", "violencia intrafamiliar", "medidas de protección",
        "ley contra la violencia doméstica", "orden de alejamiento"
    ]
}

# Mapeo de categorías a leyes
CATEGORY_TO_LAWS = {
    "pension_alimentaria": ["7654", "codigo_familia"],
    "laboral": ["codigo_trabajo"],
    "penal": ["codigo_penal"],
    "familia": ["codigo_familia"],
    "civil": ["codigo_civil"],
    "transito": ["ley_transito"],
    "violencia_domestica": ["ley_violencia_domestica"]
}

# Configuración del LLM - PREMIUM QUALITY MODE
LLM_CONFIG = {
    "max_tokens": 4000,  # Respuestas más largas y detalladas
    "temperature": 0.7,  # Más consistente para respuestas legales
    "model": "openai/gpt-4o",  # Modelo más potente para máxima calidad
}

# Configuración de caché
CACHE_CONFIG = {
    "ttl_seconds": 3600,  # 1 hora
    "max_size": 1000,
}

# Configuración de búsqueda de documentos - PREMIUM QUALITY MODE
SEARCH_CONFIG = {
    "top_k": 15,  # Recuperar muchos más documentos para contexto completo
    "confidence_threshold": 50.0,  # Más permisivo para no perder información relevante
    "similarity_threshold": 0.70,  # Ligeramente más permisivo para correcciones
}

# Configuración UI
UI_CONFIG = {
    "typing_speed_ms": 20,
    "max_message_length": 1000,
}

# Términos ambiguos que requieren clarificación
AMBIGUOUS_TERMS = {
    "demanda": ["demanda laboral", "demanda civil", "demanda penal", "demanda de alimentos"],
    "pensión": ["pensión alimentaria", "pensión de vejez", "pensión por invalidez"],
    "denuncia": ["denuncia penal", "denuncia administrativa"],
    "medidas": ["medidas cautelares", "medidas de protección"],
}

# Placeholder para contactos no verificados
CONTACT_PLACEHOLDER = "[dato de contacto no verificado]"
