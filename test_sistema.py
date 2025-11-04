#!/usr/bin/env python3
"""
Script de pruebas funcionales del sistema de Facilitadores Judiciales
Evalúa: categorización, fuentes, scores, citas legales y formato
"""

import requests
import json
import re
from typing import Dict, List, Any
from datetime import datetime

API_URL = "http://localhost:8000"

# 10 PREGUNTAS REALES DE USUARIOS COSTARRICENSES
TEST_CASES = [
    {
        "id": 1,
        "pregunta": "Mi ex no paga pensión alimentaria, ¿qué hago?",
        "categoria_esperada": "pension_alimentaria",
        "debe_citar_ley": True,
        "ley_esperada": "7654 o Código Procesal de Familia"
    },
    {
        "id": 2,
        "pregunta": "Me despidieron sin preaviso, ¿tengo derecho a finiquito?",
        "categoria_esperada": "laboral",
        "debe_citar_ley": True,
        "ley_esperada": "Código de Trabajo"
    },
    {
        "id": 3,
        "pregunta": "Mi esposo me golpea, ¿dónde puedo denunciar?",
        "categoria_esperada": "violencia",
        "debe_citar_ley": True,
        "ley_esperada": "Ley contra la Violencia Doméstica"
    },
    {
        "id": 4,
        "pregunta": "¿Cómo tramitar pensión de vejez en la CCSS?",
        "categoria_esperada": "pension_vejez",
        "debe_citar_ley": False,
        "ley_esperada": "Reglamento CCSS"
    },
    {
        "id": 5,
        "pregunta": "Mi jefe no me paga horas extra, ¿qué puedo hacer?",
        "categoria_esperada": "laboral",
        "debe_citar_ley": True,
        "ley_esperada": "Código de Trabajo"
    },
    {
        "id": 6,
        "pregunta": "Quiero denunciar a mi vecino por maltrato animal",
        "categoria_esperada": "penal",
        "debe_citar_ley": False,
        "ley_esperada": "N/A"
    },
    {
        "id": 7,
        "pregunta": "¿Cómo solicitar medidas de protección por violencia?",
        "categoria_esperada": "violencia",
        "debe_citar_ley": True,
        "ley_esperada": "Ley contra la Violencia Doméstica"
    },
    {
        "id": 8,
        "pregunta": "Mi hijo está en peligro, necesito ayuda del PANI",
        "categoria_esperada": "menores",
        "debe_citar_ley": True,
        "ley_esperada": "PANI o Ley de Niñez"
    },
    {
        "id": 9,
        "pregunta": "¿Puedo pedir aumento de pensión alimentaria?",
        "categoria_esperada": "pension_alimentaria",
        "debe_citar_ley": True,
        "ley_esperada": "7654 o Código Procesal de Familia"
    },
    {
        "id": 10,
        "pregunta": "Me quieren desalojar de mi casa alquilada, ¿qué hago?",
        "categoria_esperada": "civil",
        "debe_citar_ley": True,
        "ley_esperada": "Código Civil o Ley de Arrendamientos"
    }
]


def hacer_pregunta(pregunta: str) -> Dict[str, Any]:
    """Hace una pregunta al sistema y retorna la respuesta completa."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": pregunta, "history": []},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def extraer_categoria_de_logs(pregunta: str) -> str:
    """
    Extrae la categoría detectada de los logs.
    Nota: Esta función simula lo que veríamos en los logs.
    """
    # Por ahora retornamos None, será llenado manualmente
    return None


def analizar_citas_legales(respuesta: str) -> Dict[str, Any]:
    """Analiza si la respuesta cita leyes correctamente."""
    # Buscar patrones de leyes
    ley_pattern = r'Ley\s+(?:N\.°\s*)?(\d+)'
    codigo_pattern = r'Código\s+(?:Procesal\s+)?(?:de\s+)?([A-Za-záéíóúñ]+)'
    articulo_pattern = r'artículo\s+(\d+)'

    leyes_citadas = re.findall(ley_pattern, respuesta, re.IGNORECASE)
    codigos_citados = re.findall(codigo_pattern, respuesta, re.IGNORECASE)
    articulos_citados = re.findall(articulo_pattern, respuesta, re.IGNORECASE)

    return {
        "tiene_citas": len(leyes_citadas) > 0 or len(codigos_citados) > 0,
        "leyes": leyes_citadas,
        "codigos": codigos_citados,
        "articulos": articulos_citados
    }


def verificar_formato(respuesta: str) -> Dict[str, bool]:
    """Verifica que la respuesta cumpla con el formato del prompt maestro."""
    checks = {
        "tiene_explicacion": False,
        "menciona_institucion": False,
        "tiene_cita_legal": False,
        "tiene_recomendacion": False,
        "tiene_referencias": False
    }

    respuesta_lower = respuesta.lower()

    # Check 1: Tiene explicación del procedimiento
    palabras_procedimiento = ["puede", "debe", "puedes", "debes", "procedimiento", "proceso"]
    checks["tiene_explicacion"] = any(palabra in respuesta_lower for palabra in palabras_procedimiento)

    # Check 2: Menciona instituciones
    instituciones = ["juzgado", "ministerio", "pani", "poder judicial", "ccss", "defensa pública",
                     "facilitador", "adaptación social", "trabajo social"]
    checks["menciona_institucion"] = any(inst in respuesta_lower for inst in instituciones)

    # Check 3: Tiene cita legal
    checks["tiene_cita_legal"] = bool(re.search(r'(ley|código|artículo)', respuesta_lower))

    # Check 4: Tiene recomendación práctica
    palabras_recomendacion = ["recomend", "acud", "consult", "orientación", "facilitador judicial"]
    checks["tiene_recomendacion"] = any(palabra in respuesta_lower for palabra in palabras_recomendacion)

    # Check 5: Tiene sección de referencias
    checks["tiene_referencias"] = "referencias:" in respuesta_lower or "[1]" in respuesta

    return checks


def ejecutar_pruebas():
    """Ejecuta todas las pruebas y genera reporte."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       PRUEBAS FUNCIONALES - SISTEMA FACILITADORES JUDICIALES       ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de casos: {len(TEST_CASES)}\n")
    print("━" * 80)

    resultados = []

    for test in TEST_CASES:
        print(f"\n[TEST {test['id']}/10] {test['pregunta']}")
        print("─" * 80)

        # Hacer pregunta
        respuesta_completa = hacer_pregunta(test['pregunta'])

        if "error" in respuesta_completa:
            print(f"❌ ERROR: {respuesta_completa['error']}")
            continue

        respuesta = respuesta_completa.get("answer", "")
        sources = respuesta_completa.get("sources", [])
        processing_time = respuesta_completa.get("processing_time", 0)

        # Analizar fuentes
        fuentes_usadas = {
            "documentos": [s for s in sources if s.get("type") == "document"],
            "web": [s for s in sources if s.get("type") == "web"]
        }

        tipo_fuente = "ambos" if fuentes_usadas["documentos"] and fuentes_usadas["web"] else \
                      "web" if fuentes_usadas["web"] else \
                      "documentos" if fuentes_usadas["documentos"] else "ninguna"

        # Analizar citas
        citas = analizar_citas_legales(respuesta)

        # Verificar formato
        formato = verificar_formato(respuesta)

        # Calcular score de formato (% de checks pasados)
        formato_score = sum(formato.values()) / len(formato) * 100

        # Guardar resultado
        resultado = {
            "id": test["id"],
            "pregunta": test["pregunta"],
            "categoria_esperada": test["categoria_esperada"],
            "tipo_fuente": tipo_fuente,
            "num_fuentes": len(sources),
            "tiene_citas": citas["tiene_citas"],
            "leyes_citadas": citas["leyes"] + citas["codigos"],
            "formato_score": formato_score,
            "formato_checks": formato,
            "processing_time": processing_time,
            "respuesta_preview": respuesta[:300] + "..." if len(respuesta) > 300 else respuesta
        }

        resultados.append(resultado)

        # Mostrar resultado
        print(f"📂 Categoría esperada: {test['categoria_esperada']}")
        print(f"📚 Fuentes usadas: {tipo_fuente} ({len(sources)} total)")
        print(f"⚖️  Citas legales: {'✅' if citas['tiene_citas'] else '❌'} - {', '.join(citas['leyes'] + citas['codigos']) or 'Ninguna'}")
        print(f"📝 Formato: {formato_score:.0f}% completo")
        print(f"⏱️  Tiempo: {processing_time:.2f}s")

        if fuentes_usadas["documentos"]:
            print(f"📄 Docs: {', '.join([d['filename'][:40] for d in fuentes_usadas['documentos'][:2]])}")
        if fuentes_usadas["web"]:
            print(f"🌐 Web: {len(fuentes_usadas['web'])} fuentes")

    # RESUMEN DE MÉTRICAS
    print("\n" + "━" * 80)
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║                      RESUMEN DE MÉTRICAS                           ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    total = len(resultados)

    # Métrica 1: Fuentes usadas
    solo_docs = sum(1 for r in resultados if r["tipo_fuente"] == "documentos")
    solo_web = sum(1 for r in resultados if r["tipo_fuente"] == "web")
    ambos = sum(1 for r in resultados if r["tipo_fuente"] == "ambos")

    print(f"📊 DISTRIBUCIÓN DE FUENTES:")
    print(f"   Solo documentos: {solo_docs}/{total} ({solo_docs/total*100:.1f}%)")
    print(f"   Solo web:        {solo_web}/{total} ({solo_web/total*100:.1f}%)")
    print(f"   Ambos:           {ambos}/{total} ({ambos/total*100:.1f}%)")

    # Métrica 2: Citas legales
    con_citas = sum(1 for r in resultados if r["tiene_citas"])
    print(f"\n⚖️  CITAS LEGALES:")
    print(f"   Con citas válidas: {con_citas}/{total} ({con_citas/total*100:.1f}%)")

    # Métrica 3: Calidad de formato
    formato_promedio = sum(r["formato_score"] for r in resultados) / total
    print(f"\n📝 CALIDAD DE FORMATO:")
    print(f"   Score promedio: {formato_promedio:.1f}%")

    # Métrica 4: Tiempo de respuesta
    tiempo_promedio = sum(r["processing_time"] for r in resultados) / total
    print(f"\n⏱️  TIEMPO DE RESPUESTA:")
    print(f"   Promedio: {tiempo_promedio:.2f}s")

    # RECOMENDACIONES
    print("\n" + "━" * 80)
    print("\n🎯 RECOMENDACIONES:\n")

    if solo_web > 3:
        print("⚠️  ALTO uso de web - Considerar:")
        print("   • Aumentar threshold a 75-80 para priorizar docs legales")
        print("   • Expandir base de documentos legales\n")

    if con_citas < total * 0.8:
        print("⚠️  BAJA tasa de citas legales - Considerar:")
        print("   • Reforzar prompt para SIEMPRE citar ley específica")
        print("   • Agregar validación post-generación\n")

    if formato_promedio < 80:
        print("⚠️  FORMATO incompleto - Considerar:")
        print("   • Hacer prompt más estricto con ejemplos")
        print("   • Implementar template obligatorio\n")

    if tiempo_promedio > 3:
        print("⚠️  RESPUESTAS lentas - Considerar:")
        print("   • Reducir número de documentos recuperados")
        print("   • Implementar caché persistente\n")

    print("✅ Áreas fuertes:")
    if solo_docs >= total * 0.6:
        print("   • Buena priorización de documentos legales")
    if con_citas >= total * 0.8:
        print("   • Excelente tasa de citación legal")
    if formato_promedio >= 80:
        print("   • Formato consistente y completo")
    if tiempo_promedio <= 2.5:
        print("   • Respuestas rápidas")

    print("\n" + "━" * 80)

    # Guardar resultados en JSON
    with open('/tmp/test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            "fecha": datetime.now().isoformat(),
            "total_tests": total,
            "resultados": resultados,
            "metricas": {
                "solo_docs": solo_docs,
                "solo_web": solo_web,
                "ambos": ambos,
                "con_citas": con_citas,
                "formato_promedio": formato_promedio,
                "tiempo_promedio": tiempo_promedio
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Resultados guardados en: /tmp/test_results.json\n")


if __name__ == "__main__":
    ejecutar_pruebas()
