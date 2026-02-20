import os
import sys
import argparse
from pinecone import Pinecone
from dotenv import load_dotenv, find_dotenv

# Try to load env vars
load_dotenv("config/config.env")
load_dotenv(find_dotenv())

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatfj-legal-index")
NAMESPACE = "corrections"

def get_index():
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY no encontrada.")
        sys.exit(1)
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(name=PINECONE_INDEX_NAME)

def list_corrections(index):
    print(f"\nObteniendo lista de correcciones de Pinecone ({NAMESPACE})...")
    # Generar vector dummy porque requerimos el index dimension (1024 en este caso)
    dummy_vector = [0.0] * 1024
    
    try:
        response = index.query(
            vector=dummy_vector,
            top_k=100, 
            include_metadata=True,
            namespace=NAMESPACE
        )
        
        matches = response.get("matches", [])
        if not matches:
            print("No se encontraron aportes.")
            return []
            
        print(f"Se encontraron {len(matches)} aportes:\n")
        for i, match in enumerate(matches, 1):
            metadata = match.get("metadata", {})
            question = metadata.get("original_question", "N/A")
            trainer = metadata.get("trainer", "Desconocido")
            # Truncar textos largos para la vista previa
            q_preview = question[:60] + "..." if len(question) > 60 else question
            
            print(f"[{i}] ID: {match['id']}")
            print(f"    Autor: {trainer}")
            print(f"    Pregunta: {q_preview}")
            print("-" * 50)
            
        return matches
    except Exception as e:
        print(f"❌ Error al consultar Pinecone: {e}")
        return []

def view_correction(matches, index_num):
    try:
        idx = int(index_num) - 1
        if idx < 0 or idx >= len(matches):
            print("❌ Número de aporte inválido.")
            return None
        
        match = matches[idx]
        metadata = match.get("metadata", {})
        
        print("\n" + "="*50)
        print(f"APORTE DETALLADO (ID: {match['id']})")
        print("="*50)
        print(f"Autor : {metadata.get('trainer', 'N/A')}")
        print(f"Fecha : {metadata.get('timestamp', 'N/A')}")
        print(f"Intención : {metadata.get('intent', 'N/A')}")
        print("-"*50)
        print("PREGUNTA ORIGINAL:")
        print(metadata.get('original_question', 'N/A'))
        print("\nRESPUESTA/CORRECCIÓN DEL ABOGADO:")
        print(metadata.get('text', 'N/A'))
        print("="*50 + "\n")
        
        return match
    except ValueError:
        print("❌ Por favor ingresa un número válido.")
        return None

def update_correction(index, match):
    print("\n📝 MODO EDICIÓN")
    print("Deja en blanco y presiona Enter para mantener el texto actual.")
    
    metadata = match.get("metadata", {})
    
    # 1. Editar autor
    current_trainer = metadata.get('trainer', '')
    new_trainer = input(f"Nuevo autor [{current_trainer}]: ").strip()
    if new_trainer:
        metadata['trainer'] = new_trainer
        
    # 2. Editar Pregunta Original
    current_question = metadata.get('original_question', '')
    print(f"\nPregunta original actual:\n{current_question}")
    new_question = input("\nNueva pregunta original: ").strip()
    if new_question:
        metadata['original_question'] = new_question
        
    # 3. Editar Corrección (la respuesta en sí)
    current_text = metadata.get('text', '')
    print(f"\nCorrección actual:\n{current_text}")
    print("\nIntroduce el nuevo texto (o presiona Enter para cancelar la edición de texto):")
    # Para textos largos usamos input sencillo aunque no soporte saltos de línea fácilmente
    # Lo ideal para multilínea en CLI es complicado, probaremos simple.
    new_text = input("> ").strip()
    if new_text:
        metadata['text'] = new_text
        
    # Guardar cambios a Pinecone reescribiendo el ID existente
    try:
        # Recuperamos sus values del metadata actual y valores viejos
        index.upsert(
            vectors=[(match['id'], match['values'], metadata)],
            namespace=NAMESPACE
        )
        print("\n✅ ¡Aporte actualizado exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al actualizar en Pinecone: {e}")

def delete_correction(index, match):
    verify = input(f"\n⚠️ ¿Estás COMPLETAMENTE SEGURO de borrar el aporte '{match['id']}'? (escribe 'si' para confirmar): ")
    if verify.lower() == 'si':
        try:
            index.delete(ids=[match['id']], namespace=NAMESPACE)
            print("✅ ¡Aporte borrado del entrenamiento de manera permanente!")
        except Exception as e:
            print(f"❌ Error al intentar borrar: {e}")
    else:
        print("Operación cancelada.")

def main():
    print("========================================")
    print("🌲 ADMINISTRADOR DE ENTRENAMIENTO CHATFJ")
    print("========================================")
    
    index = get_index()
    matches = []
    
    while True:
        if not matches:
            matches = list_corrections(index)
            if not matches:
                break
                
        print("\nOpciones:")
        print(" [1..N] Escribe el número del aporte para ver detalles/editar/borrar")
        print(" [R] Recargar lista desde Pinecone")
        print(" [Q] Salir")
        
        choice = input("\nElige una opción: ").strip().lower()
        
        if choice == 'q':
            print("¡Hasta luego!")
            break
        elif choice == 'r':
            matches = [] # Forzar recarga
            continue
        elif choice.isdigit():
            match = view_correction(matches, choice)
            if match:
                while True:
                    print("\nAcciones sobre este aporte:")
                    print(" [E] Editar este aporte")
                    print(" [B] Borrar este aporte")
                    print(" [V] Volver a la lista")
                    
                    action = input("¿Qué deseas hacer?: ").strip().lower()
                    
                    if action == 'v':
                        break
                    elif action == 'e':
                        update_correction(index, match)
                        matches = [] # Forzar recarga para la sig vista
                        break
                    elif action == 'b':
                        delete_correction(index, match)
                        matches = [] # Forzar recarga para la sig vista
                        break
                    else:
                        print("Opción inválida.")
        else:
            print("Entrada inválida.")

if __name__ == "__main__":
    main()
