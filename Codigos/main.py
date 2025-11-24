"""
Sistema de Clasificación de Equipos de Laboratorio
Proyecto Tercer Corte - Ingeniería Electrónica
"""

import os
import sys
from pathlib import Path

# Agregar directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from web_scraper import LabEquipmentScraper
from classifier import LabEquipmentClassifier


def print_menu():
    """Imprime el menú principal"""
    print("\n" + "="*60)
    print("🔬 SISTEMA CLASIFICADOR DE EQUIPOS DE LABORATORIO")
    print("="*60)
    print("\n1. 📥 Configurar Base de Datos (Web Scraping)")
    print("2. 🤖 Entrenar/Cargar Clasificador")
    print("3. 🔍 Clasificar una Imagen")
    print("4. 📊 Ver Estado del Sistema")
    print("5. 📸 Clasificar desde Webcam (Próximamente)")
    print("6. ❌ Salir")
    print("\n" + "="*60)


def setup_database():
    """Configura la base de datos de imágenes"""
    print("\n🔄 Configurando base de datos...")
    scraper = LabEquipmentScraper()
    
    # Descargar imágenes de ejemplo
    scraper.download_sample_images()
    
    # Mostrar instrucciones
    scraper.create_manual_database_instructions()
    
    # Verificar estado
    has_images = scraper.check_database_status()
    
    if has_images:
        print("\n✓ Base de datos configurada correctamente")
    else:
        print("\n⚠️  Necesitas agregar más imágenes manualmente")
    
    input("\nPresiona Enter para continuar...")


def train_classifier():
    """Entrena/carga el clasificador"""
    print("\n🤖 Cargando clasificador...")
    
    try:
        classifier = LabEquipmentClassifier()
        
        if len(classifier.categories) == 0:
            print("\n❌ No hay categorías en la base de datos")
            print("   Primero debes configurar la base de datos (Opción 1)")
            return None
        
        print(f"\n✓ Clasificador cargado exitosamente")
        print(f"   Categorías disponibles: {len(classifier.categories)}")
        return classifier
        
    except Exception as e:
        print(f"\n❌ Error al cargar el clasificador: {str(e)}")
        return None


def classify_single_image(classifier):
    """Clasifica una sola imagen"""
    if classifier is None:
        print("\n❌ Primero debes cargar el clasificador (Opción 2)")
        input("Presiona Enter para continuar...")
        return
    
    print("\n🔍 Clasificar Imagen")
    print("-" * 40)
    print("Opciones:")
    print("1. Usar imagen de la base de datos")
    print("2. Especificar ruta de imagen")
    print("3. Volver al menú principal")
    
    option = input("\nElige una opción (1-3): ").strip()
    
    if option == "1":
        # Listar imágenes disponibles
        print("\nCategorías disponibles:")
        for idx, cat in enumerate(classifier.categories, 1):
            print(f"{idx}. {cat.replace('_', ' ').title()}")
        
        try:
            cat_idx = int(input("\nElige una categoría: ")) - 1
            if 0 <= cat_idx < len(classifier.categories):
                category = classifier.categories[cat_idx]
                cat_path = os.path.join('database/images', category)
                
                images = [f for f in os.listdir(cat_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if images:
                    print(f"\nImágenes en {category}:")
                    for idx, img in enumerate(images, 1):
                        print(f"{idx}. {img}")
                    
                    img_idx = int(input("\nElige una imagen: ")) - 1
                    if 0 <= img_idx < len(images):
                        image_path = os.path.join(cat_path, images[img_idx])
                        
                        print("\n🔄 Clasificando...")
                        classifier.classify_with_visualization(image_path)
                else:
                    print("No hay imágenes en esta categoría")
        except (ValueError, IndexError):
            print("❌ Opción inválida")
    
    elif option == "2":
        image_path = input("\nIngresa la ruta de la imagen: ").strip()
        
        if os.path.exists(image_path):
            print("\n🔄 Clasificando...")
            classifier.classify_with_visualization(image_path)
        else:
            print("❌ La imagen no existe")
    
    input("\nPresiona Enter para continuar...")


def show_system_status():
    """Muestra el estado del sistema"""
    print("\n📊 ESTADO DEL SISTEMA")
    print("="*60)
    
    # Estado de la base de datos
    scraper = LabEquipmentScraper()
    has_images = scraper.check_database_status()
    
    # Estado del clasificador
    print("\n🤖 Estado del Clasificador:")
    try:
        classifier = LabEquipmentClassifier()
        print(f"  ✓ Funcionando correctamente")
        print(f"  ✓ Categorías cargadas: {len(classifier.categories)}")
        
        for cat in classifier.categories:
            print(f"    • {cat.replace('_', ' ').title()}")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "="*60)
    input("\nPresiona Enter para continuar...")


def main():
    """Función principal del programa"""
    classifier = None
    
    while True:
        print_menu()
        
        try:
            choice = input("Elige una opción (1-6): ").strip()
            
            if choice == "1":
                setup_database()
            
            elif choice == "2":
                classifier = train_classifier()
                if classifier:
                    input("\nPresiona Enter para continuar...")
            
            elif choice == "3":
                classify_single_image(classifier)
            
            elif choice == "4":
                show_system_status()
            
            elif choice == "5":
                print("\n🚧 Función en desarrollo...")
                print("Esta función permitirá clasificar objetos en tiempo real")
                input("\nPresiona Enter para continuar...")
            
            elif choice == "6":
                print("\n👋 ¡Hasta luego!")
                print("Proyecto desarrollado para Tercer Corte")
                break
            
            else:
                print("\n❌ Opción inválida. Intenta de nuevo.")
                input("Presiona Enter para continuar...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {str(e)}")
            input("Presiona Enter para continuar...")


if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('database'):
        os.makedirs('database/images', exist_ok=True)
    
    if not os.path.exists('src'):
        os.makedirs('src', exist_ok=True)
    
    main()
