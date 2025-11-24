import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

class LabEquipmentScraper:
    def __init__(self, output_folder='database/images'):
        """
        Inicializa el scraper para obtener imágenes de equipos de laboratorio
        """
        self.output_folder = output_folder
        self.create_folders()
        
        # Categorías de equipos que vamos a buscar (10 elementos)
        self.categories = {
            'capacitor': 'capacitor electronics',
            'cautin': 'soldering iron',
            'fuente_poder': 'power supply electronics',
            'generador': 'function generator electronics',
            'motor': 'electric motor',
            'multimetro': 'multimeter electronics',
            'osciloscopio': 'oscilloscope electronics',
            'pinzas': 'clamp meter',
            'protoboard': 'breadboard electronics',
            'transformador': 'transformer electronics',
        }
    
    def create_folders(self):
        """Crea las carpetas necesarias"""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"✓ Carpeta creada: {self.output_folder}")
    
    def download_sample_images(self):
        """
        Descarga imágenes de ejemplo usando URLs públicas
        """
        # URLs de ejemplo de imágenes libres de uso
        sample_urls = {
            'multimetro': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Digital_Multimeter_Pocket-Size.jpg/320px-Digital_Multimeter_Pocket-Size.jpg',
            ],
            'osciloscopio': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Oscilloscope_at_work.jpg/320px-Oscilloscope_at_work.jpg',
            ],
            'motor': [
                'https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Electric_motor.jpg/320px-Electric_motor.jpg',
            ]
        }
        
        print("\n🔄 Descargando imágenes de ejemplo...")
        
        for category, urls in sample_urls.items():
            category_folder = os.path.join(self.output_folder, category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)
            
            for idx, url in enumerate(urls):
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        filename = f"{category}_{idx+1}.jpg"
                        filepath = os.path.join(category_folder, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"  ✓ Descargada: {filename}")
                        time.sleep(1)  # Pausa para no saturar el servidor
                    
                except Exception as e:
                    print(f"  ✗ Error descargando {url}: {str(e)}")
        
        print("\n✓ Proceso de descarga completado")
    
    def create_manual_database_instructions(self):
        """
        Imprime instrucciones para crear la base de datos manualmente
        """
        print("\n" + "="*60)
        print("📸 INSTRUCCIONES PARA CREAR TU BASE DE DATOS")
        print("="*60)
        print("\nOpción 1: Buscar imágenes en Google")
        print("  1. Busca cada equipo en Google Imágenes")
        print("  2. Descarga 5-10 imágenes de cada uno")
        print("  3. Guárdalas en: database/images/[nombre_equipo]/")
        
        print("\nEquipos a buscar:")
        for category, search_term in self.categories.items():
            print(f"  • {category.replace('_', ' ').title()}")
            print(f"    Carpeta: database/images/{category}/")
        
        print("\nOpción 2: Tomar fotos propias")
        print("  1. Ve al laboratorio de tu universidad")
        print("  2. Toma fotos de los equipos disponibles")
        print("  3. Organízalas en las carpetas correspondientes")
        
        print("\n" + "="*60)
    
    def check_database_status(self):
        """
        Verifica el estado de la base de datos
        """
        print("\n📊 Estado de la Base de Datos:")
        print("-" * 40)
        
        total_images = 0
        for category in self.categories.keys():
            category_path = os.path.join(self.output_folder, category)
            if os.path.exists(category_path):
                images = [f for f in os.listdir(category_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                total_images += count
                status = "✓" if count > 0 else "✗"
                print(f"{status} {category.replace('_', ' ').title()}: {count} imágenes")
            else:
                print(f"✗ {category.replace('_', ' ').title()}: Carpeta no existe")
        
        print("-" * 40)
        print(f"Total: {total_images} imágenes")
        
        if total_images == 0:
            print("\n⚠️  No hay imágenes en la base de datos")
            print("Ejecuta create_manual_database_instructions() para ver cómo agregarlas")
        
        return total_images > 0


# Función para ejecutar el scraper
def main():
    scraper = LabEquipmentScraper()
    
    print("="*60)
    print("🔬 LABORATORIO - WEB SCRAPER")
    print("="*60)
    
    # Descargar algunas imágenes de ejemplo
    scraper.download_sample_images()
    
    # Mostrar instrucciones para completar la base de datos
    scraper.create_manual_database_instructions()
    
    # Verificar el estado
    scraper.check_database_status()


if __name__ == "__main__":
    main()