"""
Script para corregir la estructura de carpetas de la base de datos
Ejecuta: python arreglar_carpetas.py
"""

import os
import shutil

def arreglar_estructura():
    print("="*60)
    print("🔧 ARREGLANDO ESTRUCTURA DE CARPETAS")
    print("="*60)
    
    base_path = 'database/images'
    
    # Carpetas que debe tener el proyecto (10 elementos)
    carpetas_necesarias = [
        'capacitor',
        'cautin',
        'fuente_poder',
        'generador',
        'motor',
        'multimetro',
        'osciloscopio',
        'pinzas',
        'protoboard',
        'transformador'
    ]
    
    # Renombrar carpeta incorrecta si existe
    carpeta_incorrecta = os.path.join(base_path, 'multímetro')
    carpeta_correcta = os.path.join(base_path, 'multimetro')
    
    if os.path.exists(carpeta_incorrecta):
        print("\n📝 Renombrando 'multímetro' a 'multimetro'...")
        try:
            # Mover contenido si la carpeta correcta ya existe
            if os.path.exists(carpeta_correcta):
                for item in os.listdir(carpeta_incorrecta):
                    src = os.path.join(carpeta_incorrecta, item)
                    dst = os.path.join(carpeta_correcta, item)
                    shutil.move(src, dst)
                os.rmdir(carpeta_incorrecta)
            else:
                os.rename(carpeta_incorrecta, carpeta_correcta)
            print("  ✓ Renombrado exitosamente")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Crear carpetas faltantes
    print("\n📁 Verificando carpetas necesarias...")
    for carpeta in carpetas_necesarias:
        carpeta_path = os.path.join(base_path, carpeta)
        
        if os.path.exists(carpeta_path):
            # Contar imágenes
            imagenes = [f for f in os.listdir(carpeta_path) 
                       if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
            print(f"  ✓ {carpeta:20} ({len(imagenes)} imágenes)")
        else:
            # Crear carpeta
            os.makedirs(carpeta_path, exist_ok=True)
            print(f"  + {carpeta:20} (creada - 0 imágenes)")
    
    # Resumen
    print("\n" + "="*60)
    print("📊 ESTADO ACTUAL")
    print("="*60)
    
    total_imagenes = 0
    carpetas_vacias = []
    
    for carpeta in carpetas_necesarias:
        carpeta_path = os.path.join(base_path, carpeta)
        imagenes = [f for f in os.listdir(carpeta_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        count = len(imagenes)
        total_imagenes += count
        
        if count == 0:
            carpetas_vacias.append(carpeta)
            status = "⚠️  VACÍA"
        elif count < 5:
            status = f"🟡 {count} imágenes (necesitas más)"
        else:
            status = f"✅ {count} imágenes"
        
        print(f"{status:40} {carpeta.replace('_', ' ').title()}")
    
    print("-" * 60)
    print(f"Total: {total_imagenes} imágenes")
    
    # Recomendaciones
    if carpetas_vacias:
        print("\n" + "="*60)
        print("📋 PRÓXIMOS PASOS")
        print("="*60)
        print(f"\n⚠️  Carpetas vacías: {len(carpetas_vacias)}")
        print("\nNecesitas agregar imágenes a:")
        for carpeta in carpetas_vacias:
            print(f"  • database/images/{carpeta}/")
        
        print("\n💡 Cómo agregar imágenes:")
        print("  1. Busca cada equipo en Google Imágenes")
        print("  2. Descarga 5-10 imágenes de cada uno")
        print("  3. Guárdalas en la carpeta correspondiente")
        print("\n  O ejecuta: python descargar_imagenes.py")
    else:
        print("\n✅ ¡Todas las carpetas tienen imágenes!")
    
    print("\n" + "="*60)
    print("✨ Estructura corregida. Ahora puedes ejecutar:")
    print("   python main.py")
    print("="*60)


if __name__ == "__main__":
    arreglar_estructura()