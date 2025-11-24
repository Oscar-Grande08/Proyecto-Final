import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

class LabEquipmentClassifier:
    """
    Clasificador de elementos de laboratorio usando MediaPipe
    y análisis de características visuales
    """
    
    def __init__(self, database_path='database/images'):
        self.database_path = database_path
        self.categories = []
        self.reference_features = {}
        
        # Inicializar MediaPipe para detección de objetos
        self.mp_objectron = mp.solutions.objectron
        self.mp_drawing = mp.solutions.drawing_utils
        
        print("🤖 Inicializando clasificador...")
        self.load_database()
    
    def load_database(self):
        """
        Carga la base de datos de imágenes y extrae características
        """
        if not os.path.exists(self.database_path):
            print(f"⚠️  La carpeta {self.database_path} no existe")
            return
        
        print("\n📂 Cargando base de datos...")
        
        # Obtener categorías (carpetas)
        for item in os.listdir(self.database_path):
            category_path = os.path.join(self.database_path, item)
            if os.path.isdir(category_path):
                self.categories.append(item)
                
                # Extraer características de cada categoría
                features = self.extract_category_features(category_path)
                self.reference_features[item] = features
                
                print(f"  ✓ Cargada categoría: {item}")
        
        if len(self.categories) == 0:
            print("  ⚠️  No se encontraron categorías en la base de datos")
        else:
            print(f"\n✓ Base de datos cargada: {len(self.categories)} categorías")
    
    def extract_category_features(self, category_path):
        """
        Extrae características visuales de todas las imágenes de una categoría
        """
        features = {
            'color_histograms': [],
            'edge_density': [],
            'shape_complexity': []
        }
        
        # Procesar cada imagen en la categoría
        for filename in os.listdir(category_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Extraer características
                    features['color_histograms'].append(
                        self.calculate_color_histogram(img)
                    )
                    features['edge_density'].append(
                        self.calculate_edge_density(img)
                    )
                    features['shape_complexity'].append(
                        self.calculate_shape_complexity(img)
                    )
        
        return features
    
    def calculate_color_histogram(self, image):
        """
        Calcula el histograma de colores de una imagen
        """
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calcular histograma para cada canal
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalizar
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenar histogramas
        return np.concatenate([hist_h, hist_s, hist_v])
    
    def calculate_edge_density(self, image):
        """
        Calcula la densidad de bordes en la imagen
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density
    
    def calculate_shape_complexity(self, image):
        """
        Calcula la complejidad de las formas en la imagen
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return 0
        
        # Complejidad basada en el número de contornos y perímetro
        complexity = len(contours) + sum(cv2.arcLength(c, True) for c in contours) / 1000
        return complexity
    
    def classify_image(self, image_path):
        """
        Clasifica una imagen de equipo de laboratorio
        """
        if len(self.categories) == 0:
            return None, 0.0, "No hay categorías en la base de datos"
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0, "No se pudo cargar la imagen"
        
        # Extraer características de la imagen de prueba
        test_hist = self.calculate_color_histogram(img)
        test_edge = self.calculate_edge_density(img)
        test_shape = self.calculate_shape_complexity(img)
        
        # Comparar con cada categoría
        best_match = None
        best_score = -1
        scores = {}
        
        for category, features in self.reference_features.items():
            if len(features['color_histograms']) == 0:
                continue
            
            # Calcular similitud promedio
            color_scores = []
            for ref_hist in features['color_histograms']:
                # Correlación de histogramas
                score = cv2.compareHist(
                    test_hist, ref_hist, cv2.HISTCMP_CORREL
                )
                color_scores.append(score)
            
            # Similitud de bordes
            edge_scores = [
                1 - abs(test_edge - ref_edge) 
                for ref_edge in features['edge_density']
            ]
            
            # Similitud de formas
            shape_scores = [
                1 / (1 + abs(test_shape - ref_shape))
                for ref_shape in features['shape_complexity']
            ]
            
            # Puntuación combinada con mejor balance
            avg_color = np.mean(color_scores) if color_scores else 0
            avg_edge = np.mean(edge_scores) if edge_scores else 0
            avg_shape = np.mean(shape_scores) if shape_scores else 0
            
            # Usar el mejor score individual también (no solo promedio)
            max_color = max(color_scores) if color_scores else 0
            max_edge = max(edge_scores) if edge_scores else 0
            max_shape = max(shape_scores) if shape_scores else 0
            
            # Peso ponderado mejorado
            avg_score = (0.4 * avg_color + 0.3 * avg_edge + 0.3 * avg_shape)
            max_score = (0.4 * max_color + 0.3 * max_edge + 0.3 * max_shape)
            
            # Combinar promedio y máximo
            final_score = (0.7 * avg_score + 0.3 * max_score)
            scores[category] = final_score
            
            if final_score > best_score:
                best_score = final_score
                best_match = category
        
        # Convertir score a porcentaje
        confidence = min(100, max(0, best_score * 100))
        
        return best_match, confidence, scores
    
    def classify_with_visualization(self, image_path, show=True):
        """
        Clasifica y muestra el resultado visualmente
        """
        category, confidence, scores = self.classify_image(image_path)
        
        # Cargar imagen para visualización
        img = cv2.imread(image_path)
        if img is None:
            print("Error al cargar la imagen")
            return
        
        # Redimensionar para visualización
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # Agregar texto con el resultado
        if category:
            text = f"Categoria: {category.replace('_', ' ').upper()}"
            conf_text = f"Confianza: {confidence:.1f}%"
            
            # Fondo para el texto
            cv2.rectangle(img, (10, 10), (500, 100), (0, 0, 0), -1)
            
            # Texto principal
            cv2.putText(img, text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, conf_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if show:
            cv2.imshow('Clasificacion de Equipo', img)
            print("\n📊 Resultados de clasificación:")
            print(f"  Categoría: {category}")
            print(f"  Confianza: {confidence:.1f}%")
            print("\n  Puntuaciones por categoría:")
            for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {cat}: {score*100:.1f}%")
            print("\nPresiona cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return category, confidence, img


def main():
    """
    Función de prueba del clasificador
    """
    print("="*60)
    print("🔬 CLASIFICADOR DE EQUIPOS DE LABORATORIO")
    print("="*60)
    
    # Crear clasificador
    classifier = LabEquipmentClassifier()
    
    if len(classifier.categories) == 0:
        print("\n⚠️  Primero debes crear la base de datos")
        print("Ejecuta: python src/web_scraper.py")
        return
    
    print("\n✓ Clasificador listo para usar")
    print("\nPara clasificar una imagen, usa:")
    print("  classifier.classify_with_visualization('ruta/imagen.jpg')")


if __name__ == "__main__":
    main()