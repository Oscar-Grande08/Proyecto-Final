#!/usr/bin/env python3
"""
detector_main.py - Sistema de Detección Principal
Detecta: PERSONAS (con velocidad) y EQUIPOS ELECTRÓNICOS (con modelo entrenado)
Autor: Sistema de Detección Inteligente
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from collections import deque
import threading
from queue import Queue, Empty
import os

# Configuración
cv2.setNumThreads(4)

class EquipmentDetector:
    """
    Detector de equipos electrónicos usando modelo entrenado
    """
    
    def __init__(self, model_path='entrenamiento/modelo_equipo_electrico.h5'):
        print("📦 Cargando modelo de equipos electrónicos...")
        
        # Clases de equipos (según tu dataset)
        self.classes = [
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
        
        # Colores para cada clase
        self.colors = {
            'capacitor': (255, 200, 0),
            'cautin': (255, 100, 50),
            'fuente_poder': (100, 255, 100),
            'generador': (255, 100, 100),
            'motor': (0, 255, 255),
            'multimetro': (255, 165, 0),
            'osciloscopio': (0, 255, 0),
            'pinzas': (255, 0, 255),
            'protoboard': (255, 255, 0),
            'transformador': (150, 100, 255)
        }
        
        # Cargar modelo
        self.model = None
        self.input_size = (224, 224)  # Tamaño estándar
        
        if os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"✓ Modelo cargado: {model_path}")
                # Obtener tamaño de entrada del modelo
                input_shape = self.model.input_shape
                self.input_size = (input_shape[1], input_shape[2])
                print(f"✓ Tamaño de entrada: {self.input_size}")
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}")
                print("⚠️ Se usará detección por contornos")
        else:
            print(f"⚠️ Modelo no encontrado: {model_path}")
            print("⚠️ Se usará detección por contornos")
        
        # Background subtractor para detectar objetos
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=30, detectShadows=True
        )
        
        # Parámetros de detección - MÁS SENSIBLES
        self.min_area = 800  # Reducido para detectar objetos más pequeños
        self.max_area = 150000  # Aumentado para objetos grandes
        self.confidence_threshold = 0.3  # Reducido para más detecciones
    
    def detect_regions(self, frame):
        """
        Detecta regiones de interés (ROI) donde pueden estar los equipos
        MEJORADO: Múltiples métodos de detección
        """
        regions = []
        h, w = frame.shape[:2]
        
        # MÉTODO 1: Detección por bordes (Canny)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)  # Umbral más bajo
        
        # Dilatar más agresivamente
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        dilated = cv2.erode(dilated, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Verificar aspect ratio razonable
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Evitar líneas muy delgadas
                    # Expandir región
                    margin = 20
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w_box = min(w - x, w_box + 2*margin)
                    h_box = min(h - y, h_box + 2*margin)
                    
                    regions.append((x, y, w_box, h_box))
        
        # MÉTODO 2: Sliding window si hay pocas detecciones
        if len(regions) < 3:
            # Crear grid de regiones
            grid_size = 200
            overlap = 50
            
            for y in range(0, h - grid_size, grid_size - overlap):
                for x in range(0, w - grid_size, grid_size - overlap):
                    # Verificar si hay contenido interesante
                    roi = gray[y:y+grid_size, x:x+grid_size]
                    if roi.std() > 15:  # Hay suficiente variación
                        regions.append((x, y, grid_size, grid_size))
        
        return regions
    
    def classify_region(self, frame, x, y, w, h):
        """
        Clasifica una región usando el modelo entrenado
        """
        if self.model is None:
            return None, 0.0
        
        try:
            # Extraer ROI
            roi = frame[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None, 0.0
            
            # Preprocesar
            roi_resized = cv2.resize(roi, self.input_size)
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            # Predicción
            predictions = self.model.predict(roi_batch, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            if confidence > self.confidence_threshold:
                class_name = self.classes[class_idx]
                return class_name, float(confidence)
            
        except Exception as e:
            print(f"⚠️ Error clasificando región: {e}")
        
        return None, 0.0
    
    def detect(self, frame):
        """
        Detecta equipos electrónicos en el frame
        """
        detections = []
        
        # Detectar regiones
        regions = self.detect_regions(frame)
        
        # Si no hay modelo, usar clasificación básica por geometría
        if self.model is None:
            for (x, y, w, h) in regions:
                area = w * h
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Clasificación simple por tamaño
                if area > 20000:
                    class_name = 'OSCILOSCOPIO'
                    color = self.colors['osciloscopio']
                elif area > 10000:
                    class_name = 'GENERADOR'
                    color = self.colors['generador']
                elif aspect_ratio > 1.5:
                    class_name = 'PROTOBOARD'
                    color = self.colors['protoboard']
                else:
                    class_name = 'MULTIMETRO'
                    color = self.colors['multimetro']
                
                detections.append({
                    'type': class_name,
                    'bbox': (x, y, x+w, y+h),
                    'center': (x + w//2, y + h//2),
                    'color': color,
                    'confidence': 0.6
                })
        else:
            # Clasificar cada región con el modelo
            for (x, y, w, h) in regions:
                class_name, confidence = self.classify_region(frame, x, y, w, h)
                
                if class_name:
                    color = self.colors.get(class_name, (200, 200, 200))
                    
                    detections.append({
                        'type': class_name.upper().replace('_', ' '),
                        'bbox': (x, y, x+w, y+h),
                        'center': (x + w//2, y + h//2),
                        'color': color,
                        'confidence': confidence
                    })
        
        return detections


class PersonDetector:
    """
    Detector de personas usando MediaPipe
    """
    
    def __init__(self):
        print("📦 Inicializando MediaPipe...")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        print("✓ MediaPipe inicializado")
    
    def detect(self, frame):
        """
        Detecta personas en el frame
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        detections = []
        
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Dibujar esqueleto
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 200, 200), thickness=2
                )
            )
            
            # Calcular bounding box
            landmarks = pose_results.pose_landmarks.landmark
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x1 = max(0, int(min(x_coords)) - 10)
            y1 = max(0, int(min(y_coords)) - 10)
            x2 = min(w, int(max(x_coords)) + 10)
            y2 = min(h, int(max(y_coords)) + 10)
            
            detections.append({
                'type': 'PERSONA',
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'color': (0, 255, 255),
                'confidence': 1.0
            })
        
        return detections


class SpeedTracker:
    """
    Analizador de velocidad usando tracking de objetos
    """
    
    def __init__(self, pixels_per_meter=40, fps=30):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.tracked_objects = {}
        self.next_id = 0
        self.lock = threading.Lock()
        self.max_distance = 120
        self.history_length = 12
        
        print(f"📊 SpeedTracker inicializado (escala: {pixels_per_meter} px/m)")
    
    def calculate_speeds(self, detections):
        """
        Calcula velocidades de objetos detectados
        """
        with self.lock:
            speeds = {}
            current_detections = [(d['center'], d) for d in detections]
            matched_ids = set()
            
            # Asociar detecciones
            for center, detection in current_detections:
                best_id = self._find_best_match(center, matched_ids)
                
                if best_id is None:
                    best_id = self._create_new_track()
                
                matched_ids.add(best_id)
                self._update_track(best_id, center, detection)
                
                # Calcular velocidad
                speed_data = self._compute_speed(best_id, detection)
                if speed_data:
                    speeds[best_id] = speed_data
            
            # Limpiar tracks viejos
            self._cleanup_old_tracks(matched_ids)
            
            return speeds
    
    def _find_best_match(self, center, matched_ids):
        best_id = None
        min_dist = self.max_distance
        
        for obj_id, track_data in self.tracked_objects.items():
            if obj_id in matched_ids:
                continue
            
            if len(track_data['positions']) > 0:
                last_pos = track_data['positions'][-1]
                dist = np.linalg.norm(np.array(center) - np.array(last_pos))
                
                if dist < min_dist:
                    min_dist = dist
                    best_id = obj_id
        
        return best_id
    
    def _create_new_track(self):
        track_id = self.next_id
        self.next_id += 1
        self.tracked_objects[track_id] = {
            'positions': deque(maxlen=self.history_length),
            'timestamps': deque(maxlen=self.history_length),
            'type': None
        }
        return track_id
    
    def _update_track(self, track_id, center, detection):
        self.tracked_objects[track_id]['positions'].append(center)
        self.tracked_objects[track_id]['timestamps'].append(time.time())
        self.tracked_objects[track_id]['type'] = detection['type']
    
    def _compute_speed(self, track_id, detection):
        track = self.tracked_objects[track_id]
        
        if len(track['positions']) < 4:
            return None
        
        positions = list(track['positions'])
        timestamps = list(track['timestamps'])
        
        n_samples = min(6, len(positions) - 1)
        total_distance = 0
        
        for i in range(len(positions) - n_samples, len(positions)):
            if i > 0:
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += np.sqrt(dx * dx + dy * dy)
        
        total_time = timestamps[-1] - timestamps[-n_samples]
        
        if total_time <= 0:
            return None
        
        pixels_per_sec = total_distance / total_time
        meters_per_sec = pixels_per_sec / self.pixels_per_meter
        kmh = meters_per_sec * 3.6
        kmh = min(kmh, 50.0)
        
        return {
            'speed_kmh': kmh,
            'speed_ms': meters_per_sec,
            'type': detection['type'],
            'bbox': detection['bbox'],
            'color': detection['color'],
            'confidence': detection.get('confidence', 1.0)
        }
    
    def _cleanup_old_tracks(self, matched_ids):
        to_remove = [tid for tid in self.tracked_objects.keys() if tid not in matched_ids]
        for tid in to_remove:
            del self.tracked_objects[tid]


class DetectionSystem:
    """
    Sistema principal que combina todos los detectores
    """
    
    def __init__(self, model_path='entrenamiento/modelo_equipo_electrico.h5'):
        print("\n" + "="*70)
        print("SISTEMA DE DETECCIÓN Y VELOCIDAD")
        print("="*70)
        print("\n🔧 Componentes:")
        print("  • MediaPipe: Detección de personas")
        print("  • CNN Entrenada: Clasificación de equipos electrónicos")
        print("  • OpenCV: Análisis de velocidad y tracking")
        print("  • Threading: Procesamiento paralelo")
        
        # Inicializar detectores
        self.equipment_detector = EquipmentDetector(model_path)
        self.person_detector = PersonDetector()
        self.speed_tracker = SpeedTracker(pixels_per_meter=40, fps=30)
        
        self.window_name = 'Sistema_Deteccion_Velocidad'
        
        print("\n✓ Sistema inicializado correctamente")
    
    def draw_ui(self, frame, speeds, fps, person_count, equipment_count):
        """
        Dibuja la interfaz de usuario con información detallada
        """
        h, w = frame.shape[:2]
        
        # Dibujar detecciones con velocidad
        for obj_id, data in speeds.items():
            x1, y1, x2, y2 = data['bbox']
            color = data['color']
            obj_type = data['type']
            speed_kmh = data['speed_kmh']
            confidence = data['confidence']
            
            # Bounding box más grueso
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Label con velocidad
            if obj_type == 'PERSONA':
                label = f"PERSONA: {speed_kmh:.1f} km/h"
            else:
                label = f"{obj_type}: {speed_kmh:.1f} km/h"
            
            # Fondo del texto
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-th-18), (x1+tw+15, y1), color, -1)
            
            # Texto principal
            cv2.putText(frame, label, (x1+5, y1-10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
            # ID y confianza debajo del box
            info = f"ID:{obj_id} | Conf:{confidence:.0%}"
            cv2.putText(frame, info, (x1, y2+22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Panel de información detallado
        panel_w, panel_h = 500, 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Título
        cv2.putText(frame, "=== SISTEMA DE DETECCION SIMULTANEA ===", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        
        # Información detallada
        info_lines = [
            f"PERSONAS detectadas: {person_count}",
            f"EQUIPOS detectados: {equipment_count}",
            f"Tracks con velocidad: {len(speeds)}",
            f"FPS: {fps:.1f}",
        ]
        
        y_pos = 75
        for line in info_lines:
            cv2.putText(frame, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y_pos += 28
        
        # Instrucciones
        cv2.putText(frame, "q/ESC: Salir | f: Ventana | m: Pantalla completa", (20, panel_h-12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        # Indicador de estado (esquina superior derecha)
        status_text = "ACTIVO"
        (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        cv2.rectangle(frame, (w-stw-25, 10), (w-10, 45), (0, 255, 0), -1)
        cv2.putText(frame, status_text, (w-stw-20, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def run(self, video_source=0):
        """
        Ejecuta el sistema de detección
        """
        print("\n" + "="*70)
        print("INICIANDO DETECCIÓN")
        print("="*70)
        print(f"\n📹 Abriendo cámara: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ ERROR: No se pudo abrir la cámara")
            return
        
        # Configurar cámara con mayor resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Cámara abierta")
        print("\n💡 Instrucciones:")
        print("  • Muévete frente a la cámara (PERSONA)")
        print("  • Muestra equipos electrónicos al encuadre")
        print("  • El sistema detectará y calculará velocidades")
        print("  • Presiona 'q' o ESC para salir")
        print("  • Presiona 'f' para ventana normal / 'm' para pantalla completa")
        print("\n▶️  Iniciando...\n")
        
        # Crear ventana maximizada
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # O usar tamaño grande fijo
        # cv2.resizeWindow(self.window_name, 1280, 720)
        
        fps_counter = deque(maxlen=30)
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ No se pudo leer frame")
                    break
                
                frame_count += 1
                
                # DETECCIÓN SIMULTÁNEA
                # 1. Detectar PERSONAS (cada frame)
                person_detections = self.person_detector.detect(frame.copy())
                
                # 2. Detectar EQUIPOS (cada frame para mejor detección)
                equipment_detections = self.equipment_detector.detect(frame.copy())
                
                # 3. Combinar TODAS las detecciones
                all_detections = person_detections + equipment_detections
                
                # VISUALIZACIÓN DE DETECCIONES CRUDAS (sin tracking)
                # Esto te permite ver si el sistema está detectando correctamente
                for det in all_detections:
                    x1, y1, x2, y2 = det['bbox']
                    # Dibujar contorno delgado para mostrar detección cruda
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                
                # 4. Calcular velocidades para TODOS los objetos detectados
                speeds = self.speed_tracker.calculate_speeds(all_detections)
                
                # Debug: mostrar en consola cada 30 frames
                if frame_count % 30 == 0:
                    print(f"✓ Frame {frame_count} | Personas: {len(person_detections)} | Equipos: {len(equipment_detections)} | Tracks: {len(speeds)}")
                
                # Calcular FPS
                fps = 1.0 / max(time.time() - start_time, 0.001)
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                
                # Dibujar UI con información detallada
                display_frame = self.draw_ui(
                    frame, 
                    speeds, 
                    avg_fps,
                    len(person_detections),
                    len(equipment_detections)
                )
                
                # Mostrar
                cv2.imshow(self.window_name, display_frame)
                
                # Control de salida y opciones
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n✓ Cerrando sistema...")
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
                elif key == ord('m'):  # Maximizar
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                    
        except KeyboardInterrupt:
            print("\n✓ Interrumpido por usuario")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Sistema finalizado")


def main():
    """
    Función principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Detección y Velocidad')
    parser.add_argument('--model', type=str, 
                       default='entrenamiento/modelo_equipo_electrico.h5',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--video', type=str, default='0',
                       help='Fuente de video (0=webcam, o ruta)')
    
    args = parser.parse_args()
    
    # Convertir '0' a int
    video_source = 0 if args.video == '0' else args.video
    
    print("\n" + "="*70)
    print("DETECTOR PRINCIPAL - SISTEMA INTEGRADO")
    print("="*70)
    
    try:
        system = DetectionSystem(model_path=args.model)
        system.run(video_source=video_source)
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("PROGRAMA FINALIZADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

