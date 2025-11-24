# main_streamlit.py
# Interfaz Streamlit integrada para: scraping, ETL, clasificación y detección/velocidad en tiempo real.
import streamlit as st
import threading
import time
import cv2
import numpy as np
import os

# Importar tus módulos (asegúrate que estén en la misma carpeta o en PYTHONPATH)
from web_scraper import LabEquipmentScraper
from arreglar_carpetas import arreglar_estructura
from classifier import LabEquipmentClassifier
from detector_main import EquipmentDetector, PersonDetector, SpeedTracker, DetectionSystem

st.set_page_config(page_title="Lab Detector", layout="wide")

st.title("🔬 Plataforma - Reconocimiento de herramientas y análisis de velocidad")
st.sidebar.title("Controles")

# Sidebar: opciones
mode = st.sidebar.radio("Modo", ["Inicio", "Dataset / ETL", "Clasificador", "Detección en vivo"])

# Estado (para compartir objetos entre runs)
if 'detector_thread' not in st.session_state:
    st.session_state.detector_thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'detection_system' not in st.session_state:
    st.session_state.detection_system = None

# -------------------------
# MODO: Inicio
# -------------------------
if mode == "Inicio":
    st.markdown("""
    **Descripción rápida**
    - Usa la pestaña **Dataset / ETL** para crear / validar la base de datos de imágenes.
    - Usa **Clasificador** para ver las categorías y probar el clasificador.
    - Usa **Detección en vivo** para detectar objetos y personas con la cámara y ver velocidades.
    """)
    st.info("Recuerda ejecutar `python main_streamlit.py` o usar Docker con --device /dev/video0 (Linux) para usar la webcam.")

# -------------------------
# MODO: Dataset / ETL
# -------------------------
elif mode == "Dataset / ETL":
    st.header("Dataset / ETL")
    st.write("Aquí puedes crear carpetas, descargar imágenes de ejemplo y verificar el estado del dataset.")
    scraper = LabEquipmentScraper()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Descargar imágenes de ejemplo"):
            with st.spinner("Descargando..."):
                scraper.download_sample_images()
            st.success("Descarga de ejemplo completada.")
    with col2:
        if st.button("🧰 Arreglar estructura de carpetas (ETL básico)"):
            with st.spinner("Arreglando estructura..."):
                arreglar_estructura()
            st.success("Estructura verificada / corregida.")
    st.markdown("---")
    if st.button("🔎 Ver estado base de datos"):
        has = scraper.check_database_status()
        if not has:
            st.warning("No se encontraron imágenes. Añade imágenes a `database/images/<categoria>/`")
        else:
            st.success("Base de datos OK (revisa consola para detalles).")

# -------------------------
# MODO: Clasificador
# -------------------------
elif mode == "Clasificador":
    st.header("Clasificador de herramientas")
    st.write("Carga el clasificador basado en características (classifier.py).")
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### 🔎 Probar clasificación sobre imagen subida")
        uploaded = st.file_uploader("Sube imagen (.jpg/.png)", type=['jpg','jpeg','png'])
        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            tmp_path = "tmp_uploaded.jpg"
            cv2.imwrite(tmp_path, img)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen subida", use_column_width=True)
            if st.button("Clasificar imagen subida"):
                with st.spinner("Clasificando..."):
                    classifier = LabEquipmentClassifier()
                    cat, conf, scores = classifier.classify_image(tmp_path)
                if cat is None:
                    st.error("No se pudo clasificar (¿dataset vacío?). Revisa `database/images/`")
                else:
                    st.success(f"Categoría: {cat} — Confianza: {conf:.1f}%")
                    st.json({k: float(v) for k,v in scores.items()})
    with col2:
        st.markdown("### 📂 Estado del clasificador")
        if st.button("Cargar clasificador"):
            with st.spinner("Cargando..."):
                classifier = LabEquipmentClassifier()
            st.success(f"Categorías encontradas: {len(classifier.categories)}")
            if classifier.categories:
                st.write(classifier.categories)
        st.markdown("### ⚠️ Notas")
        st.write("- El clasificador actual usa histogramas y características sencillas. Para producción usa transferencia (MobileNet/TensorFlow).")

# -------------------------
# MODO: Detección en vivo
# -------------------------
elif mode == "Detección en vivo":
    st.header("Detección en vivo - cámara")
    st.write("Muestra resultados en tiempo real: detección de personas, clasificación de equipos y velocidad.")
    device = st.sidebar.text_input("Fuente de video (0 = webcam)", "0")
    model_path = st.sidebar.text_input("Ruta modelo Keras (.h5) (opcional)", "entrenamiento/modelo_equipo_electrico.h5")

    if st.button("Iniciar detección en vivo"):
        # crear sistema de detección y evento de parada
        st.session_state.stop_event = threading.Event()
        try:
            sys_model_path = model_path if os.path.exists(model_path) else model_path
            st.session_state.detection_system = DetectionSystem(model_path=sys_model_path)
        except Exception as e:
            st.error(f"No se pudo iniciar DetectionSystem: {e}")
            raise

        # placeholder para imagen
        img_placeholder = st.empty()
        info_placeholder = st.empty()

        def capture_loop(stop_event, src):
            # abrir camara
            try:
                src_int = int(src) if str(src).isdigit() else src
            except:
                src_int = src
            cap = cv2.VideoCapture(src_int)
            if not cap.isOpened():
                info_placeholder.error("No se pudo abrir la cámara. Revisa permisos / dispositivo.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            fps_list = []
            frame_count = 0
            system = st.session_state.detection_system
            person_detector = system.person_detector
            equipment_detector = system.equipment_detector
            speed_tracker = system.speed_tracker

            while not stop_event.is_set():
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    info_placeholder.error("Error leyendo frame de la cámara.")
                    break

                frame_count += 1
                # detecciones
                person_detections = person_detector.detect(frame.copy())
                equipment_detections = equipment_detector.detect(frame.copy())
                all_detections = person_detections + equipment_detections
                speeds = speed_tracker.calculate_speeds(all_detections)

                # calcular fps
                fps = 1.0 / max(time.time() - t0, 1e-6)
                fps_list.append(fps)
                avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])

                # dibujar UI con método ya existente
                display = system.draw_ui(frame, speeds, avg_fps, len(person_detections), len(equipment_detections))
                # convertir BGR -> RGB
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img_placeholder.image(display_rgb, use_column_width=True)
                info_placeholder.text(f"FPS aprox: {avg_fps:.1f} | Frames procesados: {frame_count}")

            cap.release()
            info_placeholder.info("Detección detenida.")

        # iniciar hilo
        t = threading.Thread(target=capture_loop, args=(st.session_state.stop_event, device), daemon=True)
        st.session_state.detector_thread = t
        t.start()

    if st.button("Detener detección"):
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
            st.success("Parando detección...")
        else:
            st.warning("No hay detección en curso.")

    st.markdown("""
    **Notas**
    - En Linux, ejecuta el contenedor Docker con `--device /dev/video0` para que el contenedor acceda a la cámara.
    - En Windows/Mac la cámara desde contenedor es más compleja; para pruebas usa local (no contenedorizado) o usa RTSP.
    """)

# -------------------------
# FIN
# -------------------------
st.sidebar.markdown("---")
st.sidebar.info("Proyecto: Reconocimiento de herramientas + detección de velocidad\nHecho con OpenCV, MediaPipe y TensorFlow (opcional).")
