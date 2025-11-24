import os
import time
import threading
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# -----------------------------
# CONFIGURACIÓN DEL PROYECTO
# -----------------------------

COMPONENTES = [
    "capacitor",
    "cautin",
    "fuente_poder",
    "generador",
    "motor",
    "multimetro",
    "osciloscopio",
    "pinzas",
    "protoboard",
    "transformador"
]

IMAGENES_POR_COMPONENTE = 200

BASE_DIR = "data/raw"

# Semáforo: limita cuántos hilos descargan simultáneamente
semaforo = threading.Semaphore(3)  # máximo 3 descargas paralelas
# Mutex para proteger la sección crítica
mutex = threading.Lock()


# -----------------------------
# DESCARGAR UNA IMAGEN (Sección crítica)
# -----------------------------
def descargar_imagen(url, carpeta, nombre_archivo):
    try:
        semaforo.acquire()  # entrar a la zona con límite de hilos

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            ruta = os.path.join(carpeta, nombre_archivo)

            with mutex:  # sección crítica: escribir archivo
                with open(ruta, "wb") as f:
                    f.write(response.content)
                print(f"[OK] Guardada: {ruta}")

    except Exception as e:
        print(f"[ERROR] No se pudo descargar {url}: {e}")

    finally:
        semaforo.release()  # liberar espacio para otro hilo


# -----------------------------
# SCRAPING DE GOOGLE IMAGES
# -----------------------------
def scrape_componente(componente):
    print(f"\n=== Descargando imágenes de: {componente} ===")

    carpeta_objetivo = os.path.join(BASE_DIR, componente)
    os.makedirs(carpeta_objetivo, exist_ok=True)

    # Configurar Chrome
    opciones = Options()
    opciones.add_argument("--headless")  # sin abrir ventana gráfica
    opciones.add_argument("--no-sandbox")
    opciones.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opciones
    )

    # Buscar en Google Images
    query = f"{componente} electronic component"
    driver.get(f"https://www.google.com/search?tbm=isch&q={query}")

    time.sleep(3)

    # Hacer scroll para cargar más imágenes
    for _ in range(10):
        driver.execute_script("window.scrollBy(0, 2000);")
        time.sleep(1)

    imagenes = driver.find_elements(By.CSS_SELECTOR, "img")

    urls = []
    for img in imagenes:
        try:
            url = img.get_attribute("src")
            if url and url.startswith("http"):
                urls.append(url)
        except:
            pass

    driver.quit()

    print(f"Se encontraron {len(urls)} imágenes. Procesando...")

    threads = []
    count = 0

    for url in urls:
        if count >= IMAGENES_POR_COMPONENTE:
            break

        nombre_archivo = f"img_{count:03d}.jpg"

        hilo = threading.Thread(
            target=descargar_imagen,
            args=(url, carpeta_objetivo, nombre_archivo)
        )

        hilo.start()
        threads.append(hilo)
        count += 1

    # Esperar a que todos terminen
    for hilo in threads:
        hilo.join()

    print(f"[✓] {componente}: DESCARGA COMPLETA\n")


# -----------------------------
# PROGRAMA PRINCIPAL
# -----------------------------
if __name__ == "__main__":
    inicio = time.time()

    print("=== INICIANDO SCRAPER MULTIHILO ===")

    for componente in COMPONENTES:
        scrape_componente(componente)

    fin = time.time()
    print(f"\nTiempo total: {fin - inicio:.2f} segundos")
    print("=== PROCESO FINALIZADO ===")
