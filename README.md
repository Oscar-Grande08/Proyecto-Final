# Proyecto-Final
Proyecto: Sistema de Clasificación y Monitoreo de Laboratorio de Electrónica
Introducción

Este proyecto tiene como objetivo desarrollar un sistema integral para la clasificación de elementos de laboratorio de electrónica y la detección de la velocidad de los usuarios dentro del laboratorio, utilizando tecnologías modernas de visión por computador y procesamiento de datos. La propuesta busca combinar conceptos de Ingeniería Electrónica, programación concurrente y despliegue de aplicaciones, integrando todo en una plataforma web interactiva.

Para lograrlo, se planteó el siguiente protocolo de solución:

Desarrollo de una base de datos de imágenes de los elementos de laboratorio:

Creación de un sistema clasificador de elementos usando la librería MediaPipe.

Reconocimiento de la velocidad de las personas en el laboratorio mediante OpenCV2.

Despliegue de la plataforma mediante Streamlit.

Metodología de desarrollo

El proyecto se estructuró en cuatro fases principales:

## 1. Creación de la base de datos

Implementación de web scraping para la búsqueda y recopilación de imágenes de al menos 10 elementos electrónicos utilizando librerías como Selenium.

Generación de un mínimo de 200 imágenes por elemento.

Uso de programación concurrente (hilos, sección crítica, semaforización, mutex) para optimizar la recolección y asegurar la integridad de los datos.

## 2. Extracción, Transformación y Carga (ETL)

Preprocesamiento y procesamiento de la información recopilada mediante hilos para eficiencia.

Limpieza de datos: identificación y eliminación de imágenes incorrectas o irrelevantes.

Extracción de información relevante y transformación de los datos para ser utilizados en el algoritmo de clasificación.

Documentación detallada del manejo de hilos y procesos concurrentes en cada etapa.

## 3. Desarrollo del algoritmo de clasificación y monitoreo

Clasificación de los elementos de laboratorio con MediaPipe.

Análisis de la velocidad de los usuarios en tiempo real usando OpenCV2.

Integración de la detección de elementos y monitoreo de velocidad mediante la cámara del PC.

Implementación y explicación detallada del uso de hilos para garantizar un procesamiento eficiente en tiempo real.

## 4. Despliegue de la aplicación

Empaquetado de la aplicación en un contenedor Docker para su ejecución local, incluyendo instrucciones paso a paso.

Despliegue de la plataforma en la web mediante Streamlit.

Publicación de la imagen del contenedor en Docker Hub para facilitar su distribución y uso.

Documentación completa del proyecto en Overleaf, explicando todos los procesos y metodologías implementadas.
