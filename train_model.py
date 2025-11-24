import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np

# -----------------------------
# 1. CONFIGURACIÓN DE RUTAS
# -----------------------------
# Si estás ejecutando desde: deteccion_velocidad/entrenamiento/
# Entonces data está en: ../data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

print("Ruta DATA_DIR:", DATA_DIR)
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError("No existe la carpeta data en ../data")

# -----------------------------
# 2. PARÁMETROS
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# -----------------------------
# 3. GENERADOR CON VALIDACIÓN
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2
)

train_gen_raw = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen_raw = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# -----------------------------
# 4. FUNCIÓN PARA IGNORAR ERRORES
# -----------------------------
def safe_generator(gen):
    while True:
        try:
            batch = next(gen)
            yield batch
        except Exception as e:
            print("⚠ Imagen corrupta ignorada:", e)
            continue

train_gen = safe_generator(train_gen_raw)
val_gen = safe_generator(val_gen_raw)

# -----------------------------
# 5. DEFINICIÓN DEL MODELO
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(train_gen_raw.num_classes, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 6. ENTRENAMIENTO
# -----------------------------
steps_train = train_gen_raw.samples // BATCH_SIZE
steps_val = val_gen_raw.samples // BATCH_SIZE

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=EPOCHS
)

# -----------------------------
# 7. GUARDAR MODELO KERAS
# -----------------------------
model.save("modelo_equipo_electrico.h5")
print("✅ Modelo guardado como modelo_equipo_electrico.h5")

# -----------------------------
# 8. EXPORTAR A TFLITE
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_equipo_electrico.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Modelo exportado a modelo_equipo_electrico.tflite")
