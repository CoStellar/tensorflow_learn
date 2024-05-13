import numpy as np
import os
import tensorflow as tf
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata

# Upewnij się, że używana jest wersja TensorFlow 2.x
assert tf.__version__.startswith('2')

# Ustawienie poziomu logowania TensorFlow na ERROR
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Wyświetlenie dostępnych urządzeń GPU
print(tf.config.list_physical_devices('GPU'))

# Definicja rozmiaru wsadowego (batch size) i rozmiaru obrazu
BATCH_SIZE = 64
IMG_SIZE = (128, 128)

# Konfiguracja generatora danych obrazowych
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalizacja wartości pikseli do zakresu [0, 1]
    validation_split=0.2  # Podział danych na zbiór treningowy i walidacyjny
)

# Załadowanie danych treningowych z formatu Pascal VOC
train_data = object_detector.DataLoader.from_pascal_voc(
    'cats_dogs/train',  # Ścieżka do folderu zawierającego dane treningowe
    'cats_dogs/train',  # Ścieżka do pliku zawierającego adnotacje treningowe
    ['cat', 'dog']  # Lista klas obiektów do rozpoznawania
)

# Załadowanie danych walidacyjnych z formatu Pascal VOC
val_data = object_detector.DataLoader.from_pascal_voc(
    'cats_dogs/validate',  # Ścieżka do folderu zawierającego dane walidacyjne
    'cats_dogs/validate',  # Ścieżka do pliku zawierającego adnotacje walidacyjne
    ['cat', 'dog']  # Lista klas obiektów do rozpoznawania
)

# Wybór specyfikacji modelu (np. EfficientDet Lite 0)
spec = model_spec.get('efficientdet_lite0')

# Utworzenie modelu detekcji obiektów na podstawie danych treningowych i specyfikacji modelu
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

# Ewaluacja modelu na danych walidacyjnych
model.evaluate(val_data)

# Eksport modelu do formatu TensorFlow Lite
model.export(export_dir='.', export_format=[ExportFormat.TFLITE], tflite_filename='koty_psy.tflite')

# Ewaluacja modelu TensorFlow Lite na danych walidacyjnych
model.evaluate_tflite('koty_psy.tflite', val_data)
