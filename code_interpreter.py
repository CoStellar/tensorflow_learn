import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tflite_model_maker import object_detector

# Ustawienie poziomu logowania TensorFlow na ERROR
tf.get_logger().setLevel('ERROR')

# Załadowanie modelu TensorFlow Lite
model_path = 'koty_psy.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Funkcja do wczytania i przetworzenia obrazu zgodnie z wymaganiami modelu
def preprocess_image(image_path, input_size):
    image = Image.open(image_path)
    image = image.resize(input_size)
    image = np.asarray(image)
    image = (image / 255.0).astype(np.float32)  # Normalize image to [0, 1] and convert to float32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# w funkcji main lub gdzieś w GUI
try:
    results = detect_dogs_cats(file_path)
    display_results(file_path, results)
except Exception as e:
    print(f"Error: {e}")
# Funkcja do rozpoznawania psów i kotów na zdjęciu
def detect_dogs_cats(image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_size = tuple(input_details[0]['shape'][1:3])
    image = preprocess_image(image_path, input_size)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes_scores = interpreter.get_tensor(output_details[1]['index'])[0]
    class_ids = [int(i) for i in classes_scores[:, 0]]
    scores = classes_scores[:, 1]
    
    results = []
    for i, score in enumerate(scores):
        if score > 0.5:  # Only consider results with confidence above 50%
            class_id = class_ids[i]
            if class_id == 0:
                label = 'cat'
            elif class_id == 1:
                label = 'dog'
            else:
                label = 'unknown'
            results.append((label, score, boxes[i]))
    
    return results

# Funkcja do wyświetlenia wyników detekcji na obrazie
def display_results(image_path, results):
    image = cv2.imread(image_path)
    for result in results:
        label, score, box = result
        if isinstance(box, np.ndarray):
            box = box.tolist()  # Convert numpy array to list
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin * image.shape[1]), int(ymin * image.shape[0]), int(xmax * image.shape[1]), int(ymax * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((600, 600))
    image = ImageTk.PhotoImage(image)
    result_label.configure(image=image)
    result_label.image = image


# Funkcja obsługująca przycisk wyboru pliku
def choose_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        results = detect_dogs_cats(file_path)
        display_results(file_path, results)

# Tworzenie interfejsu użytkownika
root = tk.Tk()
root.title('Detekcja kotów i psów')
root.geometry('800x600')

choose_button = tk.Button(root, text='Wybierz zdjęcie', command=choose_file)
choose_button.pack(pady=20)

result_label = tk.Label(root)
result_label.pack(expand=True, fill='both')

root.mainloop()
