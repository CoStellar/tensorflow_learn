import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

directory = "cats_dogs/train"


train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)

feature_extractor_url = "https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2"


feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(IMG_SIZE, IMG_SIZE, 3))


feature_extractor_layer.trainable = False

model = tf.keras.Sequential([feature_extractor_layer, layers.Dense(train_data.num_classes, activation = 'softmax')])



model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = np.ceil(train_data.samples/train_data.batch_size)
history = model.fit(train_data, epochs = 10, steps_per_epoch = steps_per_epoch)

class_names = sorted(test_data.class_indices.items(), key=lambda pair:pair[1])


class_names = np.array([key.title() for key, values in class_names])


for image_batch, label_batch in test_data:
    predicted_batch = model.predict(image_batch)
    predicted_id = np.argmax(predicted_batch, axis=-1)
    predicted_label_batch = class_names[predicted_id]
model.evaluate(test_data)




    