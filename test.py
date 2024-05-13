import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pylab as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())


COLORS = np.random.uniform(0, 255, size=(2, 3))

print("TF version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
validation_dir = 'archive/dataset/test_set'
train_dir = 'archive/dataset/training_set'


BATCH_SIZE = 32
IMG_SIZE = (160,160)

train_dataset = image_dataset_from_directory(train_dir, shuffle = True, batch_size = BATCH_SIZE, image_size = IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir, shuffle = True, batch_size = BATCH_SIZE, image_size = IMG_SIZE)

class_names = train_dataset.class_names


val_batches = tf.data.experimental.cardinality(validation_dataset) #zbiera ilość batches
print(val_batches)
test_dataset = validation_dataset.take(val_batches // 5) #wyciągnięcie batches z validation_dataset o ilości batches podzielone przez 5 i zaokrąglone
metrics_dataset = validation_dataset
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))



AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
metrics_dataset = metrics_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


rescale = tf.keras.layers.Rescaling(1./127.5,offset = -1)



IMG_SHAPE = IMG_SIZE + (3,)
model = tf.keras.models.load_model('trained_model.keras')


#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
image_metrics, label_metrics = metrics_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()
metrics_predictions = model.predict_on_batch(image_metrics).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
metrics_predictions = tf.nn.sigmoid(metrics_predictions)
predictions = tf.where(predictions < 0.5, 0, 1)
metrics_predictions = tf.where(metrics_predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

print("Metrics for one batch:")
print("Accuracy score: {:.3f}".format(metrics.accuracy_score(label_batch, predictions)))
print("Balanced accuracy score: {:.3f}".format(metrics.balanced_accuracy_score(label_batch, predictions)))
print("Average precision score: {:.3f}".format(metrics.average_precision_score(label_batch, predictions)))
print('\n')
print("Metrics for test_dataset:")
print("Accuracy score: {:.3f}".format(metrics.accuracy_score(label_metrics, metrics_predictions)))
print("Balanced accuracy score: {:.3f}".format(metrics.balanced_accuracy_score(label_metrics, metrics_predictions)))
print("Average precision score: {:.3f}".format(metrics.average_precision_score(label_metrics, metrics_predictions)))

plt.figure(figsize=(10, 10))
for i in range(32):
  ax = plt.subplot(4, 8, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()