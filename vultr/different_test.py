import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)

# Download model
model_name = "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"
model_date = "20200711"
PATH_TO_MODEL_DIR = download_model(model_name, model_date)

# Load model
model_fn = tf.saved_model.load(PATH_TO_MODEL_DIR + "/saved_model")

# Load labels
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Images to run detection
images = [r"archive\dataset\test_set\cats\cat.4001.jpg", r"archive\dataset\test_set\cats\cat.4025.jpg"]

for image in images:
    print(f"Running inference for image - {image}")

    # Load image into a numpy array
    image_np = np.array(Image.open(image))

    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)

    # Add an axis
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = model_fn(input_tensor)

    # Outputs are tensor batches.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes cast as ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            line_thickness=8)

    # Save image with detections
    img = Image.fromarray(image_np_with_detections)
    img_filename = image[0:-4] + "_detect" + image[-4:]
    img.save(img_filename)