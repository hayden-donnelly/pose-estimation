import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

def render_keypoints(image, image_width, image_height):

    cv2.imwrite("../data/output/test.jpg", image)

# Load the input image.
#image_path = '../data/input/basketball_guy.jpg'
#image = tf.io.read_file(image_path)
#original_image = image
#print(original_image.shape)
#image = tf.compat.v1.image.decode_jpeg(image)
#image = tf.expand_dims(image, axis=0)

# Extract image dimensions.
#image_width = image.shape[2]
#image_height = image.shape[1]

#render_keypoints(original_image, image_width, image_height)

image = cv2.imread('../data/input/basketball_guy.jpg')
image_height, image_width, _ = image.shape
render_keypoints(image, image_width, image_height)

tf_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 512, 512)
tf_image = tf.cast(tf_image, dtype=tf.int32)