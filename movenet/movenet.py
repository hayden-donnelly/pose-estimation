import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

ordered_keypoint_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'
]

def render_keypoints(image_width, image_height, keypoints, confidence_threshold):
    scaled_keypoints = np.multiply(keypoints, [image_width, image_height, 1])

    for i in range(scaled_keypoints.shape[0]):
        x, y, score = scaled_keypoints[i]
        is_confident = score > confidence_threshold
        print(ordered_keypoint_labels[i], x, y, score, is_confident)

print(tf.config.list_physical_devices('GPU'))

# Load the input image.
image_path = 'movenet/data/basketball_guy.jpg'
image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)
# Resize and pad the image to keep the aspect ratio and fit the expected size.
image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Run model inference.
outputs = movenet(image)
# Output is a [1, 6, 56] tensor.
output = outputs['output_0'].numpy()
print("output shape:", output.shape)

# Throw away bounding box, then reshape.
keypoints = output[0, :6, :17*3].reshape(1, 6, 17, 3)
print("keypoints shape:", keypoints.shape)

render_keypoints(256, 256, keypoints[0, 0], 0.5)