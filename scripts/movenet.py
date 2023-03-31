import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

ordered_keypoint_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'
]

print(tf.config.list_physical_devices('GPU'))

# Load the input image.
image = cv2.imread('../data/input/basketball_guy.jpg')
image_height, image_width, _ = image.shape
tf_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 384, 640)
tf_image = tf.cast(tf_image, dtype=tf.int32)

# Download the model from TF Hub.
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Run model inference.
outputs = movenet(tf_image)
output = outputs['output_0'].numpy()
print("output shape:", output.shape)

# Throw away bounding box, then reshape.
keypoints = output[0, :6, :17*3].reshape(1, 6, 17, 3)
print("keypoints shape:", keypoints.shape)

# Throw away extra keypoints.
keypoints = keypoints[0, 0]
print(keypoints.shape)

# Scale keypoints to original image dimensions.
scaled_keypoints = np.multiply(keypoints, [image_height, image_width, 1])

confidence_threshold = 0.3

# Print out keypoints in easy to read format.
for i in range(scaled_keypoints.shape[0]):
    x, y, score = scaled_keypoints[i]
    is_confident = score > confidence_threshold
    print(ordered_keypoint_labels[i], x, y, score, is_confident)

# Draw keypoints as circles.
for i in range(scaled_keypoints.shape[0]):
    x, y, score = scaled_keypoints[i]
    if(score > confidence_threshold):
        cv2.circle(image, (int(y), int(x)), 5, (0, 255, 0), -1)

cv2.imwrite("../data/output/movenet_output.jpg", image)