import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

ordered_keypoint_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'
]

def render_keypoints(image, image_width, image_height, keypoints, confidence_threshold):
    
    # Convert to OpenCV format.
    image = image.numpy()[0]
    image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    scaled_keypoints = np.multiply(keypoints, [image_width, image_height, 1])

    for i in range(scaled_keypoints.shape[0]):
        x, y, score = scaled_keypoints[i]
        is_confident = score > confidence_threshold
        print(ordered_keypoint_labels[i], x, y, score, is_confident)

    for i in range(scaled_keypoints.shape[0]):
        x, y, score = scaled_keypoints[i]
        if(score > confidence_threshold):
            cv2.circle(image, (int(y), int(x)), 5, (0, 255, 0), -1)

    # Show image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite("output.jpg", image)

print(tf.config.list_physical_devices('GPU'))

# Load the input image.
image_path = '../data/basketball_guy.jpg'
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

render_keypoints(image, 256, 256, keypoints[0, 0], 0.0)