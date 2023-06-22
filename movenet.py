import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import argparse
import os

ordered_keypoint_labels = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'
]

edges = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

def draw_connections(image, keypoints, confidence_threshold):
    for i in range(len(edges)):
        index1, index2 = edges[i]
        x1, y1, score1 = keypoints[index1]
        x2, y2, score2 = keypoints[index2]

        if(score1 > confidence_threshold and score2 > confidence_threshold):
            cv2.line(image, (int(y1), int(x1)), (int(y2), int(x2)), (0, 0, 255), 4)


def draw_keypoints(image, keypoints, confidence_threshold):
    for i in range(keypoints.shape[0]):
        x, y, score = keypoints[i]
        if(score > confidence_threshold):
            cv2.circle(image, (int(y), int(x)), 5, (0, 255, 0), -1)


def cv_to_tf(image):
    tf_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 384, 640)
    tf_image = tf.cast(tf_image, dtype=tf.int32)   
    return tf_image


def video_pose_estimation(args, model):
    # Load the video.
    video = cv2.VideoCapture(args.input_path)

    # Check if the video was loaded successfully.
    if not video.isOpened():
        print("Error loading video file")
        return
    else:
        print("Successfully loaded", args.input_path)

    # Get the width and height of the video.
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(args.output_path, 'output.mp4'), fourcc, 30.0, (width, height))

    # Loop through the video frame by frame.
    while video.isOpened():
        # Read a frame from the video.
        ret, frame = video.read()

        # Check if the frame was read successfully.
        if not ret:
            break

        # Run model inference.
        outputs = model(cv_to_tf(frame))
        output = outputs['output_0'].numpy()

        # Throw away bounding box, then reshape.
        keypoints = output[0, :output.shape[1], :17*3].reshape(output.shape[1], 17, 3)

        for i in range(keypoints.shape[0]):
            # Scale keypoints to original image dimensions.
            scaled_keypoints = np.multiply(keypoints[i], [height, width, 1])

            # Draw connections as lines.
            draw_connections(frame, scaled_keypoints, args.confidence_threshold)

            # Draw keypoints as circles.
            draw_keypoints(frame, scaled_keypoints, args.confidence_threshold)

        # Write the modified frame to the output video.
        out.write(frame)
    print('Saved video to', args.output_path)


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))

    parser = argparse.ArgumentParser()
    help_text = "Path to input video."
    parser.add_argument('--input_path', type=str, help=help_text)
    help_text = "Path to output directory."
    parser.add_argument('--output_path', type=str, default='data', help=help_text)
    help_text = "Confidence threshold for keypoints."
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help=help_text)
    args = parser.parse_args()

    assert args.input_path is not None, "Input path is required"

    # Download the model from TF Hub.
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet = model.signatures['serving_default']

    video_pose_estimation(args, movenet)
    