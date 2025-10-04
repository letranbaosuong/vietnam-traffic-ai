#!/usr/bin/env python3
"""Object detection using TensorFlow Lite directly (without tflite-support)."""
import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

def load_labels(filename):
    """Load labels from file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def run_detection(interpreter, image):
    """Run object detection on an image."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input size
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize if needed
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    return boxes, classes, scores

def visualize_detections(image, boxes, classes, scores, labels, threshold=0.3):
    """Draw bounding boxes and labels on image."""
    height, width, _ = image.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            # Get box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)

            # Get class name
            class_id = int(classes[i])
            if class_id < len(labels):
                label = labels[class_id]
            else:
                label = f"Class {class_id}"

            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label with background
            label_text = f"{label}: {scores[i]:.2f}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (left, top - label_size[1] - 5),
                         (left + label_size[0], top), (0, 255, 0), -1)
            cv2.putText(image, label_text, (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        labels_file: str = None) -> None:
    """Continuously run inference on images acquired from the camera."""

    # Load TFLite model
    print(f"Loading model: {model}")
    interpreter = tf.lite.Interpreter(model_path=model, num_threads=num_threads)
    interpreter.allocate_tensors()
    print("Model loaded!")

    # Load labels
    labels = []
    if labels_file:
        try:
            labels = load_labels(labels_file)
            print(f"Loaded {len(labels)} labels")
        except:
            print(f"Warning: Could not load labels from {labels_file}")

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20
    left_margin = 24
    text_color = (0, 0, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    print("Starting detection... Press ESC to quit.")

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam.')

        counter += 1
        image = cv2.flip(image, 1)

        # Run object detection
        boxes, classes, scores = run_detection(interpreter, image)

        # Visualize results
        image = visualize_detections(image, boxes, classes, scores, labels)

        # Calculate FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show FPS
        fps_text = f'FPS = {fps:.1f}'
        cv2.putText(image, fps_text, (left_margin, row_size),
                   cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

        # Display
        cv2.imshow('Object Detection', image)

        # Stop on ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path to TFLite model',
                       default='efficientdet_lite0.tflite')
    parser.add_argument('--labels', help='Path to labels file', default=None)
    parser.add_argument('--cameraId', help='Camera ID', type=int, default=0)
    parser.add_argument('--frameWidth', help='Frame width', type=int, default=640)
    parser.add_argument('--frameHeight', help='Frame height', type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads', type=int, default=4)
    args = parser.parse_args()

    run(args.model, args.cameraId, args.frameWidth, args.frameHeight,
        args.numThreads, args.labels)

if __name__ == '__main__':
    main()
