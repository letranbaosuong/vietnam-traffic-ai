#!/usr/bin/env python3
"""Simple test script to run object detection on test image."""
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

def main():
    print("Testing TFLite Object Detection...")

    # Load test image
    image_path = 'test_data/table.jpg'
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return

    print(f"Image shape: {image.shape}")

    # Initialize the object detection model
    model_path = 'efficientdet_lite0.tflite'
    print(f"Loading model: {model_path}")

    base_options = core.BaseOptions(
        file_name=model_path, use_coral=False, num_threads=4)
    detection_options = processor.DetectionOptions(
        max_results=10, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    print("Model loaded successfully!")

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection
    print("\nRunning detection...")
    detection_result = detector.detect(input_tensor)

    # Print results
    print(f"\nDetected {len(detection_result.detections)} objects:")
    for detection in detection_result.detections:
        category = detection.categories[0]
        bbox = detection.bounding_box
        print(f"  - {category.category_name}: {category.score:.2f} at [{bbox.origin_x}, {bbox.origin_y}, {bbox.width}, {bbox.height}]")

    # Visualize results
    output_image = utils.visualize(image.copy(), detection_result)

    # Save output
    output_path = 'test_output.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"\nOutput saved to: {output_path}")

    print("\nTest completed successfully!")

if __name__ == '__main__':
    main()
