#!/usr/bin/env python3
"""Simple test on image using TensorFlow Lite."""
import cv2
import numpy as np
import tensorflow as tf

def main():
    print("Testing Object Detection with TensorFlow Lite...")

    # Load test image
    image_path = 'test_data/table.jpg'
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"ERROR: Could not load image")
        return

    print(f"Image shape: {image.shape}")

    # Load model
    model_path = 'efficientdet_lite0.tflite'
    print(f"Loading model: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")

    # Prepare input
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32) / 255.0

    print(f"Preprocessed input shape: {input_data.shape}")

    # Run inference
    print("\nRunning inference...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    print(f"\nDetection results:")
    print(f"Number of output tensors: {len(output_details)}")

    # Show detections above threshold
    threshold = 0.3
    img_height, img_width, _ = image.shape
    detections = 0

    for i in range(len(scores)):
        if scores[i] > threshold:
            detections += 1
            ymin, xmin, ymax, xmax = boxes[i]
            left = int(xmin * img_width)
            top = int(ymin * img_height)
            right = int(xmax * img_width)
            bottom = int(ymax * img_height)

            print(f"Detection {detections}:")
            print(f"  Class: {int(classes[i])}")
            print(f"  Score: {scores[i]:.3f}")
            print(f"  Box: [{left}, {top}, {right}, {bottom}]")

            # Draw on image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"Class {int(classes[i])}: {scores[i]:.2f}"
            cv2.putText(image, label, (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"\nTotal detections (score > {threshold}): {detections}")

    # Save result
    output_path = 'test_output.jpg'
    cv2.imwrite(output_path, image)
    print(f"Output saved to: {output_path}")

    print("\nTest completed successfully!")

if __name__ == '__main__':
    main()
