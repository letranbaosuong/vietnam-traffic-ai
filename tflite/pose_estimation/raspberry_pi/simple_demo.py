#!/usr/bin/env python3
"""Simple demo to test pose estimation."""
import sys
import cv2
import numpy as np

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except:
    print("TensorFlow not available")

try:
    from ai_edge_litert.interpreter import Interpreter
    print("ai-edge-litert available")
except:
    print("ai-edge-litert not available")

print("\nAttempting to load model...")
try:
    from ml import Movenet
    print("Successfully imported Movenet")

    print("Initializing MoveNet...")
    pose_detector = Movenet('movenet_lightning')
    print("Model initialized!")

    print("\nReading test image...")
    image = cv2.imread('test_data/image1.png')
    if image is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    print(f"Image loaded: {image.shape}")

    print("\nRunning pose detection...")
    person = pose_detector.detect(image)
    print(f"SUCCESS! Detected {len(person.keypoints)} keypoints")

    print("\nKeypoint details:")
    for i, kp in enumerate(person.keypoints):
        print(f"  {i}: pos=({kp.coordinate.x:.2f}, {kp.coordinate.y:.2f}), score={kp.score:.3f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDemo completed successfully!")
