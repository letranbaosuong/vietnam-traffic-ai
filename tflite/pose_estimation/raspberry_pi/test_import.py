#!/usr/bin/env python3
"""Test imports step by step."""
import sys

print("Step 1: Import cv2...")
try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\nStep 2: Import numpy...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\nStep 3: Import tensorflow...")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\nStep 4: Import data module...")
try:
    from data import Person, BodyPart
    print("  ✓ data module")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\nStep 5: Create TFLite Interpreter...")
try:
    model_path = 'movenet_lightning.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"  ✓ Loaded model: {model_path}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\nStep 6: Read test image...")
try:
    image = cv2.imread('test_data/image1.png')
    if image is None:
        raise Exception("Could not read image")
    print(f"  ✓ Image loaded: {image.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
