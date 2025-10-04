#!/usr/bin/env python3
"""Demo script to run pose estimation on test images and save results."""
import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
from ml import Movenet
import utils

# Load test images
test_image_paths = ['test_data/image1.png', 'test_data/image2.jpeg', 'test_data/image3.jpeg']

# Initialize MoveNet Lightning model
print("Initializing MoveNet Lightning model...")
try:
    pose_detector = Movenet('movenet_lightning')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Process each test image
for image_path in test_image_paths:
    print(f"\nProcessing {image_path}...")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not read {image_path}")
        continue

    print(f"  Image size: {image.shape}")

    # Detect pose
    try:
        person = pose_detector.detect(image)
        print(f"  Detected {len(person.keypoints)} keypoints")
    except Exception as e:
        print(f"  Error detecting pose: {e}")
        continue

    # Visualize keypoints
    try:
        output_image = utils.visualize(image, [person])
    except Exception as e:
        print(f"  Error visualizing: {e}")
        continue

    # Save result
    output_path = image_path.replace('test_data/', 'test_data/output_')
    try:
        cv2.imwrite(output_path, output_image)
        print(f"  Saved result to {output_path}")
    except Exception as e:
        print(f"  Error saving: {e}")
        continue

    # Print keypoint information
    for i, keypoint in enumerate(person.keypoints):
        print(f"    Keypoint {i}: ({keypoint.coordinate.x:.2f}, {keypoint.coordinate.y:.2f}), score: {keypoint.score:.2f}")

print("\nDone! Check test_data/ folder for output images.")
