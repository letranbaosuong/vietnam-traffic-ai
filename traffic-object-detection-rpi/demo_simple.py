#!/usr/bin/env python3

import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.detector import TrafficObjectDetector
from src.rpi_optimizer import RPiOptimizer


def demo_image():
    print("=== Simple Traffic Detection Demo ===\n")

    optimizer = RPiOptimizer()
    optimizer.optimize_system()

    detector = TrafficObjectDetector()

    test_image_path = "data/images/traffic_sample.jpg"

    if not Path(test_image_path).exists():
        print(f"Creating sample image at {test_image_path}")
        import numpy as np
        sample_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.putText(sample_img, "Place your traffic image here",
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        Path("data/images").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(test_image_path, sample_img)
        print(f"Sample placeholder created. Replace with real traffic image.")
        return

    image = cv2.imread(test_image_path)
    print(f"Processing image: {test_image_path}")

    detections = detector.detect(image)

    print(f"\nDetected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class_name']}: confidence {det['confidence']:.2%}")

    result = detector.draw_detections(image, detections)

    output_path = "data/outputs/demo_result.jpg"
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"\nResult saved to: {output_path}")


def demo_camera():
    print("=== Camera Traffic Detection Demo ===\n")

    optimizer = RPiOptimizer()
    optimizer.optimize_system()

    detector = TrafficObjectDetector()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit, 's' to save snapshot\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        result = detector.draw_detections(frame, detections)

        cv2.putText(result, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Traffic Detection', result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_path = f"data/outputs/snapshot_{frame_count}.jpg"
            cv2.imwrite(snapshot_path, result)
            print(f"Saved: {snapshot_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simple Traffic Detection Demo')
    parser.add_argument('--mode', choices=['image', 'camera'], default='image',
                       help='Demo mode: image or camera')

    args = parser.parse_args()

    if args.mode == 'image':
        demo_image()
    else:
        demo_camera()