#!/usr/bin/env python3
"""
Test processing directly to find errors
"""

import cv2
import time
from lane_detector import LaneDetector

# Test with uploaded video
input_path = "uploads/1761970552_5943714-hd_1920_1080_30fps.mp4"
output_path = "test_output.mp4"

print(f"Testing processing: {input_path}")

try:
    # Initialize lane detector
    lane_detector = LaneDetector()
    print("✅ Lane detector initialized")

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {input_path}")
        exit(1)

    print("✅ Video opened")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Cannot create output video writer")
        exit(1)

    print("✅ Video writer created")

    processed_frames = 0
    start_time = time.time()

    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect lanes
        lane_frame = lane_detector.detect(frame)

        # Write output
        out.write(lane_frame)
        processed_frames += 1

        if processed_frames % 30 == 0:
            print(f"  Processed {processed_frames}/{total_frames} frames...")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    avg_fps = processed_frames / elapsed if elapsed > 0 else 0

    print(f"\n✅ SUCCESS!")
    print(f"Processed {processed_frames} frames in {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output: {output_path}")

except Exception as e:
    import traceback
    print(f"\n❌ ERROR: {e}")
    print(traceback.format_exc())
