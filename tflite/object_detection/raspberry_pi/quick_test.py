#!/usr/bin/env python3
"""
Quick test script - Processes first 100 frames of video to verify lane detection
"""

import cv2
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lane_detector import LaneDetector


def quick_test(video_path, max_frames=100):
    """Quick test on first N frames of video"""

    print(f"Quick Test: {video_path}")
    print(f"Processing first {max_frames} frames...")
    print("-" * 60)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return False

    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print("-" * 60)

    # Initialize detector
    detector = LaneDetector(img_height=height, img_width=width)

    # Statistics
    frames_processed = 0
    lanes_detected = 0
    left_detected = 0
    right_detected = 0
    both_detected = 0

    start_time = time.time()

    # Process frames
    while frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1

        # Detect lanes
        result = detector.detect(frame)

        # Count detections
        has_left = result['left_lane'] is not None
        has_right = result['right_lane'] is not None

        if has_left or has_right:
            lanes_detected += 1
        if has_left:
            left_detected += 1
        if has_right:
            right_detected += 1
        if has_left and has_right:
            both_detected += 1

        # Show progress
        if frames_processed % 10 == 0:
            print(f"Processed: {frames_processed}/{max_frames} frames", end='\r')

    elapsed = time.time() - start_time
    avg_fps = frames_processed / elapsed if elapsed > 0 else 0

    # Print results
    print()
    print("-" * 60)
    print("Results:")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print()
    print(f"  Frames with lanes: {lanes_detected} ({100*lanes_detected/frames_processed:.1f}%)")
    print(f"  Left lane detected: {left_detected} ({100*left_detected/frames_processed:.1f}%)")
    print(f"  Right lane detected: {right_detected} ({100*right_detected/frames_processed:.1f}%)")
    print(f"  Both lanes detected: {both_detected} ({100*both_detected/frames_processed:.1f}%)")
    print("-" * 60)

    # Cleanup
    cap.release()

    return True


def main():
    # Test videos
    test_videos = [
        "test_videos/solidWhiteRight.mp4",
        "test_videos/detect_video_danang.mp4"
    ]

    print("=" * 60)
    print("Lane Detection Quick Test")
    print("=" * 60)
    print()

    for video_path in test_videos:
        if Path(video_path).exists():
            quick_test(video_path, max_frames=100)
            print()
        else:
            print(f"SKIP: {video_path} not found")
            print()

    print("=" * 60)
    print("Quick test completed!")
    print("=" * 60)
    print()
    print("To view full video processing:")
    print("  python3 test_lane_detection.py --mode video --source test_videos/solidWhiteRight.mp4")
    print()


if __name__ == '__main__':
    main()
