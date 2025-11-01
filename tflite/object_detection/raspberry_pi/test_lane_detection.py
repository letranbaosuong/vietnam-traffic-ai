#!/usr/bin/env python3
"""
Test script for lane detection module
Demonstrates lane detection on sample video or camera feed
"""

import cv2
import argparse
import time
from lane_detector import LaneDetector


def test_on_camera(camera_id=0, width=640, height=480):
    """Test lane detection on camera feed"""
    print(f"Starting camera {camera_id}...")

    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    # Initialize lane detector
    detector = LaneDetector(img_height=height, img_width=width)

    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()

    print("Press 'q' or ESC to quit")
    print("Press 'r' to toggle ROI visualization")
    print("Press 's' to save screenshot")

    show_roi = False
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect lanes
        lane_result = detector.detect(frame)

        # Visualize
        output = detector.visualize(frame, lane_result, show_roi=show_roi)

        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        # Draw FPS
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(output, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw lane info
        left_status = "YES" if lane_result['left_lane'] is not None else "NO"
        right_status = "YES" if lane_result['right_lane'] is not None else "NO"

        info_text = f'Left: {left_status} | Right: {right_status}'
        cv2.putText(output, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Lane Detection Test', output)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Quitting...")
            break
        elif key == ord('r'):  # Toggle ROI
            show_roi = not show_roi
            print(f"ROI visualization: {'ON' if show_roi else 'OFF'}")
        elif key == ord('s'):  # Save screenshot
            filename = f'lane_detection_screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, output)
            print(f"Saved: {filename}")
            screenshot_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


def test_on_video(video_path, save_output=False):
    """Test lane detection on video file"""
    print(f"Opening video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Initialize lane detector
    detector = LaneDetector(img_height=height, img_width=width)

    # Video writer for output
    writer = None
    if save_output:
        output_path = 'lane_detection_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_num = 0
    start_time = time.time()

    print("Press 'q' or ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        frame_num += 1

        # Detect lanes
        lane_result = detector.detect(frame)

        # Visualize
        output = detector.visualize(frame, lane_result, show_roi=False)

        # Draw progress
        progress = f'Frame {frame_num}/{total_frames}'
        cv2.putText(output, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write output
        if writer is not None:
            writer.write(output)

        # Show frame
        cv2.imshow('Lane Detection - Video', output)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Stopping...")
            break

    # Cleanup
    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0

    print(f"Processed {frame_num} frames in {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Test lane detection module',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['camera', 'video'],
        default='camera',
        help='Test mode: camera or video'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Camera ID (for camera mode) or video path (for video mode)'
    )

    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Frame width (camera mode only)'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Frame height (camera mode only)'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output video (video mode only)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Lane Detection Test Script")
    print("=" * 60)

    if args.mode == 'camera':
        camera_id = int(args.source)
        test_on_camera(camera_id, args.width, args.height)
    else:
        test_on_video(args.source, args.save)


if __name__ == '__main__':
    main()
