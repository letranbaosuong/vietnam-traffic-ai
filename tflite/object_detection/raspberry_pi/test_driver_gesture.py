#!/usr/bin/env python3
"""
Test script for Driver Gesture Detection
Tests dangerous driving behavior detection on video or camera
"""

import cv2
import argparse
import time
from driver_gesture_detector import DriverGestureDetector
from gesture_warning_system import GestureWarningSystem


def test_on_video(video_path, save_output=False):
    """Test gesture detection on video file"""
    print(f"Testing driver gesture detection on: {video_path}")
    print("-" * 60)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print("-" * 60)

    # Initialize detector and warning system
    gesture_detector = DriverGestureDetector()
    warning_system = GestureWarningSystem()

    # Video writer for output
    writer = None
    if save_output:
        output_path = 'driver_gesture_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_num = 0
    start_time = time.time()
    warning_count = 0

    print("\nPress 'q' or ESC to stop")
    print("Press 's' to save statistics")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video")
            break

        frame_num += 1

        # Detect gestures
        warnings, annotated_frame = gesture_detector.detect(frame)

        # Clear old warnings from system
        warning_system.clear_old_warnings(max_age=2.0)

        # Add warnings
        for warning in warnings:
            if "điện thoại" in warning.lower():
                warning_system.add_warning(warning, 'phone_usage')
            elif "tập trung" in warning.lower():
                warning_system.add_warning(warning, 'distraction')
            elif "vô lăng" in warning.lower():
                warning_system.add_warning(warning, 'hands_off_wheel')

        # Visualize
        output_frame = warning_system.draw_warnings(annotated_frame, warnings)
        output_frame = warning_system.draw_status_bar(output_frame)

        # Update warning count
        if warnings:
            warning_count += 1

        # Draw progress
        progress_text = f"Frame {frame_num}/{total_frames}"
        cv2.putText(output_frame, progress_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write output
        if writer is not None:
            writer.write(output_frame)

        # Show frame
        cv2.imshow('Driver Gesture Detection Test', output_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\nStopping...")
            break
        elif key == ord('s'):
            print("\n" + warning_system.get_warning_report())

    # Cleanup
    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0

    print(f"\nProcessed {frame_num} frames in {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Frames with warnings: {warning_count} ({100*warning_count/frame_num:.1f}%)")

    # Print statistics
    print("\n" + "=" * 60)
    print("GESTURE DETECTION STATISTICS")
    print("=" * 60)
    print(warning_system.get_warning_report())

    gesture_stats = gesture_detector.get_statistics()
    print("\nDetector Statistics:")
    for key, value in gesture_stats.items():
        print(f"  {key}: {value}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def test_on_camera(camera_id=0, width=640, height=480):
    """Test gesture detection on camera"""
    print(f"Testing driver gesture detection on camera {camera_id}")
    print("-" * 60)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return

    print(f"Camera: {width}x{height}")
    print("\nPress 'q' or ESC to quit")
    print("Press 's' to save statistics")
    print("Press 'r' to reset statistics")
    print("-" * 60)

    # Initialize
    gesture_detector = DriverGestureDetector()
    warning_system = GestureWarningSystem()

    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        frame_count += 1

        # Detect gestures
        warnings, annotated_frame = gesture_detector.detect(frame)

        # Clear old warnings
        warning_system.clear_old_warnings(max_age=2.0)

        # Add warnings
        for warning in warnings:
            if "điện thoại" in warning.lower():
                warning_system.add_warning(warning, 'phone_usage')
            elif "tập trung" in warning.lower():
                warning_system.add_warning(warning, 'distraction')
            elif "vô lăng" in warning.lower():
                warning_system.add_warning(warning, 'hands_off_wheel')

        # Visualize
        output_frame = warning_system.draw_warnings(annotated_frame, warnings)
        output_frame = warning_system.draw_status_bar(output_frame)

        # Calculate FPS
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        # Draw FPS
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(output_frame, fps_text, (width - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Driver Gesture Detection - Camera', output_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\nQuitting...")
            break
        elif key == ord('s'):
            print("\n" + warning_system.get_warning_report())
        elif key == ord('r'):
            print("\nResetting statistics...")
            warning_system.reset()
            gesture_detector.reset_statistics()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(warning_system.get_warning_report())


def main():
    parser = argparse.ArgumentParser(
        description='Test Driver Gesture Detection',
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
    print("DRIVER GESTURE DETECTION TEST")
    print("=" * 60)
    print()

    if args.mode == 'camera':
        camera_id = int(args.source)
        test_on_camera(camera_id, args.width, args.height)
    else:
        test_on_video(args.source, args.save)

    print("\nTest completed!")


if __name__ == '__main__':
    main()
