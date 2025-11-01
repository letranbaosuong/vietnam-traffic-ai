#!/usr/bin/env python3
"""
Demo Simulation for Driver Gesture Detection
Simulates gesture detection without MediaPipe for testing visualization
"""

import cv2
import numpy as np
import time
import random
from gesture_warning_system import GestureWarningSystem


def simulate_gestures(frame_num):
    """Simulate gesture warnings based on frame number"""
    warnings = []

    # Simulate different scenarios
    if 50 < frame_num < 100:
        warnings.append("⚠️ NGUY HIỂM: Đang gọi điện thoại!")

    if 150 < frame_num < 200:
        warnings.append("⚠️ MẤT TẬP TRUNG: Đang nhìn sang PHẢI (35°)!")

    if 250 < frame_num < 280:
        warnings.append("⚠️ CẢNH BÁO: Tay rời vô lăng!")

    if 320 < frame_num < 360:
        warnings.append("⚠️ NGUY HIỂM: Đang xem điện thoại!")

    # Random occasional warnings
    if random.random() > 0.95:
        warnings.append("⚠️ MẤT TẬP TRUNG: Đang nhìn sang TRÁI (28°)!")

    return warnings


def demo_on_video(video_path, save_output=False):
    """Demo gesture detection visualization on video"""
    print(f"Demo Driver Gesture Detection Simulation")
    print(f"Video: {video_path}")
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

    # Initialize warning system
    warning_system = GestureWarningSystem()

    # Video writer for output
    writer = None
    if save_output:
        output_path = 'driver_gesture_demo_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")

    frame_num = 0
    start_time = time.time()
    warning_count = 0

    print("\nSimulating gesture detection...")
    print("Press 'q' or ESC to stop")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video")
            break

        frame_num += 1

        # Simulate gesture detection
        simulated_warnings = simulate_gestures(frame_num)

        # Clear old warnings
        warning_system.clear_old_warnings(max_age=2.0)

        # Add warnings
        for warning in simulated_warnings:
            if "điện thoại" in warning.lower():
                warning_system.add_warning(warning, 'phone_usage')
            elif "tập trung" in warning.lower():
                warning_system.add_warning(warning, 'distraction')
            elif "vô lăng" in warning.lower():
                warning_system.add_warning(warning, 'hands_off_wheel')

        # Visualize
        output_frame = warning_system.draw_warnings(frame, simulated_warnings)
        output_frame = warning_system.draw_status_bar(output_frame)

        # Update warning count
        if simulated_warnings:
            warning_count += 1

        # Draw simulation notice
        cv2.putText(output_frame, "SIMULATION MODE", (width - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw progress
        progress_text = f"Frame {frame_num}/{total_frames}"
        cv2.putText(output_frame, progress_text, (10, height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write output
        if writer is not None:
            writer.write(output_frame)

        # Show frame
        cv2.imshow('Driver Gesture Detection DEMO', output_frame)

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\nStopping...")
            break

    # Cleanup
    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0

    print(f"\nProcessed {frame_num} frames in {elapsed:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Frames with warnings: {warning_count} ({100*warning_count/frame_num:.1f}%)")

    # Print statistics
    print("\n" + "=" * 60)
    print("SIMULATED GESTURE DETECTION STATISTICS")
    print("=" * 60)
    print(warning_system.get_warning_report())

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nOutput saved to: {output_path}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("DRIVER GESTURE DETECTION DEMO")
    print("(Simulation Mode - No MediaPipe Required)")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "test_videos/car-driver.mp4"

    save = '--save' in sys.argv

    demo_on_video(video_path, save_output=save)

    print("\n" + "=" * 60)
    print("NOTE: This is a simulation to demonstrate visualization")
    print("For actual gesture detection, run on Raspberry Pi with:")
    print("  python3 test_driver_gesture.py --mode video --source <video>")
    print("=" * 60)
