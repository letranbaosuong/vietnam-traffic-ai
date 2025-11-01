#!/usr/bin/env python3
"""
Record frontal driver video for testing gesture detection
Quick tool to create test videos using webcam
"""
import cv2
import time
import os

def record_test_video(output_path='test_videos/frontal_driver_webcam.mp4', duration=30):
    """Record video from webcam for driver gesture testing"""

    # Ensure test_videos directory exists
    os.makedirs('test_videos', exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        print("Please check:")
        print("  - Camera permissions in System Preferences")
        print("  - Camera is not being used by another app")
        return

    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("=" * 70)
    print("📹 RECORDING TEST VIDEO FOR DRIVER GESTURE DETECTION")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Duration: {duration}s")
    print()
    print("📝 Instructions for Best Results:")
    print("-" * 70)
    print("  1. Sit 50-80cm from camera")
    print("  2. Ensure good lighting (face clearly visible)")
    print("  3. Simulate different gestures during recording:")
    print()
    print("     Seconds 0-5:   ✅ Look straight (normal driving)")
    print("     Seconds 5-10:  ⚠️  Look left/right (distraction)")
    print("     Seconds 10-15: ⚠️  Phone near ear (phone call)")
    print("     Seconds 15-20: ⚠️  Hands down (hands off wheel)")
    print("     Seconds 20-25: ⚠️  Look down (looking at phone)")
    print("     Seconds 25-30: ✅ Back to normal")
    print()
    print("-" * 70)
    print("Press 'q' to stop early, or recording will auto-stop")
    print("=" * 70)
    print()

    input("Press ENTER when ready to start recording...")
    print("\n🔴 RECORDING...")
    print("-" * 70)

    start_time = time.time()
    frame_count = 0
    last_instruction_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            break

        # Write frame
        out.write(frame)
        frame_count += 1

        # Calculate elapsed
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # Mirror for better UX
        display_frame = cv2.flip(frame, 1)

        # Draw recording info
        info_text = f"Recording: {int(elapsed)}s / {duration}s"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw current instruction
        instruction = ""
        if elapsed < 5:
            instruction = "Look straight (normal)"
        elif elapsed < 10:
            instruction = "Look left/right (distraction)"
        elif elapsed < 15:
            instruction = "Phone near ear (call)"
        elif elapsed < 20:
            instruction = "Hands down (off wheel)"
        elif elapsed < 25:
            instruction = "Look down (at phone)"
        else:
            instruction = "Back to normal"

        cv2.putText(display_frame, instruction, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Progress bar
        bar_width = width - 20
        progress = int((elapsed / duration) * bar_width)
        cv2.rectangle(display_frame, (10, height - 30), (10 + progress, height - 10),
                     (0, 255, 0), -1)
        cv2.rectangle(display_frame, (10, height - 30), (10 + bar_width, height - 10),
                     (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Recording Test Video (Press Q to stop)', display_frame)

        # Print instruction changes
        if int(elapsed) != last_instruction_time:
            last_instruction_time = int(elapsed)
            print(f"  [{int(elapsed)}s] {instruction}")

        # Check for quit or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Stopped by user")
            break

        if elapsed >= duration:
            print(f"\n✅ {duration}s completed")
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    actual_duration = frame_count / fps

    print()
    print("=" * 70)
    print("✅ VIDEO RECORDING COMPLETE")
    print("=" * 70)
    print(f"Output file: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {actual_duration:.1f}s")
    print(f"Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print()
    print("🚀 Next Steps:")
    print("-" * 70)
    print(f"1. View the video:")
    print(f"   open {output_path}")
    print()
    print(f"2. Test with gesture detection simulation:")
    print(f"   python3 demo_gesture_simulation.py {output_path} --save")
    print()
    print(f"3. Test with actual MediaPipe (on Raspberry Pi):")
    print(f"   python3 test_driver_gesture.py --mode video --source {output_path}")
    print()
    print("=" * 70)


if __name__ == '__main__':
    import sys

    print("\n")
    print("🎬 Driver Gesture Test Video Recorder")
    print("=" * 70)
    print()

    # Parse arguments
    duration = 30
    output = 'test_videos/frontal_driver_webcam.mp4'

    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}")
            print("Usage: python3 record_test_video.py [duration_seconds] [output_path]")
            sys.exit(1)

    if len(sys.argv) > 2:
        output = sys.argv[2]

    print(f"Duration: {duration} seconds")
    print(f"Output: {output}")
    print()

    record_test_video(output_path=output, duration=duration)
