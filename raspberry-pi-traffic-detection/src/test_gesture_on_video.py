#!/usr/bin/env python3
"""
Test Hand Gesture Detection on Driving Videos
Test phát hiện cử chỉ tay trên video lái xe
"""

import cv2
import time
from pathlib import Path
from hand_gesture_detector import HandGestureDetector

def test_gesture_on_video(video_path, max_frames=300):
    """Test gesture detection on a specific video"""
    print(f"🤚 Testing Hand Gesture Detection on: {Path(video_path).name}")
    print("=" * 60)

    # Initialize gesture detector
    detector = HandGestureDetector()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📺 Video: {width}x{height}, {fps} FPS, {total_frames} total frames")
    print(f"🎬 Processing {min(max_frames, total_frames)} frames...")

    # Performance tracking
    frame_count = 0
    gesture_detections = []
    processing_times = []

    # Output video setup
    output_path = f"outputs/gesture_test_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detect gestures
        try:
            gestures, mp_results = detector.detect_gestures(frame)

            # Draw results
            result_frame = detector.draw_gestures(frame, gestures, mp_results)

            # Generate alerts
            alerts = detector.generate_gesture_alerts(gestures)

            # Log detected gestures
            for gesture_info in gestures:
                gesture_detections.append({
                    'frame': frame_count,
                    'hand_side': gesture_info['hand_side'],
                    'gesture': gesture_info['gesture'],
                    'gesture_name': gesture_info['gesture_name']
                })

            # Log alerts
            for alert in alerts:
                print(f"Frame {frame_count}: {alert['message']}")

        except Exception as e:
            print(f"⚠️ Error processing frame {frame_count}: {e}")
            result_frame = frame

        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Write output frame
        out.write(result_frame)

        frame_count += 1

        # Progress update
        if frame_count % 50 == 0:
            avg_fps = len(processing_times) / sum(processing_times)
            print(f"  📊 Frame {frame_count}/{min(max_frames, total_frames)} - Avg FPS: {avg_fps:.1f}")

    # Cleanup
    cap.release()
    out.release()

    # Generate summary
    print(f"\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Gestures detected: {len(gesture_detections)}")

    if processing_times:
        avg_fps = len(processing_times) / sum(processing_times)
        avg_time = sum(processing_times) / len(processing_times)
        print(f"  - Average FPS: {avg_fps:.1f}")
        print(f"  - Average processing time: {avg_time*1000:.1f}ms/frame")

    # Gesture statistics
    if gesture_detections:
        gesture_counts = {}
        for detection in gesture_detections:
            gesture = detection['gesture']
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        print(f"\n🤚 Gesture Statistics:")
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            gesture_name = detector.traffic_gestures.get(gesture, gesture)
            print(f"  - {gesture_name}: {count} times")

    print(f"\n💾 Output saved: {output_path}")

def main():
    """Test gesture detection on all available videos"""
    print("🤚 Hand Gesture Detection Test on Driving Videos")
    print("=" * 60)

    # Available test videos
    test_videos = [
        "video/sample.mp4",
        "video/car-driver.mp4",
        "video/mobycle-driver.mp4"
    ]

    for video_path in test_videos:
        if Path(video_path).exists():
            print(f"\n{'='*60}")
            test_gesture_on_video(video_path, max_frames=200)
        else:
            print(f"⚠️ Video not found: {video_path}")

    print(f"\n🎉 All tests complete!")
    print(f"📁 Check outputs/ directory for result videos")

if __name__ == "__main__":
    main()