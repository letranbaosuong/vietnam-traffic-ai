#!/usr/bin/env python3
"""
Test Lane Departure Warning System
Test hệ thống cảnh báo chệch làn đường
"""

import cv2
import time
from pathlib import Path
from enhanced_lane_detector import EnhancedLaneDetector

def test_lane_departure_on_video(video_path, max_frames=300):
    """Test lane departure warning on a specific video"""
    print(f"🛣️ Testing Lane Departure Warning on: {Path(video_path).name}")
    print("=" * 60)

    # Initialize enhanced lane detector
    detector = EnhancedLaneDetector()

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
    warnings_detected = []
    processing_times = []

    # Output video setup
    output_path = f"outputs/lane_departure_test_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detect enhanced lanes with departure warning
        try:
            lane_data = detector.detect_enhanced_lanes(frame)

            # Draw results
            result_frame = detector.draw_enhanced_lanes(frame, lane_data)

            # Log warnings
            departure_info = lane_data['departure_info']
            confidence = lane_data['confidence']

            if departure_info['warning_level'] > 0 and confidence > detector.confidence_threshold:
                warning_data = {
                    'frame': frame_count,
                    'warning_type': departure_info['warning_type'],
                    'warning_level': departure_info['warning_level'],
                    'departure_ratio': departure_info['departure_ratio'],
                    'confidence': confidence
                }
                warnings_detected.append(warning_data)

                # Print warning
                warning_msg = detector.warning_messages.get(
                    departure_info['warning_type'], 'CẢNH BÁO!'
                )
                print(f"Frame {frame_count}: {warning_msg} (Level {departure_info['warning_level']})")

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
    print(f"\\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Warnings detected: {len(warnings_detected)}")

    if processing_times:
        avg_fps = len(processing_times) / sum(processing_times)
        avg_time = sum(processing_times) / len(processing_times)
        print(f"  - Average FPS: {avg_fps:.1f}")
        print(f"  - Average processing time: {avg_time*1000:.1f}ms/frame")

    # Warning statistics
    if warnings_detected:
        warning_counts = {}
        for warning in warnings_detected:
            warning_type = warning['warning_type']
            warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1

        print(f"\\n🚨 Warning Statistics:")
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
            warning_name = detector.warning_messages.get(warning_type, warning_type)
            print(f"  - {warning_name}: {count} times")

        # Warning levels breakdown
        level_counts = {}
        for warning in warnings_detected:
            level = warning['warning_level']
            level_counts[level] = level_counts.get(level, 0) + 1

        print(f"\\n📊 Warning Levels:")
        for level in sorted(level_counts.keys()):
            severity = ['None', 'Mild', 'Moderate', 'Severe'][level] if level <= 3 else 'Critical'
            print(f"  - Level {level} ({severity}): {level_counts[level]} times")

    print(f"\\n💾 Output saved: {output_path}")

def main():
    """Test lane departure warning on all available videos"""
    print("🛣️ Lane Departure Warning System Test")
    print("=" * 60)

    # Available test videos
    test_videos = [
        "video/sample.mp4",
        "video/car-driver.mp4",
        "video/mobycle-driver.mp4"
    ]

    for video_path in test_videos:
        if Path(video_path).exists():
            print(f"\\n{'='*60}")
            test_lane_departure_on_video(video_path, max_frames=200)
        else:
            print(f"⚠️ Video not found: {video_path}")

    print(f"\\n🎉 All tests complete!")
    print(f"📁 Check outputs/ directory for result videos")

    # Generate summary report
    print(f"\\n📋 Lane Departure Warning Features:")
    print(f"  ✅ Real-time lane detection with Canny + Hough Transform")
    print(f"  ✅ Vehicle position tracking")
    print(f"  ✅ Lane departure ratio calculation")
    print(f"  ✅ 3-level warning system (Mild/Moderate/Severe)")
    print(f"  ✅ Vietnamese warning messages")
    print(f"  ✅ Visual alerts with color coding")
    print(f"  ✅ Confidence-based filtering")
    print(f"  ✅ Temporal smoothing for stable detection")

if __name__ == "__main__":
    main()