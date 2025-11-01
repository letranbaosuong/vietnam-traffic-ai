#!/usr/bin/env python3
"""
Script để kiểm tra video driver có góc chính diện hay không
"""

import cv2
import os

def check_video_info(video_path):
    """Lấy thông tin và show preview của video"""
    if not os.path.exists(video_path):
        print(f"❌ Video không tồn tại: {video_path}")
        return

    print(f"\n{'='*70}")
    print(f"📹 Checking: {os.path.basename(video_path)}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video")
        return

    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Get file size
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB

    print(f"📊 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   Frames: {total_frames}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Size: {file_size:.1f} MB")
    print(f"\n📸 Preview frames:")

    # Show first, middle, and last frame info
    frames_to_check = [0, total_frames // 2, total_frames - 1]

    for frame_idx in frames_to_check:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # Save preview frame
            preview_name = f"preview_{os.path.basename(video_path).replace('.mp4', '')}_frame{frame_idx}.jpg"
            cv2.imwrite(preview_name, frame)
            print(f"   ✅ Frame {frame_idx}: Saved to {preview_name}")

    cap.release()
    print(f"\n💡 Để xem video, chạy:")
    print(f"   open {video_path}")


if __name__ == '__main__':
    # Videos to check
    videos = [
        "test_videos/car-driver.mp4",
        "test_videos/mobycle-driver.mp4",
        "/Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/raspberry-pi-traffic-detection/video/sample.mp4"
    ]

    print("🔍 DRIVER VIDEO CHECKER")
    print("Checking for frontal view driver videos...\n")

    for video in videos:
        check_video_info(video)

    print(f"\n{'='*70}")
    print("✅ Check complete! Review preview images to see video angles.")
    print("='*70}")
