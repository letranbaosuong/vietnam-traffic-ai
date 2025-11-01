# 📹 Hướng Dẫn: Lấy Video Driver Góc Chính Diện

**Date**: 2025-11-01
**Purpose**: Hướng dẫn các cách lấy video driver góc chính diện để test gesture detection

---

## 🔍 Kiểm Tra Video Hiện Có

### ❌ Video Hiện Tại Không Phù Hợp:

1. **car-driver.mp4** (3.5 MB, 10.3s)
   - Góc: Từ ghế sau, nhìn driver từ phía sau
   - ❌ Không thấy mặt chính diện
   - ✅ Có thể test hand detection và distraction (limited)

2. **mobycle-driver.mp4** (4.9 MB, 15.8s)
   - Góc: POV từ driver (first person)
   - ❌ Không thấy driver
   - ❌ Không phù hợp

3. **sample.mp4** (14.2 MB, 42.5s)
   - Góc: Traffic monitoring từ cầu vượt
   - ❌ Không có driver
   - ❌ Không phù hợp

---

## ✅ Các Cách Lấy Video Driver Chính Diện

### 🎥 Option 1: Dùng Webcam Laptop (Nhanh nhất!)

**Ưu điểm**:
- Miễn phí, nhanh chóng
- Góc chính diện hoàn hảo
- Có thể test ngay

**Cách làm**:

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Test với webcam (simulation mode)
source venv/bin/activate
python3 demo_gesture_simulation.py 0 --save

# Hoặc test thật với MediaPipe (cần Pi)
python3 test_driver_gesture.py --mode camera --source 0
```

**Recording webcam video**:

```python
# Tạo script record_webcam.py
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('frontal_driver_test.mp4', fourcc, 30.0, (640, 480))

print("Recording... Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved: frontal_driver_test.mp4")
```

Chạy:
```bash
python3 record_webcam.py
```

---

### 📱 Option 2: Dùng Điện Thoại

**Ưu điểm**:
- Chất lượng cao
- Linh hoạt
- Có thể quay trong xe thật

**Cách làm**:

1. **Setup camera trên điện thoại**:
   - Đặt điện thoại trên dashboard hoặc kính chắn gió
   - Góc nhìn: Camera hướng về mặt driver
   - Vị trí: Cách mặt khoảng 50-80cm

2. **Quay video** (10-30 giây):
   - Giả lập các hành vi:
     - ✅ Nhìn thẳng (normal)
     - ⚠️ Nhìn sang trái/phải
     - ⚠️ Giả lập gọi điện
     - ⚠️ Tay rời vô lăng
     - ⚠️ Nhìn xuống điện thoại

3. **Chuyển video vào project**:
   ```bash
   # Copy từ điện thoại
   cp /path/to/phone/video.mp4 test_videos/frontal_driver_phone.mp4

   # Hoặc AirDrop (macOS)
   # Sau đó move vào test_videos/
   ```

---

### 🌐 Option 3: Download Sample từ YouTube

**Ưu điểm**:
- Nhiều video mẫu
- Miễn phí
- Đa dạng scenarios

**Cách làm**:

1. **Tìm video trên YouTube**:
   - Search: "driver monitoring system demo"
   - Search: "driver drowsiness detection test"
   - Search: "driver distraction detection sample"

2. **Download với yt-dlp**:
   ```bash
   # Install yt-dlp
   pip install yt-dlp

   # Download video
   yt-dlp -f 'best[height<=720]' -o 'test_videos/youtube_driver.mp4' <YOUTUBE_URL>

   # Example searches:
   # - "driver monitoring system demonstration"
   # - "driver face detection test video"
   # - "DMS camera sample footage"
   ```

**Recommended searches**:
- "driver monitoring system test"
- "DMS camera demo"
- "driver drowsiness detection video"
- "driver distraction detection sample"

---

### 🎓 Option 4: Public Datasets (Advanced)

**Datasets với frontal driver views**:

1. **YawDD Dataset**:
   - 107 participants (57 males, 50 females)
   - Frontal camera view
   - Different expressions: normal, talking, yawning
   - Real conditions, varied illumination
   - With/without glasses, sunglasses
   - Link: Search "YawDD dataset download"

2. **NTHU-DDD Dataset**:
   - 9.5 hours of video
   - 18 participants
   - Drowsy and non-drowsy states
   - Link: Search "NTHU-DDD dataset"

3. **UTA-RLDD Dataset**:
   - Real-Life Drowsiness Dataset
   - Frontal camera
   - Link: Search "UTA-RLDD dataset"

**Note**: Các dataset này thường yêu cầu request access từ university.

---

## 🚀 Test Ngay với Video Hiện Có

Mặc dù không phải frontal view, bạn vẫn có thể test system:

### ✅ Demo Đã Chạy Xong:

```bash
# Xem kết quả demo
open driver_gesture_demo_output.mp4
```

**Kết quả**:
- ✅ 308 frames processed
- ✅ 47.56 FPS
- ✅ 5 warnings detected
- ✅ Visualization working perfectly

### 📊 Statistics:

```
Total Warnings: 5
- Phone usage: 1 (HIGH)
- Distraction: 3 (MEDIUM)
- Hands off wheel: 1 (MEDIUM)

Frames with warnings: 139 (45.1%)
```

---

## 💡 Recommendation

**Giải pháp tốt nhất**:

1. **Ngay bây giờ**: Xem demo output đã có
   ```bash
   open driver_gesture_demo_output.mp4
   ```

2. **Test nhanh**: Dùng webcam laptop
   ```bash
   source venv/bin/activate
   python3 test_driver_gesture.py --mode camera --source 0
   ```

3. **Test thật**: Quay video bằng điện thoại trong xe
   - Setup camera hướng vào mặt driver
   - Quay 10-30 giây
   - Copy vào test_videos/
   - Run test

4. **Professional**: Download sample từ YouTube hoặc dataset

---

## 🎬 Script Tạo Video Test Nhanh

Tạo file `record_test_video.py`:

```python
#!/usr/bin/env python3
"""
Record frontal driver video for testing
"""
import cv2
import time

def record_test_video(output_path='test_videos/frontal_driver_webcam.mp4', duration=30):
    """Record video from webcam"""

    cap = cv2.VideoCapture(0)

    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Recording {duration}s video to: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print("\nInstructions:")
    print("  - Sit in front of camera (50-80cm)")
    print("  - Simulate different gestures:")
    print("    ✓ Look straight (5s)")
    print("    ✓ Look left/right (5s)")
    print("    ✓ Phone near ear (5s)")
    print("    ✓ Hands off wheel (5s)")
    print("    ✓ Look down (5s)")
    print("\nPress 'q' to stop early, or wait for auto-stop")
    print("-" * 60)

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write frame
        out.write(frame)
        frame_count += 1

        # Calculate elapsed
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        # Draw countdown
        text = f"Recording: {elapsed:.1f}s / {duration}s (Press 'q' to stop)"
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Recording Test Video', frame)

        # Check for quit or timeout
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break

        if elapsed >= duration:
            print(f"\n{duration}s completed")
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Video saved: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Duration: {frame_count / fps:.1f}s")
    print(f"\nTo test with this video:")
    print(f"  python3 test_driver_gesture.py --mode video --source {output_path}")


if __name__ == '__main__':
    import sys

    duration = 30
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])

    record_test_video(duration=duration)
```

**Sử dụng**:
```bash
# Record 30 seconds
python3 record_test_video.py

# Record custom duration
python3 record_test_video.py 60
```

---

## 📋 Summary

| Option | Speed | Quality | Cost | Frontal View |
|--------|-------|---------|------|--------------|
| Webcam | ⚡ Fast | ⭐⭐⭐ | Free | ✅ Perfect |
| Phone | ⚡ Fast | ⭐⭐⭐⭐⭐ | Free | ✅ Perfect |
| YouTube | 🔄 Medium | ⭐⭐⭐⭐ | Free | ✅ Good |
| Dataset | 🐌 Slow | ⭐⭐⭐⭐⭐ | Free | ✅ Perfect |
| Current Videos | ⚡ Ready | ⭐⭐ | Free | ❌ No |

**Recommendation**:
1. ✅ Xem demo output hiện tại để thấy visualization
2. ✅ Record webcam video để test thật (5 phút)
3. ✅ Quay video bằng phone trong xe (professional)

---

**Created**: 2025-11-01
**Status**: Ready to use
**Next**: Record webcam video hoặc download sample từ YouTube
