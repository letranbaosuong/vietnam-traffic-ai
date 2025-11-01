# Driver Gesture Detection for Raspberry Pi

## Tổng Quan

Hệ thống phát hiện cử chỉ và hành vi nguy hiểm của người lái xe, được tối ưu hóa cho Raspberry Pi. Sử dụng MediaPipe để detect các hành vi:

- ⚠️ **Sử dụng điện thoại** (gọi, nhắn tin, xem)
- ⚠️ **Mất tập trung** (nhìn sang trái/phải, lên/xuống)
- ⚠️ **Tay rời vô lăng**

## Tính Năng

### 1. Driver Gesture Detection
- Phát hiện real-time các cử chỉ nguy hiểm
- Tối ưu hóa cho Raspberry Pi (Model complexity 0)
- Hỗ trợ MediaPipe Hands, Pose, FaceMesh
- FPS: ~15-20 trên Pi 3, ~25-30 trên Pi 4

### 2. Warning System
- Cảnh báo bằng tiếng Việt
- 4 mức độ nguy hiểm: LOW, MEDIUM, HIGH, CRITICAL
- Visual warnings với màu sắc khác nhau
- Statistics tracking

### 3. Integration
- Tích hợp vào object detection + lane detection hiện có
- Có thể enable/disable độc lập
- Reuse code patterns từ driver-behavior-detection project

## Cấu Trúc Files

```
raspberry_pi/
├── driver_gesture_detector.py      # Core gesture detection module
├── gesture_warning_system.py       # Warning management system
├── detect.py                        # Main script (updated)
├── test_driver_gesture.py          # Test script
└── README_DRIVER_GESTURE.md         # Documentation này
```

## Cài Đặt

Cần thêm MediaPipe:

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Activate venv nếu đã có
source venv/bin/activate

# Install MediaPipe
pip install mediapipe
```

## Cách Sử Dụng

### 1. Test Gesture Detection Độc Lập

#### Test với Camera:
```bash
source venv/bin/activate

python3 test_driver_gesture.py --mode camera --source 0
```

#### Test với Video:
```bash
python3 test_driver_gesture.py --mode video --source /path/to/video.mp4 --save
```

Keyboard controls:
- **q** hoặc **ESC**: Thoát
- **s**: Hiển thị statistics
- **r**: Reset statistics (camera mode)

### 2. Chạy Tất Cả (Object + Lane + Gesture)

```bash
source venv/bin/activate

python3 detect.py \
  --model efficientdet_lite0.tflite \
  --cameraId 0 \
  --enableLaneDetection \
  --enableGestureDetection
```

### 3. Chỉ Object + Gesture (không lanes)

```bash
python3 detect.py \
  --model efficientdet_lite0.tflite \
  --cameraId 0 \
  --enableGestureDetection
```

### 4. Customize Detection Thresholds

```python
from driver_gesture_detector import DriverGestureDetector

# Create custom config
config = {
    'phone_thresh': 0.15,           # Distance threshold for phone detection
    'phone_frames': 20,             # Frames needed to confirm phone usage
    'distraction_yaw': 30,          # Head yaw angle threshold
    'distraction_pitch': 25,        # Head pitch angle threshold
    'distraction_frames': 15,       # Frames for distraction
    'hands_off_frames': 25          # Frames for hands off wheel
}

# Initialize with config
gesture_detector = DriverGestureDetector(config=config)
```

## Các Cử Chỉ Được Detect

### 1. Phone Usage (Sử Dụng Điện Thoại)

**Patterns:**
- **Phone Call**: Tay gần tai, góc nghiêng
- **Texting/Browsing**: Tay ở chest level, stable position
- **Looking at Phone**: Tay ở reading distance từ mặt

**Warning**: ⚠️ NGUY HIỂM: Đang gọi điện thoại! / Đang xem điện thoại!

**Threshold**: `PHONE_THRESH = 0.12` (distance), `PHONE_FRAMES = 15`

### 2. Distraction (Mất Tập Trung)

**Patterns:**
- **Looking Left/Right**: Head yaw > 25°
- **Looking Up/Down**: Head pitch > 20°
- Sử dụng head pose estimation (PnP algorithm)

**Warning**: ⚠️ MẤT TẬP TRUNG: Đang nhìn sang TRÁI (XX°)!

**Threshold**: `DISTRACTION_YAW = 25°`, `DISTRACTION_PITCH = 20°`, `FRAMES = 12`

### 3. Hands Off Wheel (Tay Rời Vô Lăng)

**Patterns:**
- No hands detected
- Hands not at driving position (chest level)

**Warning**: ⚠️ CẢNH BÁO: Tay rời vô lăng!

**Threshold**: `HANDS_OFF_FRAMES = 20`

## Code Structure

### driver_gesture_detector.py

Main gesture detection logic:

```python
class DriverGestureDetector:
    def __init__(self, config=None):
        # Initialize MediaPipe
        self.hands = mp.solutions.hands.Hands(...)
        self.pose = mp.solutions.pose.Pose(...)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(...)

    def detect(self, frame):
        """
        Main detection method
        Returns: (warnings_list, annotated_frame)
        """
        # Detect phone usage
        # Detect distraction
        # Detect hands off wheel
        return warnings, frame

    def get_statistics(self):
        """Get detection statistics"""
        return stats_dict
```

### gesture_warning_system.py

Warning management:

```python
class GestureWarningSystem:
    def __init__(self, config=None):
        # Warning levels, categories, messages

    def add_warning(self, warning_text, category):
        """Add warning with cooldown check"""

    def draw_warnings(self, frame, warnings):
        """Draw visual warnings on frame"""

    def get_statistics(self):
        """Get warning statistics"""
```

## Performance

### Expected FPS

| Device | Resolution | Gesture Only | All (Object+Lane+Gesture) |
|--------|------------|--------------|---------------------------|
| macOS | 640x480 | 40-50 FPS | 20-25 FPS |
| Pi 4 | 640x480 | 25-30 FPS | 12-15 FPS |
| Pi 3 | 640x480 | 15-20 FPS | 8-10 FPS |

### Optimization Tips

1. **Giảm Resolution**:
   ```bash
   python3 detect.py --frameWidth 320 --frameHeight 240
   ```

2. **Skip Frames**:
   ```python
   # In detect.py, thêm vào gesture detection
   if frame_count % 2 == 0:  # Process every 2 frames
       warnings, image = gesture_detector.detect(image)
   ```

3. **Disable Unused Features**:
   ```bash
   # Chỉ gesture, không có lane
   python3 detect.py --enableGestureDetection
   ```

## Warning Levels

| Level | Color | Priority | Use Cases |
|-------|-------|----------|-----------|
| LOW | Yellow | 1 | Minor issues |
| MEDIUM | Orange | 2 | Distraction, hands off wheel |
| HIGH | Red | 3 | Phone usage |
| CRITICAL | Purple | 4 | Drowsiness (future) |

## Reused Patterns

Code patterns được reuse từ existing project:

### From `phone_detector.py`:
- Hand position detection
- Phone usage patterns (call, texting, viewing)
- Counter-based confirmation

### From `distraction_detector.py`:
- Head pose calculation (PnP algorithm)
- Direction detection (left, right, up, down)
- Gaze analysis

### From `main.py`:
- Alert overlay system
- Status panel
- Statistics tracking

## Troubleshooting

### 1. MediaPipe không detect được

**Nguyên nhân**: Lighting kém, góc camera không phù hợp

**Giải pháp**:
- Tăng lighting
- Adjust camera position
- Lower confidence threshold:
  ```python
  config = {'min_detection_confidence': 0.5}
  ```

### 2. FPS thấp

**Nguyên nhân**: Pi quá tải

**Giải pháp**:
- Giảm resolution
- Increase model_complexity=0
- Skip frames
- Disable unused features

### 3. False Positives

**Nguyên nhân**: Thresholds quá nhạy

**Giải pháp**:
- Tăng frame thresholds
- Tăng distance/angle thresholds
- Add temporal smoothing

## Examples

### Example 1: Basic Gesture Detection

```python
from driver_gesture_detector import DriverGestureDetector
import cv2

detector = DriverGestureDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warnings, annotated_frame = detector.detect(frame)

    # Print warnings
    for warning in warnings:
        print(warning)

    cv2.imshow('Gesture Detection', annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: With Warning System

```python
from driver_gesture_detector import DriverGestureDetector
from gesture_warning_system import GestureWarningSystem
import cv2

detector = DriverGestureDetector()
warning_system = GestureWarningSystem()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect gestures
    warnings, annotated_frame = detector.detect(frame)

    # Manage warnings
    warning_system.clear_old_warnings(max_age=2.0)
    for warning in warnings:
        if "điện thoại" in warning.lower():
            warning_system.add_warning(warning, 'phone_usage')

    # Visualize
    output = warning_system.draw_warnings(annotated_frame, warnings)
    output = warning_system.draw_status_bar(output)

    cv2.imshow('Driver Safety', output)
    if cv2.waitKey(1) == ord('q'):
        break

# Print report
print(warning_system.get_warning_report())

cap.release()
cv2.destroyAllWindows()
```

## Future Improvements

- [ ] Add drowsiness detection (eye aspect ratio)
- [ ] Add yawn detection
- [ ] Add smoking detection
- [ ] Audio alerts
- [ ] Data logging to file
- [ ] Cloud integration for fleet management

## References

- [MediaPipe Hand Landmarks](https://google.github.io/mediapipe/solutions/hands.html)
- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [driver-behavior-detection project](../../driver-behavior-detection/)

## Support

Nếu gặp vấn đề, check:
1. `gesture_detect.md` - Research và guidelines
2. `driver-behavior-detection/` - Reference implementation
3. GitHub issues

---

**Created**: 2025-11-01
**Version**: 1.0
**Optimized for**: Raspberry Pi 3/4
