# Hệ Thống Phát Hiện Hành Vi Nguy Hiểm Của Người Lái Xe

## Giới thiệu
Hệ thống giám sát và cảnh báo hành vi nguy hiểm của người lái xe sử dụng Computer Vision và AI. Phát hiện các hành vi như buồn ngủ, mất tập trung, sử dụng điện thoại, hút thuốc khi lái xe.

## Tính năng chính

### 1. Phát hiện buồn ngủ (Drowsiness Detection)
- Tính toán Eye Aspect Ratio (EAR)
- Phát hiện mắt nhắm trong thời gian dài
- Cảnh báo khi phát hiện ngủ gật

### 2. Phát hiện ngáp (Yawning Detection)
- Tính toán Mouth Aspect Ratio (MAR)
- Phát hiện miệng mở rộng (ngáp)
- Cảnh báo khi ngáp liên tục

### 3. Phát hiện mất tập trung (Distraction Detection)
- Phân tích hướng đầu (Head Pose)
- Theo dõi hướng nhìn (Gaze Direction)
- Cảnh báo khi nhìn sang hai bên, nhìn xuống

### 4. Phát hiện sử dụng điện thoại (Phone Usage Detection)
- Nhận diện cử chỉ cầm điện thoại
- Phát hiện vị trí tay gần tai (gọi điện)
- Phát hiện nhìn điện thoại, nhắn tin

### 5. Phát hiện hút thuốc (Smoking Detection)
- Nhận diện cử chỉ hút thuốc
- Phát hiện tay gần miệng với tư thế đặc trưng
- Cảnh báo khi phát hiện hút thuốc

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Webcam hoặc camera
- RAM: 4GB minimum
- CPU: Intel Core i5 hoặc tương đương

### Cài đặt dependencies
```bash
cd driver-behavior-detection
pip install -r requirements.txt
```

## Sử dụng

### Chạy ứng dụng cơ bản
```bash
python main.py
```

### Chạy với các tùy chọn
```bash
# Sử dụng webcam (mặc định)
python main.py --source 0

# Sử dụng video file
python main.py --source path/to/video.mp4

# Bật ghi video khi phát hiện cảnh báo
python main.py --record

# Tắt hiển thị thống kê
python main.py --no-stats

# Bật chế độ debug
python main.py --debug
```

### Phím tắt
- `q`: Thoát chương trình
- `s`: Chụp ảnh màn hình
- `r`: Bật/tắt ghi video
- `d`: Bật/tắt chế độ debug

## Cấu trúc dự án
```
driver-behavior-detection/
├── modules/
│   ├── drowsiness_detector.py    # Module phát hiện buồn ngủ
│   ├── yawn_detector.py          # Module phát hiện ngáp
│   ├── distraction_detector.py   # Module phát hiện mất tập trung
│   ├── phone_detector.py         # Module phát hiện sử dụng điện thoại
│   └── smoking_detector.py       # Module phát hiện hút thuốc
├── utils/
│   ├── alert_system.py          # Hệ thống cảnh báo âm thanh
│   └── video_recorder.py        # Ghi video và log sự kiện
├── logs/                         # Thư mục lưu log và video
├── main.py                       # Ứng dụng chính
├── requirements.txt              # Dependencies
└── README.md                     # Tài liệu

```

## Ngưỡng cảnh báo

### Drowsiness (Buồn ngủ)
- EAR < 0.25 trong 15 frames liên tiếp
- Mức độ: CRITICAL

### Yawning (Ngáp)
- MAR > 0.5 trong 15 frames liên tiếp
- Mức độ: LOW

### Distraction (Mất tập trung)
- Góc quay đầu > 30 độ trong 20 frames
- Góc nhìn lệch > 35 độ trong 25 frames
- Mức độ: MEDIUM

### Phone Usage (Sử dụng điện thoại)
- Phát hiện tay gần tai/mặt trong 10 frames
- Mức độ: HIGH

### Smoking (Hút thuốc)
- Phát hiện cử chỉ hút thuốc trong 15 frames
- Mức độ: HIGH

## API và Module

### DrowsinessDetector
```python
from modules.drowsiness_detector import DrowsinessDetector

detector = DrowsinessDetector()
status, ear_value, frame = detector.detect(frame)
stats = detector.get_statistics()
```

### AlertSystem
```python
from utils.alert_system import AlertSystem

alert = AlertSystem()
message = alert.trigger_alert('DROWSY')
color = alert.get_alert_color('DROWSY')
```

## Hiệu suất
- FPS: 20-30 (tùy thuộc vào cấu hình máy)
- Độ chính xác phát hiện mắt nhắm: ~94%
- Độ chính xác phát hiện ngáp: ~96%
- Độ trễ cảnh báo: < 1 giây

## Lưu ý an toàn
- Hệ thống chỉ hỗ trợ cảnh báo, không thay thế ý thức an toàn của người lái
- Cần đảm bảo camera có góc nhìn rõ mặt người lái
- Hoạt động tốt nhất trong điều kiện ánh sáng đầy đủ
- Nên dừng xe nghỉ ngơi khi nhận được cảnh báo buồn ngủ

## Phát triển thêm
- Tích hợp với hệ thống xe hơi
- Thêm phát hiện uống rượu/bia
- Phát hiện sử dụng tai nghe
- Gửi cảnh báo qua app mobile
- Lưu trữ dữ liệu lên cloud

## Liên hệ
Để báo lỗi hoặc đóng góp, vui lòng tạo issue trên GitHub.