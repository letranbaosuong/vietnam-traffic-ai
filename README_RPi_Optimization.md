# 🚀 Vietnam Traffic AI - Raspberry Pi Optimization Guide

## Hướng dẫn tối ưu cho Raspberry Pi real-time detection

### 🎯 Cải tiến chính

1. **Model tối ưu**: YOLO11n với NCNN format - nhanh hơn 62% so với PyTorch
2. **Performance**: 8-15 FPS trên RPi 5, 5-8 FPS trên RPi 4  
3. **Code đơn giản**: Dễ hiểu và maintain cho người Việt
4. **Traffic focus**: Tập trung xe máy và giao thông VN

## 📋 System Requirements

### Hardware
- **Khuyên dùng**: Raspberry Pi 5 (8GB RAM)
- **Tối thiểu**: Raspberry Pi 4 (4GB RAM)
- Camera USB hoặc Pi Camera
- MicroSD card 32GB+ (Class 10)

### Software
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv libopencv-dev -y

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install ultralytics opencv-python numpy pyyaml
```

## 🚀 Quick Start

### 1. Clone và setup
```bash
git clone <repo-url>
cd vietnam-traffic-ai

# Chạy demo tối ưu cho RPi
python raspberry_pi_demo.py
```

### 2. Điều khiển demo
- `q`: Thoát
- `s`: Bật/tắt statistics
- `f`: Hiển thị FPS  
- `c`: Lưu ảnh
- `SPACE`: Pause/Resume

## 🔧 Tối ưu Performance

### 1. Model Optimization
```python
# Sử dụng NCNN format (tự động export)
detector = OptimizedDetector(use_ncnn=True)

# Model sẽ tự động export sang NCNN lần đầu chạy
# File: yolo11n_ncnn_model (thư mục)
```

### 2. Camera Settings
```python
# Tối ưu camera cho RPi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm latency
```

### 3. Inference Settings
```python
# Input size nhỏ hơn để tăng tốc
input_size = (416, 416)  # thay vì 640x640

# Confidence cao hơn để giảm false positives
confidence = 0.4  # thay vì 0.25

# Chỉ detect objects quan trọng
traffic_classes = {
    0: 'person',
    2: 'car', 
    3: 'motorcycle',  # Focus xe máy VN
    5: 'bus',
    7: 'truck'
}
```

## 📊 Performance Benchmarks

### Raspberry Pi 5
- **YOLOv8n PyTorch**: ~3-5 FPS
- **YOLO11n NCNN**: ~8-15 FPS ⚡
- **Memory usage**: ~500MB
- **CPU usage**: ~70-80%

### Raspberry Pi 4  
- **YOLOv8n PyTorch**: ~1-3 FPS
- **YOLO11n NCNN**: ~5-8 FPS ⚡
- **Memory usage**: ~400MB
- **CPU usage**: ~85-95%

## 🏍️ Vietnam Traffic Focus

### Classes được detect
```python
classes = {
    'person': 'Người đi bộ',        # Màu xanh lá
    'bicycle': 'Xe đạp',           # Màu vàng  
    'car': 'Ô tô',                # Màu đỏ
    'motorcycle': 'Xe máy',        # Màu tím - ưu tiên
    'bus': 'Xe buýt',             # Màu cyan
    'truck': 'Xe tải',            # Màu tím đậm
    'traffic_light': 'Đèn giao thông' # Màu cam
}
```

### Statistics
- **Total objects**: Tổng đối tượng phát hiện
- **Vehicles**: Tổng phương tiện  
- **People**: Số người
- **Motorcycles**: Số xe máy (focus VN)

## 🛠️ Troubleshooting

### Camera không hoạt động
```bash
# Kiểm tra camera
ls /dev/video*

# Add user vào video group
sudo usermod -a -G video $USER

# Reboot
sudo reboot
```

### FPS thấp
1. **Giảm resolution**: 640x480 thay vì 1280x720
2. **Tăng confidence**: 0.5 thay vì 0.25
3. **Disable threading**: Nếu RPi cũ
4. **Overclock**: (cẩn thận với nhiệt độ)

### Memory issues
```bash
# Tăng GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128MB

# Tăng swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## 🔄 Model Updates

### Export model mới
```python
from ultralytics import YOLO

# Load model mới
model = YOLO("yolo11s.pt")  # hoặc model khác

# Export sang NCNN
model.export(format="ncnn", imgsz=416)

# Model sẽ tạo thư mục: yolo11s_ncnn_model/
```

### Custom training cho VN traffic
```python
# Train với dataset Vietnam traffic
model = YOLO("yolo11n.pt")
model.train(
    data="vietnam_traffic.yaml",  # Custom dataset
    epochs=100,
    imgsz=416,
    batch=8,
    device="cpu"  # Hoặc GPU nếu có
)
```

## 📱 Integration Ideas

### 1. IoT Integration
```python
# Gửi data qua MQTT
import paho.mqtt.client as mqtt

def send_traffic_data(stats):
    client = mqtt.Client()
    client.connect("your-mqtt-broker", 1883, 60)
    client.publish("traffic/stats", json.dumps(stats))
```

### 2. Web Dashboard  
```python
# Flask web interface
from flask import Flask, render_template
import cv2

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')
```

### 3. Alert System
```python
# Cảnh báo khi có quá nhiều xe
def check_traffic_alert(stats):
    if stats['motorcycles'] > 10:
        send_alert("High motorcycle traffic detected!")
    if stats['people'] > 5:
        send_alert("Many pedestrians detected!")
```

## 🎯 Next Steps

1. **Custom dataset**: Train với data giao thông VN
2. **Edge deployment**: Sử dụng TensorRT, OpenVINO
3. **Multi-camera**: Hệ thống nhiều camera
4. **Cloud integration**: Upload data lên cloud
5. **Mobile app**: App mobile để monitor

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork repo
2. Create feature branch
3. Test trên RPi  
4. Submit PR với performance benchmarks

## 📞 Support

- **Issues**: GitHub Issues
- **Email**: your-email@domain.com
- **Community**: Discord/Telegram group

---

**Happy coding! 🇻🇳 🚗 🏍️**