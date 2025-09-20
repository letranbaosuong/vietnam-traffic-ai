# Raspberry Pi Driver Monitoring System

Phiên bản tối ưu cho Raspberry Pi với các model nhẹ và hiệu suất cao.

## 📋 Yêu cầu phần cứng

- **Raspberry Pi 4** (4GB RAM recommended) hoặc **Raspberry Pi 5**
- Raspberry Pi Camera Module v2 hoặc USB Webcam
- MicroSD card (16GB minimum)
- Optional: Coral USB Accelerator cho tăng tốc AI

## 🚀 Tính năng

- ✅ Face detection với TFLite MobileNet
- ✅ Drowsiness detection (phát hiện ngủ gật)
- ✅ Head pose estimation (phát hiện mất tập trung)
- ✅ Real-time processing (~20-30 FPS trên Pi 4)
- ✅ Cảnh báo âm thanh và visual
- ✅ Low CPU usage (~40-60%)

## 🛠️ Cài đặt

### 1. Cập nhật hệ thống
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Cài đặt dependencies
```bash
sudo apt install -y python3-pip python3-opencv python3-numpy
sudo apt install -y libatlas-base-dev libhdf5-dev
sudo apt install -y libqt5gui5 libqt5webkit5 libqt5test5
```

### 3. Cài đặt Python packages
```bash
pip3 install -r requirements.txt
```

### 4. Cài đặt TFLite Runtime
```bash
# For Raspberry Pi 4 (32-bit OS)
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl

# For Raspberry Pi 4 (64-bit OS)
pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```

## 📦 Models sử dụng

### 1. **MobileNet SSD v2 (Recommended)**
- Size: ~6.5 MB
- FPS: 15-20 on Pi 4
- Accuracy: Good
- Best for: General face detection

### 2. **MobileNet v1 0.75 depth**
- Size: ~4.3 MB
- FPS: 20-25 on Pi 4
- Accuracy: Medium
- Best for: Speed priority

### 3. **EfficientDet-Lite0**
- Size: ~4.4 MB
- FPS: 10-15 on Pi 4
- Accuracy: Very Good
- Best for: Accuracy priority

## 🏃 Chạy ứng dụng

### Sử dụng Pi Camera
```bash
python3 main_pi.py --camera pi
```

### Sử dụng USB Camera
```bash
python3 main_pi.py --camera usb --device 0
```

### Với Coral USB Accelerator
```bash
python3 main_pi.py --use-coral
```

## ⚙️ Cấu hình

Chỉnh sửa `config_pi.yaml`:

```yaml
# Camera settings
camera:
  type: "pi"  # "pi" or "usb"
  resolution: [640, 480]
  fps: 20

# Model settings
model:
  type: "mobilenet_v2"
  confidence_threshold: 0.5
  use_coral: false

# Detection settings
detection:
  drowsiness_threshold: 0.25
  distraction_angle: 20
  consecutive_frames: 10

# Performance
performance:
  num_threads: 4
  skip_frames: 2  # Process every 2nd frame
```

## 📊 Performance Benchmarks

| Model | Raspberry Pi 4 | Raspberry Pi 5 | With Coral |
|-------|---------------|---------------|------------|
| MobileNet v1 | 20-25 FPS | 35-40 FPS | 60+ FPS |
| MobileNet v2 | 15-20 FPS | 30-35 FPS | 55+ FPS |
| EfficientDet-Lite0 | 10-15 FPS | 20-25 FPS | 40+ FPS |

## 🔧 Optimization Tips

1. **Giảm độ phân giải**: 640x480 thay vì 1920x1080
2. **Skip frames**: Xử lý mỗi 2-3 frames
3. **Multi-threading**: Sử dụng 4 threads
4. **Quantization**: Dùng INT8 models
5. **GPU acceleration**: Cài đặt GPU drivers nếu có

## 📱 Web Interface

Truy cập monitoring dashboard:
```
http://<raspberry-pi-ip>:5000
```

## 🐛 Troubleshooting

### Camera không hoạt động
```bash
# Enable camera
sudo raspi-config
# Navigate to Interface Options > Camera > Enable

# Test camera
libcamera-hello
```

### Low FPS
- Giảm resolution
- Tăng skip_frames
- Sử dụng model nhẹ hơn

### High CPU temperature
```bash
# Check temperature
vcgencmd measure_temp

# Add cooling or reduce load
```

## 📚 Resources

- [TensorFlow Lite on Raspberry Pi](https://www.tensorflow.org/lite/guide/python)
- [Coral USB Accelerator](https://coral.ai/products/accelerator)
- [Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)

## 📄 License

MIT License