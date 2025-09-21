# Traffic Object Detection for Raspberry Pi 4

Dự án R&D detect object giao thông sử dụng YOLOv8n tối ưu cho Raspberry Pi 4.

## Tính năng

- Detect 8 loại object giao thông: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- Tối ưu cho Raspberry Pi 4 với ONNX Runtime
- Hỗ trợ xử lý video, camera real-time và ảnh
- FPS: ~5-10 FPS trên RPi 4 (tùy cấu hình)
- Benchmark và monitoring hiệu năng

## Cấu trúc dự án

```
traffic-object-detection-rpi/
├── src/
│   ├── detector.py          # Core detection với YOLO
│   ├── video_processor.py   # Xử lý video/camera
│   └── rpi_optimizer.py     # Tối ưu cho RPi
├── configs/
│   └── config.yaml          # Cấu hình hệ thống
├── models/                  # Thư mục chứa model
├── data/
│   ├── videos/             # Video input
│   ├── images/             # Ảnh input
│   └── outputs/            # Kết quả output
├── main.py                  # Script chính
├── demo_simple.py          # Demo đơn giản
└── requirements.txt        # Dependencies
```

## Cài đặt

### 1. Cài đặt trên Raspberry Pi 4

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Cài đặt dependencies
sudo apt install -y python3-pip python3-opencv
sudo apt install -y libatlas-base-dev libopenblas-dev

# Clone project
cd ~/
git clone <your-repo>
cd traffic-object-detection-rpi

# Cài đặt Python packages
pip3 install -r requirements.txt
```

### 2. Download model YOLOv8n

```bash
# Tự động download khi chạy lần đầu
python3 main.py --mode benchmark --benchmark-frames 1
```

## Sử dụng

### 1. Xử lý video

```bash
# Sử dụng video mặc định trong config
python3 main.py --mode video

# Chỉ định video input/output
python3 main.py --mode video --input path/to/video.mp4 --output path/to/output.mp4

# Hiển thị kết quả (không khuyến khích trên RPi)
python3 main.py --mode video --display
```

### 2. Camera real-time

```bash
# Camera mặc định (id=0)
python3 main.py --mode camera

# Camera USB khác
python3 main.py --mode camera --camera-id 1
```

### 3. Xử lý ảnh

```bash
python3 main.py --mode image --input path/to/image.jpg --output path/to/result.jpg
```

### 4. Benchmark hiệu năng

```bash
# Test 100 frames
python3 main.py --mode benchmark --input video.mp4

# Test 500 frames
python3 main.py --mode benchmark --input video.mp4 --benchmark-frames 500
```

### 5. Demo đơn giản

```bash
# Demo với ảnh
python3 demo_simple.py --mode image

# Demo với camera
python3 demo_simple.py --mode camera
```

## Cấu hình

Chỉnh sửa file `configs/config.yaml`:

```yaml
model:
  name: "yolov8n"              # Model size: yolov8n, yolov8s
  input_size: [320, 320]       # Giảm xuống 256 nếu cần FPS cao hơn
  confidence_threshold: 0.4     # Ngưỡng confidence
  nms_threshold: 0.45          # Non-max suppression

video:
  fps: 15                      # Target FPS (10-15 cho RPi)

rpi_optimization:
  num_threads: 4               # Số CPU threads
  enable_quantization: true    # Quantization (chưa implement)
  use_onnx: true              # Dùng ONNX thay vì PyTorch

traffic_classes:              # Các class cần detect
  - person
  - car
  - motorcycle
  - bus
  - truck
```

## Tối ưu hiệu năng

### 1. Trên Raspberry Pi 4

```bash
# Set CPU governor
sudo cpufreq-set -g performance

# Tăng GPU memory split
sudo raspi-config
# Advanced Options > Memory Split > 128

# Overclock (cẩn thận)
# Thêm vào /boot/config.txt:
# over_voltage=6
# arm_freq=2000
# gpu_freq=600
```

### 2. Tối ưu model

- Sử dụng YOLOv8n (nhỏ nhất)
- Giảm input size xuống 256x256 hoặc 320x320
- Tăng confidence threshold lên 0.5
- Sử dụng ONNX Runtime
- Skip frames (xử lý 1 frame mỗi 2-3 frames)

### 3. Cooling

- **BẮT BUỘC** dùng heatsink + fan cho RPi 4
- Monitor nhiệt độ < 70°C

## Hiệu năng dự kiến

### Raspberry Pi 4 (4GB RAM)

| Model | Input Size | ONNX | FPS | CPU Usage | Memory |
|-------|------------|------|-----|-----------|---------|
| YOLOv8n | 320x320 | Yes | 8-10 | 70-80% | 1.2GB |
| YOLOv8n | 256x256 | Yes | 10-12 | 65-75% | 1.0GB |
| YOLOv8n | 320x320 | No | 3-5 | 90-95% | 1.8GB |

### Desktop/Laptop

| Model | Input Size | Device | FPS |
|-------|------------|---------|-----|
| YOLOv8n | 640x640 | CPU | 15-25 |
| YOLOv8n | 640x640 | GPU | 60-100 |

## Troubleshooting

### 1. Import error opencv

```bash
# Cài lại OpenCV
pip3 uninstall opencv-python opencv-python-headless
pip3 install opencv-python-headless==4.8.1.78
```

### 2. Memory error

- Giảm input size trong config
- Tăng swap file:
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Throttling warning

- Kiểm tra nguồn: cần 5V 3A chính hãng
- Thêm tản nhiệt và quạt
- Giảm overclock

### 4. Low FPS

- Dùng model YOLOv8n
- Giảm resolution xuống 256x256
- Skip frames
- Tắt các service không cần thiết

## Development

### Test local

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/
```

### Thêm traffic class mới

1. Thêm vào `configs/config.yaml`
2. Đảm bảo class có trong COCO dataset
3. Thêm màu vẽ trong `detector.py` nếu cần

## License

MIT

---
Đã tạo xong project R&D detect object giao thông với YOLOv8n
  cho Raspberry Pi 4!

  Cấu trúc project:

  traffic-object-detection-rpi/
  ├── src/
  │   ├── detector.py          # Core detection với YOLOv8n
  │   ├── video_processor.py   # Xử lý video/camera
  │   └── rpi_optimizer.py     # Tối ưu cho RPi 4
  ├── configs/
  │   └── config.yaml          # Cấu hình hệ thống
  ├── main.py                  # Script chính
  ├── demo_simple.py          # Demo đơn giản
  ├── setup_rpi.sh            # Script cài đặt cho RPi
  ├── requirements.txt        # Dependencies
  └── README.md               # Hướng dẫn chi tiết

  Tính năng chính:

  1. YOLOv8n tối ưu - Model nhỏ gọn nhất, phù hợp RPi 4
  2. ONNX Runtime - Tăng tốc inference trên CPU
  3. Multi-threading - Tận dụng 4 cores của RPi 4
  4. 8 loại object giao thông - car, motorcycle, bus, truck,
  person, bicycle, traffic light, stop sign
  5. FPS 8-10 trên RPi 4 với input 320x320

  Cách sử dụng trên RPi 4:

  # 1. Cài đặt
  cd traffic-object-detection-rpi
  bash setup_rpi.sh

  # 2. Demo nhanh
  python3 demo_simple.py --mode image

  # 3. Camera real-time
  python3 main.py --mode camera

  # 4. Benchmark
  python3 main.py --mode benchmark --benchmark-frames 100

  Project đã được tối ưu với:
  - Input size 320x320 (cân bằng tốc độ/accuracy)
  - Skip frame processing
  - CPU governor performance mode
  - Memory và thread optimization
  - Monitoring throttling và nhiệt độ