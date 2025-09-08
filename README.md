# 🚦 Vietnam Traffic AI System

Hệ thống AI phân tích giao thông Việt Nam tích hợp nhiều module AI tiên tiến:

## 🎯 Tính năng chính

### 🚗 Phát hiện đối tượng (YOLO)
- Nhận dạng xe cộ: ô tô, xe máy, xe buýt, xe tải
- Phát hiện người đi bộ và xe đạp
- Tối ưu đặc biệt cho xe máy (đặc trưng giao thông VN)
- Thống kê mật độ giao thông real-time

### 🚸 Nhận dạng biển báo (OCR + Computer Vision)
- Đọc biển báo tiếng Việt và tiếng Anh
- Phân loại các loại biển báo: cấm, chỉ dẫn, cảnh báo
- Phát hiện biển tốc độ, biển dừng, biển một chiều
- Xử lý trong điều kiện thời tiết khác nhau

### 🚶 Phân tích tư thế và hướng nhìn (MediaPipe)
- Theo dõi hướng nhìn và tư thế của người đi bộ
- Phát hiện cử chỉ ra hiệu giao thông
- Đánh giá mức độ rủi ro và hành vi
- Phân tích sự chú ý trong giao thông

## 🏗️ Cấu trúc dự án

```
vietnam-traffic-ai/
├── src/
│   ├── modules/               # Các module AI chính
│   │   ├── object_detection.py      # YOLO object detection
│   │   ├── traffic_sign_detection.py # Traffic sign recognition
│   │   └── pose_analysis.py         # Pose & gaze analysis
│   ├── utils/                # Tiện ích hỗ trợ
│   │   └── video_processor.py       # Video processing utilities
│   └── data_processing/      # Xử lý và tăng cường dữ liệu
│       └── data_augmentation.py     # Vietnam-specific augmentation
├── data/
│   ├── raw/                  # Dữ liệu thô (videos, images)
│   ├── processed/            # Dữ liệu đã xử lý
│   └── annotations/          # File gán nhãn YOLO format
├── models/                   # Các model AI (.pt, .onnx)
├── configs/                  # File cấu hình YAML
├── scripts/                  # Scripts tiện ích
│   ├── data_collection.py          # Thu thập dữ liệu
│   └── setup_environment.py        # Setup môi trường
├── demo/                     # Demo applications
│   └── streamlit_app.py           # Web demo
├── main.py                   # Ứng dụng chính
└── requirements.txt          # Python dependencies
```

## 🚀 Cài đặt nhanh

### Tự động (Khuyên dùng)
```bash
python scripts/setup_environment.py
```

### Thủ công
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Tạo thư mục
mkdir -p data/{raw,processed,annotations} models demo/outputs
```

## 🎮 Cách sử dụng

### 1. Demo Web Interface
```bash
# Windows
run_demo.bat

# Linux/Mac
./run_demo.sh
```

### 2. Command Line Interface

#### Webcam Real-time
```bash
python main.py --mode webcam
```

#### Xử lý Video
```bash
python main.py --mode video --input traffic_video.mp4 --output processed_video.mp4
```

#### Xử lý Batch Images
```bash
python main.py --mode images --input images_folder/ --output processed_images/
```

#### Phân tích thống kê
```bash
python main.py --mode analyze --input video.mp4 --output statistics.json
```

#### Tùy chỉnh modules
```bash
# Chỉ phát hiện đối tượng
python main.py --mode webcam --no-signs --no-pose

# Chỉ nhận dạng biển báo
python main.py --mode video --input video.mp4 --no-objects --no-pose
```

### 3. Thu thập dữ liệu

#### Thu thập từ webcam
```bash
python scripts/data_collection.py --mode webcam --duration 300
```

#### Trích xuất frames từ video
```bash
python scripts/data_collection.py --mode extract --video traffic.mp4 --interval 30
```

#### Tạo template annotation
```bash
python scripts/data_collection.py --mode annotate --images_dir data/raw/images
```

## ⚙️ Cấu hình

Chỉnh sửa `configs/config.yaml`:

```yaml
model:
  yolo_model: 'yolov8n.pt'    # Model size: n, s, m, l, x
  confidence: 0.5              # Ngưỡng confidence
  iou_threshold: 0.45          # NMS threshold

video:
  input_resolution: [1280, 720]
  output_fps: 30

# ... các cấu hình khác
```

## 📊 Ví dụ kết quả

### Object Detection
- **Xe máy**: 45 chiếc
- **Ô tô**: 12 chiếc  
- **Người đi bộ**: 8 người
- **Xe buýt**: 2 chiếc

### Traffic Signs
- **Biển cấm**: 3 biển
- **Biển tốc độ**: 2 biển (40km/h, 60km/h)
- **Biển chỉ dẫn**: 5 biển

### Pose Analysis
- **Mức rủi ro cao**: 2 người (không chú ý)
- **Đang ra hiệu**: 1 người
- **Di chuyển**: 5 người

## 🛠️ Phát triển

### Thêm dữ liệu huấn luyện
1. Thu thập video/ảnh giao thông VN
2. Gán nhãn bằng LabelImg hoặc Roboflow
3. Augmentation với `data_processing/data_augmentation.py`
4. Fine-tune model với dữ liệu mới

### Tùy chỉnh cho khu vực
- Điều chỉnh từ khóa biển báo trong `traffic_sign_detection.py`
- Thêm class mới trong `config.yaml`
- Training với dữ liệu địa phương

## 🔧 Yêu cầu hệ thống

### Phần cứng tối thiểu
- **RAM**: 4GB (khuyên 8GB+)
- **GPU**: Không bắt buộc (CPU OK, GPU tăng tốc)
- **Storage**: 2GB free space

### Phần mềm
- **Python**: 3.8+
- **OpenCV**: 4.10+
- **CUDA**: Tùy chọn (để tăng tốc GPU)

## 📈 Performance

| Mode | FPS | Accuracy | GPU Memory |
|------|-----|----------|------------|
| YOLOv8n | 30+ | 85% | 1GB |
| YOLOv8s | 25+ | 88% | 2GB |
| YOLOv8m | 20+ | 90% | 4GB |

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Tạo Pull Request

## 📝 License

MIT License - xem [LICENSE](LICENSE) để biết chi tiết.

## 🙏 Cảm ơn

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

## 📞 Liên hệ

- **GitHub**: [vietnam-traffic-ai](https://github.com/your-username/vietnam-traffic-ai)
- **Issues**: Báo cáo lỗi và đóng góp ý tưởng
- **Email**: your-email@example.com

---

**⭐ Nếu project hữu ích, hãy cho 1 star để ủng hộ!**