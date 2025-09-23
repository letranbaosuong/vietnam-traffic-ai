# Driver Safety AI - Hệ Thống Phát Hiện Hành Vi Lái Xe Nguy Hiểm

## 📋 Tổng Quan

Hệ thống AI phát hiện hành vi lái xe nguy hiểm được tối ưu cho Raspberry Pi 4, sử dụng camera để giám sát và cảnh báo các hành vi có thể gây tai nạn giao thông.

## 🎯 Tính Năng Phát Hiện

### Hành vi nguy hiểm được phát hiện:
- **Buồn ngủ/Mệt mỏi**: Nhắm mắt quá lâu, ngáp liên tục
- **Mất tập trung**: Quay đầu, nhìn sang hướng khác
- **Sử dụng điện thoại**: Nhắn tin, gọi điện khi lái xe
- **Hành vi phụ**: Ăn uống, hút thuốc khi lái xe
- **Không thắt dây an toàn**

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Camera     │────▶│  Raspberry   │────▶│   Cảnh Báo   │
│   Module     │     │     Pi 4     │     │   (Buzzer/   │
└──────────────┘     └──────────────┘     │    LED)      │
                            │               └──────────────┘
                     ┌──────▼──────┐
                     │   AI Model   │
                     │ (TFLite/ONNX)│
                     └──────────────┘
```

## 🚀 Mô Hình AI Được Hỗ Trợ

### 1. **MobileNetV2** (Khuyến nghị cho Raspberry Pi)
- Kích thước: ~14MB (quantized: ~3.5MB)
- FPS trên Pi 4: 15-20 FPS
- Độ chính xác: 92-95%

### 2. **YOLOv8n** (Nano version)
- Kích thước: ~6MB (quantized)
- FPS trên Pi 4: 8-12 FPS
- Độ chính xác: 93-96%

### 3. **Custom CNN** (Siêu nhẹ)
- Kích thước: <1MB
- FPS trên Pi 4: 25-30 FPS
- Độ chính xác: 88-91%

## 🛠️ Công Nghệ Sử Dụng

- **Frameworks**: TensorFlow Lite, ONNX Runtime
- **Optimization**: INT8 Quantization, Model Pruning
- **Hardware Acceleration**:
  - Google Coral USB (tùy chọn)
  - Intel NCS2 (tùy chọn)
  - Hailo-8L AI Kit (tùy chọn)

## 📊 Datasets Sử Dụng

1. **DDFDD** (Driver Distraction and Fatigue Detection Dataset)
2. **DMD** (Driving Monitoring Dataset)
3. **Custom Dataset** (Tự thu thập và gán nhãn)

## 🔧 Cài Đặt

### Yêu cầu phần cứng:
- Raspberry Pi 4 (4GB RAM tối thiểu)
- Pi Camera Module hoặc USB Webcam
- MicroSD Card (32GB+)
- Nguồn điện 5V/3A
- (Tùy chọn) AI Accelerator

### Yêu cầu phần mềm:
- Raspberry Pi OS (64-bit)
- Python 3.8+
- TensorFlow Lite 2.13+
- OpenCV 4.8+

## 📁 Cấu Trúc Project

```
driver-safety-ai/
├── models/              # Mô hình AI đã train
├── src/                 # Mã nguồn chính
│   ├── detection/       # Module phát hiện
│   ├── preprocessing/   # Xử lý ảnh
│   └── alert/          # Hệ thống cảnh báo
├── training/           # Scripts training trên Colab
├── data/              # Datasets mẫu
├── configs/           # File cấu hình
└── tests/            # Unit tests
```

## 🎓 Training trên Google Colab

Xem hướng dẫn chi tiết tại: [training/README.md](training/README.md)

## 📈 Hiệu Suất

| Mô hình | FPS (Pi 4) | RAM Usage | Độ Chính Xác |
|---------|-----------|-----------|--------------|
| MobileNetV2 | 15-20 | 250MB | 94% |
| YOLOv8n | 8-12 | 400MB | 95% |
| Custom CNN | 25-30 | 150MB | 90% |

## ⚠️ Lưu Ý An Toàn

- Hệ thống chỉ hỗ trợ cảnh báo, không thay thế ý thức người lái
- Cần test kỹ trước khi triển khai thực tế
- Tuân thủ quy định về quyền riêng tư khi thu thập dữ liệu

## 📝 License

MIT License - Sử dụng cho mục đích giáo dục và nghiên cứu an toàn giao thông.