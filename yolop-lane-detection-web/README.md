# 🚗 YOLOP Lane Detection Web Application

Ứng dụng web phát hiện làn đường, phương tiện và cảnh báo an toàn giao thông dựa trên YOLOP.

## 🎯 Tính Năng

### 1. **Phát Hiện Làn Đường** 🛣️
- Xác định chính xác vị trí các làn đường
- Vẽ đường cong làn đường
- Polynomial fitting cho độ chính xác cao

### 2. **Xác Định Vùng Lái Xe** ✅
- Tô màu xanh vùng có thể lái xe an toàn (drivable area)
- Segmentation mask overlay
- Real-time processing

### 3. **Phát Hiện Phương Tiện** 🚙
- Nhận diện ô tô, xe tải, xe buýt, xe máy
- Bounding box với confidence score
- Multi-class detection

### 4. **Cảnh Báo Lệch Làn** ⚠️
- Cảnh báo tiếng Việt khi xe lệch làn
- 3 mức độ: nhẹ, cảnh báo, nguy hiểm
- Theo dõi lịch sử lệch làn

## 📋 Yêu Cầu Hệ Thống

```bash
Python 3.7+
PyTorch 1.7+
OpenCV 4.5+
Flask 2.3+
```

## 🚀 Cài Đặt

### 1. Clone repository
```bash
cd yolop-lane-detection-web
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download YOLOP weights
```bash
# Download pretrained YOLOP model
wget https://github.com/hustvl/YOLOP/releases/download/1.0/yolop-640-640.pth
mv yolop-640-640.pth models/
```

## 🏃 Chạy Ứng Dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

## 📹 Sử Dụng

1. **Upload Video**: Kéo thả hoặc chọn file video (MP4, AVI, MOV, MKV)
2. **Xử Lý**: Hệ thống tự động phân tích video
3. **Xem Kết Quả**:
   - Video đã xử lý với annotations
   - Thống kê chi tiết
   - Cảnh báo an toàn

## 🔧 API Endpoints

### Upload Video
```
POST /upload
Content-Type: multipart/form-data
Body: video file
```

### Check Status
```
GET /status/{job_id}
Response: JSON with processing status
```

### Download Result
```
GET /download/{job_id}
Response: Processed video file
```

## 📊 Cấu Trúc Project

```
yolop-lane-detection-web/
├── app.py                 # Flask application
├── requirements.txt       # Dependencies
├── modules/
│   ├── yolop_detector.py     # YOLOP detection module
│   ├── lane_departure_warning.py  # Lane departure system
│   └── video_processor.py    # Video processing utilities
├── templates/
│   └── index.html         # Web interface
├── uploads/              # Uploaded videos
├── outputs/              # Processed videos
└── models/               # Model weights (optional)
```

## ⚙️ Cấu Hình

### Sử dụng YOLOP thực (Production)

1. Download YOLOP weights
2. Update `yolop_detector.py`:
```python
# Load actual YOLOP model
self.model = torch.load('models/yolop-640-640.pth')
self.model.to(self.device)
self.model.eval()
```

### Tùy chỉnh cảnh báo

Edit `lane_departure_warning.py`:
```python
self.departure_threshold = 0.15  # Ngưỡng lệch làn
self.critical_threshold = 0.25   # Ngưỡng nguy hiểm
```

## 🎨 Demo Mode

Hiện tại app đang chạy ở chế độ demo với:
- OpenCV Canny + Hough cho lane detection
- Haar Cascade cho vehicle detection
- Color segmentation cho drivable area

Để sử dụng YOLOP thực, cần download và load model weights.

## 🚨 Các Mức Cảnh Báo

1. **Nhẹ**: "Chú ý: Xe đang lệch sang trái/phải"
2. **Cảnh báo**: "⚠️ CẢNH BÁO: Xe lệch sang TRÁI/PHẢI quá nhiều!"
3. **Nguy hiểm**: "⚠️ NGUY HIỂM: Xe đang lấn làn TRÁI/PHẢI!"
4. **Rất nguy hiểm**: "🚨 RẤT NGUY HIỂM: Xe liên tục lấn làn!"

## 📈 Performance

- **Input**: Video MP4/AVI (max 100MB)
- **Processing**: ~5-10 FPS on CPU
- **Output**: Annotated video với:
  - Làn đường (vàng)
  - Vùng lái xe (xanh)
  - Phương tiện (box xanh dương)
  - Cảnh báo (đỏ)

## 🔗 Tham Khảo

- YOLOP Paper: https://arxiv.org/abs/2108.11250
- Original GitHub: https://github.com/hustvl/YOLOP
- BDD100K Dataset: https://bdd-data.berkeley.edu/

## 📄 License

MIT License

## 🤝 Đóng Góp

Pull requests are welcome! Đặc biệt:
- Integration với YOLOP weights thực
- Thêm TensorRT acceleration
- WebSocket cho real-time streaming
- Mobile responsive improvements

---
**Phát triển bởi**: Vietnam Traffic AI Team
**Version**: 1.0.0