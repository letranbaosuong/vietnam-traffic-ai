# 🚗 Các Mô Hình Lane Detection (Phát Hiện Làn Đường) 2024

## 📊 Tổng Quan

Lane detection là công nghệ quan trọng trong xe tự hành, giúp phát hiện và theo dõi làn đường trong thời gian thực.

## 🎯 Top Mô Hình Lane Detection 2024

### 1. **LD-CAM (Lane Detection with Convolutional Attention Mechanism)**
- **Năm**: 2024
- **Tác giả**: Maddiralla & Subramanian
- **Ưu điểm**:
  - Xử lý tốt điều kiện khó (thời tiết xấu, đường cong, làn hỏng)
  - Sử dụng Enhanced CBAM attention module
  - Encoder-Decoder architecture
- **Performance**: State-of-the-art trên challenging conditions

### 2. **LaneNet**
- **Loại**: Segmentation-based
- **Đặc điểm**:
  - Instance segmentation cho từng làn đường
  - Pixel embedding để phân biệt làn
  - Popular nhất trong segmentation approach
- **GitHub**: https://github.com/MaybeShewill-CV/lanenet-lane-detection

### 3. **PolyLaneNet**
- **Loại**: Parameter-based
- **Đặc điểm**:
  - Third-order polynomial regression
  - Lightweight và nhanh
  - Phù hợp cho embedded devices
- **FPS**: 115 FPS on GPU

### 4. **LSTR (Lane Shape Prediction with Transformers)**
- **Loại**: Transformer-based
- **Đặc điểm**:
  - End-to-end approach
  - Polynomial coefficient regression
  - Global context understanding
- **Accuracy**: State-of-the-art on TuSimple

### 5. **YOLOP (You Only Look Once for Panoptic Driving)**
- **Multi-task**:
  - Object detection
  - Drivable area segmentation
  - Lane detection
- **Real-time**: Chạy được trên Jetson TX2
- **GitHub**: https://github.com/hustvl/YOLOP

## 🔧 Triển Khai Với YOLO

### YOLOv8 + Lane Detection (2024)
```python
# Cài đặt
pip install ultralytics opencv-python numpy

# Lane Detection với YOLOv8
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # hoặc custom trained model

# Lane detection pipeline
1. Preprocessing: ROI selection
2. Edge detection: Canny
3. Line detection: Hough Transform
4. Post-processing: Polynomial fitting
```

**GitHub References**:
- https://github.com/maheshpaulj/Lane_Detection
- https://github.com/jkd2021/YOLOv5-with-Lane-Detection
- https://github.com/CityIsBetter/Lane_Detection

## 🍓 Triển Khai Trên Raspberry Pi

### Lightweight Lane Detection cho Pi

#### Hardware Requirements:
- Raspberry Pi 4 (4GB RAM recommended)
- Pi Camera Module hoặc USB webcam
- Power supply 2A minimum

#### Performance:
- **FPS**: 17-20 FPS trên Pi 3
- **FPS**: 25-30 FPS trên Pi 4

#### Optimized Algorithm:
```python
import cv2
import numpy as np

class LaneDetectorLite:
    def __init__(self):
        # Simplified parameters cho Pi
        self.roi_height = 0.6  # Chỉ xử lý 60% phía dưới frame

    def detect_lanes(self, frame):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # 4. ROI mask
        height, width = frame.shape[:2]
        roi_vertices = np.array([
            [(0, height),
             (width/2, height * self.roi_height),
             (width, height)]
        ], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)

        # 5. Hough lines
        lines = cv2.HoughLinesP(masked, 1, np.pi/180,
                                threshold=50,
                                minLineLength=100,
                                maxLineGap=50)

        return lines
```

### Optimization Tips cho Pi:
1. **Giảm resolution**: 320x240 thay vì 640x480
2. **Skip frames**: Xử lý mỗi 2-3 frames
3. **ROI nhỏ**: Chỉ xử lý vùng cần thiết
4. **Avoid complex algorithms**: Không dùng deep learning nặng
5. **Use HLS color space**: Tốt hơn BGR trong điều kiện sáng khác nhau

## 📚 Tài Liệu Tham Khảo

### Research Papers 2024:
1. **PMC Lane Detection Review**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11333608/
2. **Frontiers Computer Vision**: https://www.frontiersin.org/journals/mechanical-engineering
3. **ArXiv Survey**: https://arxiv.org/abs/2411.16316

### Tutorials:
1. **Hackster.io Pi Tutorial**: https://www.hackster.io/Abhinav_Abhi/road-lane-detection-with-raspberry-pi-a4711f
2. **CircuitDigest Guide**: https://circuitdigest.com/microcontroller-projects/autonomous-lane-detection-car-using-raspberry-pi-and-opecv-part1
3. **Medium YOLOv8 Guide**: https://medium.com/@mahesh.paul.j/building-a-lane-and-car-detection-system-using-yolov8-and-opencv-b562f68bf15e

### GitHub Repositories:
- https://github.com/bavmar/Lane-detection-OpenCV
- https://github.com/saloni-garg/lane-detection-for-smart-vehicle
- https://github.com/topics/lane-detection

## 🚀 Xu Hướng 2024-2025

1. **3D Lane Detection**: Chuyển từ 2D sang 3D
2. **Multi-modal fusion**: Kết hợp camera + LiDAR
3. **Transformer-based models**: Attention mechanism
4. **Edge deployment**: Tối ưu cho embedded devices
5. **HD Map construction**: Real-time map building

## 💡 Recommendations

### Cho Production:
- **Accuracy cao**: LD-CAM hoặc LSTR
- **Real-time**: YOLOP hoặc YOLOv8 integration
- **Embedded**: PolyLaneNet hoặc custom lightweight

### Cho Raspberry Pi:
- OpenCV classical methods (Canny + Hough)
- Simplified polynomial fitting
- Optimized ROI processing

## 🛠️ Quick Start Code

```bash
# Clone example repository
git clone https://github.com/maheshpaulj/Lane_Detection.git
cd Lane_Detection

# Install dependencies
pip install -r requirements.txt

# Run detection
python lane_detection.py --source video.mp4
```

---
*Cập nhật: December 2024*


  Tóm tắt các mô hình chính:

  🎯 Top Models 2024:

  1. LD-CAM - Attention-based, xử lý tốt điều kiện khó
  2. LaneNet - Instance segmentation phổ biến nhất
  3. YOLOP - Multi-task (object + lane + drivable area)
  4. PolyLaneNet - Lightweight, 115 FPS
  5. LSTR - Transformer-based, SOTA accuracy

  🍓 Cho Raspberry Pi:

  - OpenCV classical (Canny + Hough): 17-20 FPS
  - Skip frames optimization
  - ROI processing
  - HLS color space

  🔗 GitHub triển khai:

  - YOLOv8: https://github.com/maheshpaulj/Lane_Detection
  - YOLOP: https://github.com/hustvl/YOLOP
  - LaneNet:
  https://github.com/MaybeShewill-CV/lanenet-lane-detection

  File đã bao gồm code mẫu và hướng dẫn chi tiết cho cả
  production và embedded devices!