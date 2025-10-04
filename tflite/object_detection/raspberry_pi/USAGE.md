# Object Detection với TensorFlow Lite - Hướng dẫn sử dụng

## ✅ Đã hoàn thành

Project đã được sửa lỗi và chạy thành công với:
- Python 3.12
- TensorFlow 2.16.2
- OpenCV
- SSD MobileNet v1 model

## 📁 Files quan trọng

- `detect.tflite` - SSD MobileNet v1 model (4MB, quantized UINT8)
- `labelmap.txt` - COCO dataset labels (91 classes)
- `test_ssd.py` - Script demo test với ảnh
- `coco_labels.txt` - COCO labels đầy đủ

## 🚀 Cách sử dụng

### 1. Test với ảnh mẫu

```bash
python test_ssd.py
```

Kết quả được lưu vào `ssd_detection_result.jpg`

### 2. Chạy với camera (nếu có)

```bash
# Sẽ cần viết thêm script cho camera
# Model đã sẵn sàng
```

## 📊 Kết quả test

Với test image `test_data/table.jpg`, model phát hiện được:
- dining table: 48.8%
- knife: 50.0%
- cup: 43.8%
- book: 40.2%

## 🔧 Các lỗi đã sửa

1. **opencv-python version incompatible** - Đã remove version constraint
2. **tflite-support không khả dụng** - Chuyển sang dùng TensorFlow trực tiếp
3. **Model file corrupt** - Download lại model từ TensorFlow Hub
4. **EfficientDet model format phức tạp** - Chuyển sang SSD MobileNet đơn giản hơn

## 🎯 Models

### SSD MobileNet v1 (Recommended)
- File: `detect.tflite`
- Size: 4MB
- Input: 300x300 UINT8
- Output: boxes, classes, scores, num_detections
- Fast và accurate

### EfficientDet Lite0 (Advanced)
- File: `efficientdet_lite0.tflite`
- Size: 13MB
- Input: 320x320 FLOAT32
- Cần xử lý output phức tạp hơn

## 📦 Dependencies

```txt
argparse
numpy>=1.20.0
opencv-python
tensorflow
protobuf
```

## 🔗 Resources

- [TensorFlow Lite Object Detection](https://www.tensorflow.org/lite/examples/object_detection/overview)
- [COCO Dataset](https://cocodataset.org/)
- [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
