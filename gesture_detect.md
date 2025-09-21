📚 Các Mô Hình Detect Cử Chỉ Phổ Biến 2024

  1. MediaPipe Hand Gesture

  - Độ chính xác: 89.7% với Inception-v3 + LSTM
  - GitHub tham khảo:
    - https://github.com/kinivi/hand-gesture-recognition-mediapipe
    - https://github.com/KelvinPuyam/Hand-Gesture-Recognition
    - https://github.com/baukk/Gesture-Recognition

  2. YOLOv8 Hand Detection

  - Hiệu suất: mAP 0.887 với YOLOv8s
  - FPS: 30-60 FPS trên GPU
  - Tài liệu:
    - https://docs.ultralytics.com/datasets/pose/hand-keypoints/
    - https://github.com/cansik/yolo-hand-detection
    - Kaggle: YOLOv8 Train on Custom Dataset

  3. Edge AI Solutions

  - NanoEdgeAI: Accuracy 84.27% với MLP
  - Hardware: Nucleo-F401RE, Jetson Nano
  - Ứng dụng: Smart wheelchair, IoT devices

  🔗 Link Tham Khảo Chi Tiết

  Research Papers:

  1. PMC Review 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC8321080/
  2. IEEE Systematic Review:
  https://ieeexplore.ieee.org/iel8/6287639/10380310/10580881.pdf
  3. Springer Complex Systems:
  https://link.springer.com/article/10.1007/s40747-023-01173-6

  Tutorials & Implementations:

  1. Google MediaPipe Guide:
  https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer
  2. PyImageSearch YOLOv8: https://pyimagesearch.com/2023/05/15/hand-gesture-recognition-w
  ith-yolov8-on-oak-d-in-near-real-time/
  3. Papers with Code: https://paperswithcode.com/task/hand-gesture-recognition

  Datasets:

  - Roboflow: Hand Gesture Recognition datasets
  - Kaggle: Multiple hand gesture datasets
  - Hugging Face: Pre-trained models

  💡 Ứng Dụng Cho Driver Monitor

  Bạn có thể tích hợp detect cử chỉ vào hệ thống monitor tài xế:

  1. Cử chỉ điều khiển: OK, Stop, Turn signals
  2. Phát hiện sử dụng điện thoại: Hand-to-ear gesture
  3. Cảnh báo mất tập trung: Hand off steering wheel

  🚀 Khuyến Nghị

  Với Raspberry Pi của bạn:
  - Dùng MediaPipe cho accuracy cao
  - YOLOv8n cho real-time performance
  - Kết hợp với TensorFlow Lite cho edge deployment