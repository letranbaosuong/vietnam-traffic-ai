# 🎬 Cách Xem Kết Quả Lane Detection

## 📹 Video Output Đã Tạo

Bạn có **2 video** đã được xử lý với lane detection:

1. **solidWhiteRight_output.mp4** (3.5 MB)
   - Video test chuẩn với làn trắng rõ ràng
   - Detection rate: 94% cả 2 làn
   
2. **danang_output.mp4** (9.0 MB)
   - Video giao thông Đà Nẵng thực tế
   - Detection rate: 100% ít nhất 1 làn

## 🖥️ Cách Mở Video

### Trên macOS:

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Mở với QuickTime (default)
open solidWhiteRight_output.mp4
open danang_output.mp4

# Hoặc mở với VLC
vlc solidWhiteRight_output.mp4 &
vlc danang_output.mp4 &
```

### Trong Finder:

1. Mở Finder
2. Navigate tới: 
   ```
   /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi
   ```
3. Double-click vào file `.mp4`

## 🎨 Những Gì Bạn Sẽ Thấy

Trong video output:

✅ **Làn đường** được vẽ bằng **đường màu xanh lá** (green)
✅ **Vùng làn đường** được fill màu xanh semi-transparent (alpha 0.2)
✅ **Frame counter** ở góc trên bên trái
✅ **Progress**: Frame X/Total

## 📊 Kết Quả Chi Tiết

Xem file: `TEST_RESULTS.md` để biết chi tiết về:
- Performance metrics (FPS, processing time)
- Detection accuracy
- So sánh 2 videos
- Recommendations

## 🚀 Chạy Lại Với Video Khác

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Activate virtual environment
source venv/bin/activate

# Chạy với video của bạn
python3 test_lane_detection.py \
  --mode video \
  --source /path/to/your/video.mp4 \
  --save
```

## 📸 Test Với Camera

```bash
source venv/bin/activate

python3 test_lane_detection.py \
  --mode camera \
  --source 0
```

Nhấn 'q' hoặc ESC để thoát.

## 🎯 Kết Luận

- ✅ Lane detection hoạt động tốt
- ✅ Real-time (37-43 FPS trên macOS)
- ✅ 100% frames có ít nhất 1 làn được detect
- ✅ Sẵn sàng test trên Raspberry Pi

Enjoy! 🚗🛣️
