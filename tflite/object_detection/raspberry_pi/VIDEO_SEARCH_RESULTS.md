# 🔍 Video Search Results - Driver Frontal View

**Date**: 2025-11-01
**Request**: Tìm video người lái xe chính diện để test

---

## 📊 Kết Quả Tìm Kiếm

### ❌ Không Tìm Thấy Video Frontal View

Đã kiểm tra **20 video files** trong project, không có video nào quay góc chính diện driver.

### 📹 Videos Hiện Có:

| Video | Size | Duration | Góc Quay | Frontal? |
|-------|------|----------|----------|----------|
| car-driver.mp4 | 3.5 MB | 10.3s | Từ ghế sau | ❌ |
| mobycle-driver.mp4 | 4.9 MB | 15.8s | POV driver | ❌ |
| sample.mp4 | 14.2 MB | 42.5s | Traffic monitoring | ❌ |
| solidWhiteRight.mp4 | 2.5 MB | - | Lane detection | ❌ |
| detect_video_danang.mp4 | 16 MB | - | Traffic Đà Nẵng | ❌ |

### 🖼️ Preview Images Created:

✅ Đã tạo 9 preview images để check góc quay:
- preview_car-driver_frame0.jpg
- preview_car-driver_frame154.jpg
- preview_car-driver_frame307.jpg
- preview_mobycle-driver_frame0.jpg
- preview_mobycle-driver_frame189.jpg
- preview_mobycle-driver_frame378.jpg
- preview_sample_frame0.jpg
- preview_sample_frame637.jpg
- preview_sample_frame1274.jpg

---

## ✅ Giải Pháp

### 🚀 Option 1: Record Webcam (FASTEST - 5 phút)

**Tool đã tạo**: `record_test_video.py`

**Cách dùng**:
```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

source venv/bin/activate

# Record 30 seconds
python3 record_test_video.py

# Record custom duration
python3 record_test_video.py 60
```

**Output**: `test_videos/frontal_driver_webcam.mp4`

**Ưu điểm**:
- ⚡ Nhanh nhất (5 phút)
- ✅ Góc chính diện hoàn hảo
- 💰 Miễn phí
- 🎯 Control được gestures

---

### 📱 Option 2: Quay bằng Điện Thoại

**Setup**:
1. Mount điện thoại trên dashboard
2. Camera hướng về mặt driver
3. Khoảng cách: 50-80cm

**Recording**:
- Duration: 10-30 seconds
- Simulate gestures:
  - Look straight
  - Look left/right
  - Phone near ear
  - Hands off wheel
  - Look down

**Copy to project**:
```bash
cp /path/to/phone/video.mp4 test_videos/frontal_driver_phone.mp4
```

---

### 🌐 Option 3: Download từ YouTube

**Install tool**:
```bash
pip install yt-dlp
```

**Search keywords**:
- "driver monitoring system demo"
- "DMS camera test"
- "driver drowsiness detection sample"
- "driver distraction detection video"

**Download**:
```bash
yt-dlp -f 'best[height<=720]' -o 'test_videos/youtube_driver.mp4' <URL>
```

---

### 🎓 Option 4: Public Datasets (Advanced)

**Recommended Datasets**:

1. **YawDD** (Yawn Detection Dataset)
   - 107 participants
   - Frontal camera view
   - Varied conditions
   - Free but need to request

2. **NTHU-DDD** (Drowsy Driver Detection)
   - 9.5 hours video
   - 18 participants
   - University dataset

3. **UTA-RLDD** (Real-Life Drowsiness)
   - Real conditions
   - Frontal view

**Note**: Cần request access từ universities.

---

## ✅ Demo Đã Hoàn Thành

### 📹 Output Video Available:

**File**: `driver_gesture_demo_output.mp4`
**Size**: 6.4 MB
**Duration**: ~10 seconds

**Statistics**:
```
Processed: 308 frames
Time: 6.48s
FPS: 47.56 (simulation)
Warnings: 5 total
  - Phone usage: 1 (HIGH)
  - Distraction: 3 (MEDIUM)
  - Hands off: 1 (MEDIUM)
```

**View**:
```bash
open driver_gesture_demo_output.mp4
```

---

## 📋 Recommendation

### ⭐ Best Approach:

1. **Ngay bây giờ** (0 phút):
   ```bash
   # Xem demo có sẵn
   open driver_gesture_demo_output.mp4
   ```

2. **Test nhanh** (5 phút):
   ```bash
   # Record webcam video
   python3 record_test_video.py 30

   # Test simulation
   python3 demo_gesture_simulation.py test_videos/frontal_driver_webcam.mp4 --save
   ```

3. **Professional** (30 phút):
   - Quay video bằng điện thoại trong xe
   - Simulate realistic driving scenarios
   - Copy vào project và test

4. **Dataset** (nhiều giờ):
   - Request access to public datasets
   - Download và extract videos
   - Use for comprehensive testing

---

## 🎯 Next Steps

### Immediate (Now):
1. ✅ Xem demo output: `open driver_gesture_demo_output.mp4`
2. ✅ Review preview images để hiểu góc quay

### Short-term (5-10 phút):
1. 🎥 Record webcam: `python3 record_test_video.py`
2. 🧪 Test với video mới: `python3 demo_gesture_simulation.py ...`

### Medium-term (30 phút - 1 giờ):
1. 📱 Quay video bằng phone trong xe
2. 🌐 Download sample từ YouTube
3. 🧪 Test với multiple videos

### Long-term (nhiều giờ):
1. 📚 Request dataset access
2. 🔬 Professional testing
3. 📊 Collect statistics

---

## 📁 Files Created

### Tools:
- ✅ `check_driver_videos.py` - Video checker with preview
- ✅ `record_test_video.py` - Webcam recorder
- ✅ `test_with_current_video.sh` - Quick test script

### Documentation:
- ✅ `FRONTAL_VIEW_GUIDE.md` - Complete guide
- ✅ `VIDEO_SEARCH_RESULTS.md` - This file

### Preview Images:
- ✅ 9 preview images generated

---

## 💡 Key Findings

1. **No frontal view videos in project**
   - All current videos are different angles
   - Need to create or download new videos

2. **Webcam is fastest solution**
   - 5 minutes to create test video
   - Perfect frontal angle
   - Free and easy

3. **Demo already working**
   - Output video created successfully
   - Visualization confirmed
   - System ready for testing

4. **Multiple options available**
   - Webcam (fastest)
   - Phone (professional)
   - YouTube (samples)
   - Datasets (comprehensive)

---

## ✨ Summary

| Requirement | Status | Solution |
|-------------|--------|----------|
| Find frontal video | ❌ Not found | Record new video |
| Test system | ✅ Working | Demo completed |
| Tools ready | ✅ Created | record_test_video.py |
| Documentation | ✅ Complete | FRONTAL_VIEW_GUIDE.md |

**Recommended Action**:
```bash
python3 record_test_video.py 30
```

This will create a perfect frontal-view driver video in 30 seconds, ready for testing!

---

**Created**: 2025-11-01 10:47 AM
**Status**: ✅ Complete
**Tools**: Ready to use
**Next**: Record webcam video or view current demo
