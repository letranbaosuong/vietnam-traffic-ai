# 📥 Hướng Dẫn Download Video Driver Frontal View

**Date**: 2025-11-01
**Purpose**: Các cách lấy video driver góc chính diện để test gesture detection

---

## 🎯 TÓM TẮT - KHUYẾN NGHỊ

| Phương pháp | Thời gian | Chất lượng | Độ khó | Frontal View |
|-------------|-----------|------------|--------|--------------|
| ⚡ **Webcam** | 5 phút | ⭐⭐⭐⭐ | Dễ | ✅ Perfect |
| 📱 Phone | 10 phút | ⭐⭐⭐⭐⭐ | Dễ | ✅ Perfect |
| 🌐 Pexels/Pixabay | 15 phút | ⭐⭐⭐⭐ | Dễ | ⚠️ Varies |
| 📦 Dataset | 30-60 phút | ⭐⭐⭐⭐⭐ | Trung bình | ✅ Perfect |

**⚡ KHUYẾN NGHỊ: Dùng webcam!** Nhanh nhất, dễ nhất, góc chính diện hoàn hảo!

---

## 🚀 OPTION 1: WEBCAM (NHANH NHẤT - 5 PHÚT)

### ✅ Cách dùng:

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

source venv/bin/activate

# Record 30 seconds
python3 record_test_video.py

# Hoặc custom duration
python3 record_test_video.py 60
```

### 📹 Tool sẽ:
1. ✅ Mở webcam của bạn
2. ✅ Hiển thị countdown và instructions
3. ✅ Hướng dẫn simulate gestures:
   - Nhìn thẳng (normal)
   - Nhìn trái/phải (distraction)
   - Phone near ear (phone call)
   - Tay xuống (hands off wheel)
   - Nhìn xuống (looking at phone)
4. ✅ Tự động save: `test_videos/frontal_driver_webcam.mp4`

### 🎬 Sau khi record:

```bash
# Xem video vừa quay
open test_videos/frontal_driver_webcam.mp4

# Test với simulation
python3 demo_gesture_simulation.py test_videos/frontal_driver_webcam.mp4 --save

# Hoặc test thật với MediaPipe (trên Pi)
python3 test_driver_gesture.py --mode video --source test_videos/frontal_driver_webcam.mp4
```

**⚡ Total time: 5 phút**

---

## 📱 OPTION 2: QUAY BẰNG ĐIỆN THOẠI

### Setup:
1. Mount điện thoại trên dashboard hoặc holder
2. Camera hướng vào mặt driver
3. Khoảng cách: 50-80cm
4. Ensure good lighting

### Recording checklist:
- [ ] Camera stable, frontal view
- [ ] Face clearly visible
- [ ] Good lighting
- [ ] Record 20-30 seconds
- [ ] Simulate gestures:
  - [ ] Normal driving (5s)
  - [ ] Look left/right (5s)
  - [ ] Phone call gesture (5s)
  - [ ] Hands off wheel (5s)
  - [ ] Look down at phone (5s)

### Transfer to Mac:

**Via AirDrop**:
1. Select video on iPhone
2. Click Share → AirDrop → Your Mac
3. Save to Downloads

**Move to project**:
```bash
mv ~/Downloads/driver_video.mp4 test_videos/frontal_driver_phone.mp4
```

**⏱️ Total time: 10-15 phút**

---

## 🌐 OPTION 3: FREE STOCK VIDEOS (PEXELS/PIXABAY)

### 🔍 Pexels Videos (Free):

**URL**: https://www.pexels.com/search/videos/car%20interior/

**Search keywords**:
- "car interior driver"
- "driving person"
- "driver face"
- "woman driving car"
- "man driving car"

**Download steps**:
1. Browse videos on Pexels
2. Find video với driver's face visible
3. Click video → "Free Download"
4. Select quality (720p hoặc 1080p)
5. Save to `test_videos/pexels_driver.mp4`

### 📸 Pixabay Videos (Free):

**URL**: https://pixabay.com/videos/search/driving/

**Instructions**:
- Search: "driving", "driver", "car interior"
- Filter: Videos only
- Download: Click "Free Download"
- Move to: `test_videos/`

**Note**: Không phải tất cả video đều có frontal view. Phải xem preview trước.

**⏱️ Total time: 15-20 phút**

---

## 📦 OPTION 4: PUBLIC DATASETS (PROFESSIONAL)

### 1. ✅ D3S Dataset (Google Drive)

**Description**:
- Videos of drivers: eye close, yawning, neutral states
- Frontal camera view
- 3 subjects
- High quality

**Download**:
1. **Video Dataset**: https://drive.google.com/file/d/1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view
2. Click "Download" button
3. Extract zip file
4. Copy sample videos to `test_videos/`

**Citation**: Gupta et al., 2018

---

### 2. ✅ YawDD Dataset (Direct Download)

**Description**:
- 107 participants (57 male, 50 female)
- Dashboard camera, frontal view
- Expressions: normal, talking, yawning
- With/without glasses, sunglasses
- Real conditions, varied illumination

**Download**:
```bash
cd downloads

# Download dataset (~500MB)
curl -L -o YawDD.rar "http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar"

# Extract (requires unrar)
brew install unrar  # macOS
unrar x YawDD.rar

# Copy sample to test folder
cp YawDD/Normal/*.mp4 ../test_videos/yawdd_sample.mp4
```

**Direct link**: http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar

---

### 3. ✅ VBDDD Dataset (Baidu Pan - Chinese)

**Description**:
- 558 video samples (3s-50s each)
- 640x480 resolution, 30 FPS
- Frontal view
- Drowsiness detection

**Download**:
1. URL: https://pan.baidu.com/s/1qxRKT_ydBDVpCE5-OSgP2Q?pwd=4kna
2. Extraction code: `4kna`
3. Requires Baidu account (Chinese service)

**Note**: Phức tạp hơn, cần account Baidu.

---

### 4. 📚 Other Research Datasets:

**NTHU-DDD** (Drowsy Driver Detection):
- 9.5 hours of video
- 18 participants
- Drowsy and non-drowsy states
- Need to request from university

**UTA-RLDD** (Real-Life Drowsiness):
- Real driving conditions
- Frontal camera
- Need to request access

**⏱️ Total time: 30 phút - 2 giờ**

---

## 🎬 QUICK START SCRIPT

Tôi đã tạo script tự động:

```bash
# Make executable
chmod +x download_sample_videos.sh

# Run downloader
./download_sample_videos.sh
```

Script sẽ:
1. Show tất cả options
2. Hướng dẫn download
3. Option để download YawDD directly
4. Instructions cho các datasets khác

---

## 📊 SO SÁNH CHI TIẾT

### Speed:
1. 🥇 Webcam: 5 phút
2. 🥈 Phone: 10 phút
3. 🥉 Pexels: 15 phút
4. Dataset: 30+ phút

### Quality:
1. 🥇 Phone: Highest quality, realistic
2. 🥈 Datasets: Professional, varied
3. 🥉 Webcam: Good, sufficient
4. Pexels: Varies

### Frontal View Guarantee:
1. 🥇 Webcam: 100% frontal
2. 🥇 Phone: 100% (if setup correctly)
3. 🥇 Datasets: 100% frontal
4. Pexels: ~50% (need to search)

### Ease of Use:
1. 🥇 Webcam: Easiest
2. 🥈 Phone: Easy
3. 🥉 Pexels: Medium (need to browse)
4. Datasets: Complex (download, extract)

---

## ✅ KHUYẾN NGHỊ CHO TỪNG MỤC ĐÍCH

### 🧪 Quick Testing (5 phút):
```bash
python3 record_test_video.py 30
```
→ Use webcam

### 📹 Demo/Presentation (15 phút):
→ Quay bằng phone trong xe thật
→ Professional look

### 🔬 Research/Development (30+ phút):
→ Download YawDD hoặc D3S dataset
→ Multiple scenarios, varied subjects

### 🎓 Learning/Training (2+ giờ):
→ Download all datasets
→ Comprehensive testing

---

## 🎯 STEP-BY-STEP: WEBCAM METHOD (RECOMMENDED)

### 1. Preparation (1 phút):
```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi
source venv/bin/activate
```

### 2. Record (30 giây):
```bash
python3 record_test_video.py 30
```

### 3. Follow instructions:
- Sit 50-80cm from camera
- Face camera directly
- Follow on-screen gesture prompts

### 4. Review (10 giây):
```bash
open test_videos/frontal_driver_webcam.mp4
```

### 5. Test (1 phút):
```bash
# Simulation test
python3 demo_gesture_simulation.py test_videos/frontal_driver_webcam.mp4 --save

# View output
open driver_gesture_demo_output.mp4
```

**⚡ Total: 5 phút from start to tested output!**

---

## 📁 DOWNLOAD LINKS SUMMARY

### Direct Downloads:
| Dataset | Link | Size | Note |
|---------|------|------|------|
| YawDD | http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar | ~500MB | Direct .rar |
| D3S | https://drive.google.com/file/d/1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view | ~? | Google Drive |
| VBDDD | https://pan.baidu.com/s/1qxRKT_ydBDVpCE5-OSgP2Q?pwd=4kna | ~? | Baidu (Chinese) |

### Browse & Download:
| Source | Link | Quality | Frontal? |
|--------|------|---------|----------|
| Pexels | https://www.pexels.com/search/videos/car%20interior/ | High | Varies |
| Pixabay | https://pixabay.com/videos/search/driving/ | High | Varies |

---

## 🚨 TROUBLESHOOTING

### Problem: Webcam not working
**Solution**:
- Check camera permissions in System Preferences
- Close other apps using camera
- Try: `python3 -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Problem: Download too slow
**Solution**:
- Use webcam instead
- Try Pexels (faster than datasets)
- Download during off-peak hours

### Problem: No frontal view in downloaded video
**Solution**:
- Preview video before full download
- Use webcam for guaranteed frontal view
- Try YawDD dataset (guaranteed frontal)

### Problem: Video format not compatible
**Solution**:
```bash
# Convert with ffmpeg
brew install ffmpeg
ffmpeg -i input.avi -c:v libx264 output.mp4
```

---

## 📝 SUMMARY

### ⚡ FASTEST (5 phút):
```bash
python3 record_test_video.py 30
```

### 📱 BEST QUALITY (15 phút):
Quay bằng phone trong xe

### 🎓 MOST COMPREHENSIVE (30+ phút):
Download YawDD dataset

### 💡 RECOMMENDATION:
**Start với webcam ngay bây giờ!** Sau đó có thể quay video professional bằng phone nếu cần.

---

**Created**: 2025-11-01 10:55 AM
**Tools Available**:
- ✅ record_test_video.py
- ✅ download_sample_videos.sh
- ✅ check_driver_videos.py

**Next Step**:
```bash
python3 record_test_video.py 30
```

🎬 Let's create your first frontal view driver video in 5 minutes!
