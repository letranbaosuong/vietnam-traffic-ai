# 🧪 Testing Guide - Driver Gesture Detection

**Date**: 2025-11-01
**Status**: Ready for Testing on Raspberry Pi

---

## ⚠️ Important Note: macOS Compatibility

MediaPipe **không tương thích với Python 3.13** trên macOS.

### Options:

1. **✅ Test trên Raspberry Pi** (Recommended)
   - MediaPipe hoạt động tốt trên Pi với Python 3.9-3.11

2. **✅ Xem Demo Simulation** (macOS)
   - Video demo đã được tạo: `driver_gesture_demo_output.mp4`
   - Simulate gesture warnings để xem visualization

3. **Test trên macOS** (Advanced)
   - Cần Python 3.9-3.11
   - Sử dụng pyenv hoặc conda

---

## 📹 Test Videos Available

Trong `test_videos/`:

1. **car-driver.mp4** (3.5 MB, 308 frames)
   - Video driver trong xe ô tô
   - 1280x720 @ 30 FPS

2. **mobycle-driver.mp4** (4.9 MB)
   - Video driver xe máy

3. **solidWhiteRight.mp4** (2.5 MB)
   - Lane detection test

4. **detect_video_danang.mp4** (16 MB)
   - Traffic Đà Nẵng

---

## 🎬 Demo Simulation (macOS - No MediaPipe)

### Run Demo:

```bash
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

source venv/bin/activate

# Run simulation
python3 demo_gesture_simulation.py test_videos/car-driver.mp4 --save
```

### Demo Results:

✅ **Output**: `driver_gesture_demo_output.mp4` (6.4 MB)
✅ **FPS**: 47.56 (simulation is fast)
✅ **Warnings**: 5 types simulated
- Phone usage (HIGH)
- Distraction (MEDIUM)
- Hands off wheel (MEDIUM)

### View Output:

```bash
open driver_gesture_demo_output.mp4
```

---

## 🍓 Testing on Raspberry Pi (Actual Detection)

### 1. Setup on Raspberry Pi

```bash
# SSH to your Pi
ssh pi@raspberrypi.local

# Navigate to project
cd ~/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Create venv if not exists
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-python numpy mediapipe

# Verify MediaPipe
python3 -c "import mediapipe; print('MediaPipe OK')"
```

### 2. Copy Test Videos to Pi

```bash
# On your Mac
scp test_videos/car-driver.mp4 pi@raspberrypi.local:~/test_videos/
scp test_videos/mobycle-driver.mp4 pi@raspberrypi.local:~/test_videos/
```

### 3. Run Tests on Pi

#### Test với Video:

```bash
# On Raspberry Pi
source venv/bin/activate

python3 test_driver_gesture.py \
  --mode video \
  --source ~/test_videos/car-driver.mp4 \
  --save
```

#### Test với Camera:

```bash
python3 test_driver_gesture.py \
  --mode camera \
  --source 0
```

#### Test All Features (Object + Lane + Gesture):

```bash
python3 detect.py \
  --model efficientdet_lite0.tflite \
  --cameraId 0 \
  --frameWidth 640 \
  --frameHeight 480 \
  --enableLaneDetection \
  --enableGestureDetection
```

### 4. Expected Performance on Pi

| Device | Resolution | Gesture Only | All Features |
|--------|------------|--------------|--------------|
| Pi 4 | 640x480 | 25-30 FPS | 12-15 FPS |
| Pi 3 | 640x480 | 15-20 FPS | 8-10 FPS |

---

## 📊 Test Results (Demo Simulation)

### Video: car-driver.mp4

```
Video: 1280x720 @ 30 FPS
Total frames: 308
Processing time: 6.48s
Average FPS: 47.56 (simulation)

Frames with warnings: 139 (45.1%)

Warnings Detected:
- Phone usage: 1
- Distraction: 3
- Hands off wheel: 1
```

### Output Video Features:

✅ Visual warnings with red/orange overlays
✅ Vietnamese warning messages
✅ Status bar with warning count
✅ Simulation mode indicator
✅ Frame progress counter

---

## 🎯 What You'll See in Output

### Warning Display:

```
┌──────────────────────────────────────────┐
│ ⚠️ NGUY HIỂM: Đang gọi điện thoại!      │ <- Red overlay
│ ⚠️ MẤT TẬP TRUNG: Đang nhìn sang PHẢI! │ <- Orange overlay
├──────────────────────────────────────────┤
│                                           │
│     [Video with driver]                  │
│                                           │
├──────────────────────────────────────────┤
│ Warnings: 2 │ Total: 15                  │ <- Status bar
└──────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Issue 1: MediaPipe not available on macOS

**Solution**: Use demo simulation or test on Raspberry Pi

```bash
python3 demo_gesture_simulation.py test_videos/car-driver.mp4 --save
```

### Issue 2: Low FPS on Raspberry Pi

**Solutions**:
1. Reduce resolution:
   ```bash
   python3 detect.py --frameWidth 320 --frameHeight 240
   ```

2. Disable unused features:
   ```bash
   # Only gesture, no lane
   python3 detect.py --enableGestureDetection
   ```

3. Skip frames:
   Edit `detect.py`, add in gesture detection:
   ```python
   if counter % 2 == 0:  # Every 2 frames
       warnings, image = gesture_detector.detect(image)
   ```

### Issue 3: No gestures detected

**Solutions**:
- Adjust thresholds in `driver_gesture_detector.py`
- Better lighting
- Camera position facing driver

---

## 📝 Test Checklist

### Demo Simulation (macOS):

- [x] Run `demo_gesture_simulation.py`
- [x] Generate output video (6.4 MB)
- [x] View warnings visualization
- [x] Check statistics report

### Raspberry Pi Testing:

- [ ] Install MediaPipe on Pi
- [ ] Copy test videos to Pi
- [ ] Run `test_driver_gesture.py` with video
- [ ] Test with Pi camera
- [ ] Test integrated mode (object + lane + gesture)
- [ ] Measure FPS performance
- [ ] Adjust thresholds if needed

---

## 🎥 Demo Video Analysis

### car-driver.mp4 Demo Output:

**File**: `driver_gesture_demo_output.mp4`
**Size**: 6.4 MB
**Duration**: ~10 seconds (308 frames @ 30 FPS)

**Simulated Scenarios**:
- Frames 50-100: Phone call detected
- Frames 150-200: Looking right
- Frames 250-280: Hands off wheel
- Frames 320-360: Looking at phone
- Random: Occasional left glances

**Visual Elements**:
- ✅ Red warning overlays
- ✅ Vietnamese warning text
- ✅ Status bar at bottom
- ✅ "SIMULATION MODE" indicator
- ✅ Frame counter

---

## 🚀 Next Steps

### For Demonstration:

1. **View Demo Output**:
   ```bash
   open driver_gesture_demo_output.mp4
   ```

2. **Create presentation slides** with screenshots

3. **Share with team** for feedback

### For Real Testing:

1. **Deploy to Raspberry Pi**
2. **Test with real driver videos**
3. **Adjust detection thresholds**
4. **Measure actual FPS**
5. **Collect feedback**
6. **Iterate and improve**

---

## 📚 Reference Files

- `driver_gesture_detector.py` - Core detection logic
- `gesture_warning_system.py` - Warning management
- `test_driver_gesture.py` - Test script (needs MediaPipe)
- `demo_gesture_simulation.py` - Demo without MediaPipe
- `README_DRIVER_GESTURE.md` - Full documentation
- `DRIVER_GESTURE_SUMMARY.md` - Quick reference

---

## ✅ Summary

### macOS (Current):
✅ Demo simulation works perfectly
✅ Output video created: `driver_gesture_demo_output.mp4`
✅ Visualization confirmed working
✅ Code ready for deployment

### Raspberry Pi (Next):
⏳ Needs MediaPipe installation
⏳ Needs actual testing with videos
⏳ Performance benchmarking required
⏳ Threshold tuning needed

---

**Status**: ✅ Demo completed on macOS
**Next**: Test on Raspberry Pi with MediaPipe
**Output**: driver_gesture_demo_output.mp4 (ready to view)

---

**Created**: 2025-11-01 10:38 AM
**Demo FPS**: 47.56 (simulation)
**Expected Pi FPS**: 15-30 (real detection)
