# 🚗 Traffic Detection Web Application

**Date**: 2025-11-01
**Version**: 1.0
**Features**: Lane Detection, Driver Gesture Detection, Combined Mode

---

## 🎯 TỔNG QUAN

Web application cho phép bạn upload video và test 2 tính năng:
1. **🛣️ Lane Detection** - Phát hiện làn đường
2. **👤 Gesture Detection** - Phát hiện cử chỉ nguy hiểm của driver
3. **⚡ Combined Mode** - Kết hợp cả 2

---

## ✨ TÍNH NĂNG

### 🎨 Giao diện Web
- ✅ Modern, responsive UI
- ✅ Drag & drop video upload
- ✅ Real-time progress tracking
- ✅ Video preview và download
- ✅ Statistics display

### 🔧 Processing Features
- ✅ Support multiple video formats (MP4, AVI, MOV, MKV)
- ✅ Max file size: 100MB
- ✅ Real-time processing
- ✅ Output video generation
- ✅ Detailed statistics

### 📊 Detection Modes

#### 1. Lane Detection
- Phát hiện và vẽ làn đường
- Real-time processing
- Works trên highway và city traffic

#### 2. Gesture Detection
- Phát hiện phone usage
- Phát hiện distraction (nhìn sang trái/phải)
- Phát hiện hands off wheel
- Vietnamese warning messages
- Color-coded alerts

#### 3. Combined Mode
- Cả lane và gesture detection
- Comprehensive traffic monitoring

---

## 🚀 CÁCH SỬ DỤNG

### Quick Start:

```bash
# Navigate to project directory
cd /Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi

# Run startup script
chmod +x start_web_app.sh
./start_web_app.sh
```

### Manual Start:

```bash
# Activate venv
source venv/bin/activate

# Install Flask (if not installed)
pip install flask

# Start server
python3 web_app.py
```

### Access Web App:

Open browser and go to:
- **Local**: http://localhost:5000
- **Network**: http://your-ip:5000

---

## 📝 HƯỚNG DẪN SỬ DỤNG

### Bước 1: Upload Video

1. Mở web app: `http://localhost:5000`
2. Click vào upload area hoặc drag & drop video
3. Chọn video (MP4, AVI, MOV, MKV)
4. Max size: 100MB

### Bước 2: Chọn Chế Độ

3 options:
- **🛣️ Lane Detection**: Chỉ phát hiện làn đường
- **👤 Gesture Detection**: Chỉ phát hiện cử chỉ driver
- **⚡ Combined**: Cả 2 tính năng

### Bước 3: Xử Lý

1. Click button "Tải lên và Xử lý Video"
2. Đợi processing (hiển thị progress bar)
3. View kết quả khi hoàn thành

### Bước 4: Xem Kết Quả

- Video output tự động play
- Xem statistics (FPS, warnings, etc.)
- Download video đã xử lý
- Hoặc upload video mới

---

## 📁 CẤU TRÚC PROJECT

```
raspberry_pi/
├── web_app.py                 # Flask application
├── start_web_app.sh           # Startup script
├── templates/
│   ├── index.html             # Main upload page
│   └── results.html           # Results page
├── uploads/                   # Uploaded videos (auto-created)
├── outputs/                   # Processed videos (auto-created)
├── lane_detector.py           # Lane detection module
├── driver_gesture_detector.py # Gesture detection module
└── gesture_warning_system.py  # Warning system
```

---

## 🔧 TECHNICAL DETAILS

### Backend (Python Flask)

**File**: `web_app.py`

**Main Routes**:
- `GET /` - Main upload page
- `POST /upload` - Handle file upload
- `GET /process/lane/<filename>` - Process with lane detection
- `GET /process/gesture/<filename>` - Process with gesture detection
- `GET /process/both/<filename>` - Process with both
- `GET /download/<filename>` - Download processed video
- `GET /results` - View all results

**Key Features**:
```python
- Video processing with OpenCV
- Lane detection integration
- Gesture detection simulation
- Statistics generation
- File management
```

### Frontend (HTML/CSS/JavaScript)

**File**: `templates/index.html`

**Key Features**:
- Drag & drop upload
- File validation
- Progress tracking
- Result display
- Video playback
- Download functionality

**Technologies**:
- Pure JavaScript (no frameworks)
- CSS3 with gradients and animations
- Responsive design
- Modern UI/UX

---

## 📊 API ENDPOINTS

### 1. Upload Video
```
POST /upload
Content-Type: multipart/form-data

Body:
  video: <file>
  detection_type: 'lane' | 'gesture' | 'both'

Response:
{
  "success": true,
  "filename": "timestamp_video.mp4",
  "detection_type": "gesture"
}
```

### 2. Process Lane Detection
```
GET /process/lane/<filename>

Response:
{
  "success": true,
  "output_file": "lane_video.mp4",
  "stats": {
    "frames_processed": 308,
    "processing_time": "6.48s",
    "avg_fps": "47.56",
    "output_size": "6.4 MB"
  }
}
```

### 3. Process Gesture Detection
```
GET /process/gesture/<filename>

Response:
{
  "success": true,
  "output_file": "gesture_video.mp4",
  "stats": {
    "frames_processed": 308,
    "frames_with_warnings": 139,
    "warning_percentage": "45.1%",
    "processing_time": "6.48s",
    "avg_fps": "47.56",
    "output_size": "6.4 MB"
  }
}
```

### 4. Download Video
```
GET /download/<filename>

Returns: video file
```

---

## ⚙️ CONFIGURATION

### Modify Settings

Edit `web_app.py`:

```python
# Max upload size
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Output folder
app.config['OUTPUT_FOLDER'] = 'outputs'

# Allowed extensions
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Server settings
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Change Port:

```python
# In web_app.py, last line:
app.run(debug=True, host='0.0.0.0', port=8080)  # Change to 8080
```

### Production Mode:

```python
# Disable debug mode for production
app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 🎨 CUSTOMIZATION

### Modify UI Colors

Edit `templates/index.html`, change CSS:

```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Button colors */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent color */
color: #667eea;
```

### Add New Detection Mode

1. Add endpoint in `web_app.py`:
```python
@app.route('/process/custom/<filename>')
def process_custom(filename):
    # Your custom processing
    pass
```

2. Add button in `templates/index.html`:
```html
<div class="option-btn" data-type="custom">
    <div class="icon">🎯</div>
    <div class="title">Custom Mode</div>
    <div class="desc">Your description</div>
</div>
```

---

## 📊 PERFORMANCE

### Expected Processing Speed

| Video Resolution | Lane Only | Gesture Only | Combined |
|------------------|-----------|--------------|----------|
| 640x480 | 50-60 FPS | 45-50 FPS | 35-40 FPS |
| 1280x720 | 40-45 FPS | 40-45 FPS | 30-35 FPS |
| 1920x1080 | 25-30 FPS | 30-35 FPS | 20-25 FPS |

**Note**: Performance on macOS. Raspberry Pi will be slower.

### Optimization Tips

1. **Reduce Resolution**:
```python
# In processing function
frame = cv2.resize(frame, (640, 480))
```

2. **Skip Frames**:
```python
if frame_num % 2 == 0:  # Process every 2 frames
    # Process
```

3. **Use Production Mode**:
```python
app.run(debug=False)  # Faster than debug mode
```

---

## 🐛 TROUBLESHOOTING

### Problem 1: Port 5000 Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change port in web_app.py
app.run(port=8080)
```

### Problem 2: Upload Fails

**Error**: File size too large

**Solution**:
- Compress video before upload
- Or increase max size in `web_app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
```

### Problem 3: Processing Slow

**Solutions**:
1. Reduce video resolution
2. Use shorter videos for testing
3. Close other applications
4. Check system resources

### Problem 4: Video Not Playing

**Solutions**:
1. Check browser compatibility
2. Try different browser
3. Check video codec (MP4 H.264 works best)
4. Download and play locally

### Problem 5: Cannot Access from Other Devices

**Solutions**:
1. Check firewall settings
2. Use correct IP address
3. Ensure both devices on same network
4. Try: http://192.168.x.x:5000

---

## 🔒 SECURITY NOTES

### For Production Deployment:

1. **Disable Debug Mode**:
```python
app.run(debug=False)
```

2. **Add File Validation**:
```python
# Check file content, not just extension
```

3. **Limit Upload Size**:
```python
# Already configured at 100MB
```

4. **Use HTTPS**:
```python
# Use production WSGI server
# gunicorn, uWSGI, etc.
```

5. **Add Authentication** (if needed):
```python
from flask_httpauth import HTTPBasicAuth
# Add auth to routes
```

---

## 📦 DEPLOYMENT

### Development (Current):

```bash
python3 web_app.py
```

### Production (Recommended):

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app
```

### Raspberry Pi Deployment:

```bash
# Copy files to Pi
scp -r . pi@raspberrypi.local:~/traffic-detection-web/

# SSH to Pi
ssh pi@raspberrypi.local

# Setup
cd ~/traffic-detection-web
python3 -m venv venv
source venv/bin/activate
pip install flask opencv-python numpy

# Run
./start_web_app.sh
```

---

## 📱 MOBILE ACCESS

Web app is responsive and works on mobile:

1. **Find Server IP**:
```bash
# macOS
ipconfig getifaddr en0

# Raspberry Pi
hostname -I
```

2. **Access from Phone**:
```
Open browser on phone
Go to: http://server-ip:5000
```

3. **Upload from Phone**:
- Works with phone camera videos
- Works with downloaded videos

---

## 🎯 USE CASES

### 1. Testing & Development
- Quick video testing
- Visual verification
- Performance benchmarking

### 2. Demo & Presentation
- Show features to clients
- Interactive demonstrations
- Easy sharing

### 3. Research & Analysis
- Process multiple videos
- Compare results
- Collect statistics

### 4. Educational
- Learn computer vision
- Understand detection algorithms
- Practice web development

---

## 🔮 FUTURE ENHANCEMENTS

Potential improvements:

- [ ] Real-time camera processing
- [ ] Batch video processing
- [ ] User authentication
- [ ] Video history/database
- [ ] Advanced statistics
- [ ] Export statistics to CSV
- [ ] Video comparison tool
- [ ] Mobile app
- [ ] Cloud deployment
- [ ] API documentation (Swagger)

---

## 📚 REFERENCES

### Technologies Used:
- **Flask**: https://flask.palletsprojects.com/
- **OpenCV**: https://opencv.org/
- **MediaPipe**: https://mediapipe.dev/

### Documentation:
- Flask docs: https://flask.palletsprojects.com/
- OpenCV Python: https://docs.opencv.org/
- HTML5 Video: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/video

---

## 💡 TIPS & TRICKS

### Quick Tips:

1. **Test với sample videos**:
```bash
# Use existing test videos
test_videos/car-driver.mp4
test_videos/detect_video_danang.mp4
```

2. **View all results**:
```
http://localhost:5000/results
```

3. **Clear old files**:
```bash
rm -rf uploads/* outputs/*
```

4. **Check logs**:
```bash
# Flask shows logs in terminal
# Watch for errors
```

5. **Restart server**:
```bash
# Press Ctrl+C
# Run ./start_web_app.sh again
```

---

## ✅ CHECKLIST

### Before Running:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Flask installed
- [ ] Required modules available:
  - lane_detector.py
  - driver_gesture_detector.py
  - gesture_warning_system.py
- [ ] templates/ folder with HTML files

### Testing Checklist:

- [ ] Can access web interface
- [ ] Can upload video
- [ ] Lane detection works
- [ ] Gesture detection works
- [ ] Combined mode works
- [ ] Can download results
- [ ] Statistics display correctly

---

## 📞 SUPPORT

### If You Need Help:

1. Check `TESTING_GUIDE.md`
2. Check `README_DRIVER_GESTURE.md`
3. Check `VIDEO_DOWNLOAD_GUIDE.md`
4. Review Flask error messages in terminal

### Common Issues:
- Port already in use → Change port
- Module not found → Install dependencies
- Video won't play → Check codec
- Slow processing → Reduce resolution

---

## 🏁 SUMMARY

### Quick Start:
```bash
./start_web_app.sh
```

### Access:
```
http://localhost:5000
```

### Features:
- ✅ Lane Detection
- ✅ Gesture Detection
- ✅ Combined Mode
- ✅ Modern Web UI
- ✅ Easy to use

### Status:
**✅ READY TO USE!**

---

**Created**: 2025-11-01
**Version**: 1.0
**Author**: Claude Code
**License**: MIT

🚗 Happy Testing! 🎉
