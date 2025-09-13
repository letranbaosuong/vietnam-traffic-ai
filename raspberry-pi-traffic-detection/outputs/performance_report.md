# 📊 Traffic Detection Performance Report

**Generated:** 2025-09-13 14:01:13  
**System:** macOS  
**CPU Cores:** 8  
**Memory:** 16.0GB  

## 🎯 Executive Summary

| Video | Resolution | Avg FPS | CPU % | Memory % | Vehicles/Frame |
|-------|------------|---------|-------|----------|----------------|
| Traffic Sample | 1280x720 | 10.2 | 8.4% | 76.7% | 0.0 |
| Car Driver | 1280x720 | 10.7 | 9.2% | 76.9% | 0.0 |
| Motorcycle Driver | 1280x720 | 10.1 | 9.9% | 77.2% | 0.0 |

## 📈 Detailed Analysis

### 🎬 Traffic Sample

**📺 Video Information:**
- Resolution: 1280x720
- Original FPS: 29
- Duration: 43.97s
- Total Frames: 1275

**⚡ Performance Metrics:**
- Average FPS: **10.15 FPS**
- FPS Range: 4.99 - 11.47 FPS
- Frame Time: 0.099ms ± 0.013ms
- CPU Usage: 8.4%
- Memory Usage: 76.7%
- Processing Efficiency: 98.5ms/frame

**🔍 Detection Results:**
- Average Vehicles per Frame: 0.0
- Max Vehicles Detected: 0
- Lane Detection Rate: 99.0%
- Traffic Signs Detected: 8

**🥧 Raspberry Pi 4 Estimates:**
- Estimated FPS: ~6.1 FPS
- Recommended for: Batch processing

### 🎬 Car Driver

**📺 Video Information:**
- Resolution: 1280x720
- Original FPS: 29
- Duration: 10.62s
- Total Frames: 308

**⚡ Performance Metrics:**
- Average FPS: **10.74 FPS**
- FPS Range: 2.98 - 12.44 FPS
- Frame Time: 0.093ms ± 0.018ms
- CPU Usage: 9.2%
- Memory Usage: 76.9%
- Processing Efficiency: 93.1ms/frame

**🔍 Detection Results:**
- Average Vehicles per Frame: 0.0
- Max Vehicles Detected: 0
- Lane Detection Rate: 5.0%
- Traffic Signs Detected: 61

**🥧 Raspberry Pi 4 Estimates:**
- Estimated FPS: ~6.4 FPS
- Recommended for: Batch processing

### 🎬 Motorcycle Driver

**📺 Video Information:**
- Resolution: 1280x720
- Original FPS: 24
- Duration: 15.79s
- Total Frames: 379

**⚡ Performance Metrics:**
- Average FPS: **10.14 FPS**
- FPS Range: 6.25 - 11.57 FPS
- Frame Time: 0.099ms ± 0.009ms
- CPU Usage: 9.9%
- Memory Usage: 77.2%
- Processing Efficiency: 98.7ms/frame

**🔍 Detection Results:**
- Average Vehicles per Frame: 0.0
- Max Vehicles Detected: 0
- Lane Detection Rate: 100.0%
- Traffic Signs Detected: 487

**🥧 Raspberry Pi 4 Estimates:**
- Estimated FPS: ~6.1 FPS
- Recommended for: Batch processing

## 🎯 Recommendations

**Best Performance:** Car Driver (10.74 FPS)

**Optimization Suggestions:**
- For Pi 4 deployment, consider:
  - Reducing input resolution to 640x480
  - Disabling sign detection for higher FPS
  - Using frame skipping (process every 2nd frame)
  - Implementing multi-threading optimizations

## 📝 Technical Notes

- All tests performed with YOLOv8n model
- Integrated detection includes: Vehicle + Lane + Sign detection
- Pi 4 estimates based on CPU performance scaling
- Actual Pi 4 performance may vary based on cooling and configuration
