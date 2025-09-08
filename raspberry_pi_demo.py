#!/usr/bin/env python3
"""
Demo đơn giản cho Raspberry Pi Real-time Traffic Detection
Sử dụng YOLO11n + NCNN optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import time
from modules.optimized_detection import OptimizedDetector, optimize_camera_settings

def main():
    print("🚀 VIETNAM TRAFFIC AI - RASPBERRY PI DEMO")
    print("=" * 50)
    print("🎯 Tối ưu cho real-time detection trên Raspberry Pi")
    print("⚡ Sử dụng YOLO11n + NCNN cho hiệu suất tốt nhất")
    print("🏍️ Tập trung phát hiện xe máy và giao thông Việt Nam")
    print("=" * 50)
    
    # Kiểm tra file config
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"⚠️ Không tìm thấy {config_path}")
        print("Sử dụng cấu hình mặc định...")
        config_path = None
    
    try:
        # Khởi tạo detector tối ưu
        print("🔧 Đang khởi tạo Optimized Detector...")
        detector = OptimizedDetector(
            config_path=config_path if config_path else "configs/config.yaml",
            use_ncnn=True  # Bật NCNN cho RPi
        )
        
        # Mở camera
        print("📹 Đang mở camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Không thể mở camera!")
            print("💡 Kiểm tra:")
            print("   - Camera đã kết nối chưa?")
            print("   - Permissions camera OK?") 
            print("   - Thử: sudo usermod -a -G video $USER")
            return
        
        # Tối ưu camera settings cho RPi
        cap = optimize_camera_settings(cap)
        
        # Lấy thông tin camera
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Camera ready: {width}x{height} @ {fps} FPS")
        print("\n🎮 Điều khiển:")
        print("   - 'q': Thoát")
        print("   - 's': Bật/tắt statistics") 
        print("   - 'f': Hiển thị FPS")
        print("   - 'c': Lưu ảnh hiện tại")
        print("   - SPACE: Pause/Resume")
        print("\n🚀 Bắt đầu detection...")
        
        # Biến điều khiển
        show_stats = True
        show_fps = True
        paused = False
        frame_count = 0
        total_inference_time = 0
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Không đọc được frame từ camera")
                    break
                
                # Đo thời gian inference
                inference_start = time.time()
                detections = detector.detect_optimized(frame)
                inference_time = time.time() - inference_start
                
                # Cập nhật statistics
                frame_count += 1
                total_inference_time += inference_time
                avg_inference_time = total_inference_time / frame_count
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Vẽ detections
                annotated_frame = detector.draw_detections_simple(frame, detections)
                
                # Thêm thông tin overlay
                if show_stats:
                    stats = detector.get_traffic_stats_fast(detections)
                    
                    # Background cho text
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                    
                    # Hiển thị stats
                    y = 35
                    cv2.putText(annotated_frame, "VIETNAM TRAFFIC AI - RPi", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    y += 30
                    if show_fps:
                        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Avg: {1.0/avg_inference_time:.1f}", 
                                   (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y += 25
                    
                    cv2.putText(annotated_frame, f"Total Objects: {stats['total']}", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 25
                    
                    cv2.putText(annotated_frame, f"Vehicles: {stats['vehicles']} | People: {stats['people']}", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y += 25
                    
                    cv2.putText(annotated_frame, f"Motorcycles: {stats['motorcycles']} (VN Focus)", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                    y += 25
                    
                    # Runtime info
                    runtime = time.time() - start_time
                    cv2.putText(annotated_frame, f"Runtime: {runtime:.0f}s | Frames: {frame_count}", 
                               (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                
                # Hiển thị status nếu paused
                if paused:
                    cv2.putText(annotated_frame, "PAUSED - Press SPACE to resume", 
                               (10, annotated_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            else:
                # Khi pause, vẫn hiển thị frame cuối
                cv2.putText(annotated_frame, "PAUSED - Press SPACE to resume", 
                           (10, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Hiển thị frame
            cv2.imshow('Vietnam Traffic AI - Raspberry Pi', annotated_frame)
            
            # Xử lý input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Thoát theo yêu cầu người dùng")
                break
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"📊 Statistics: {'ON' if show_stats else 'OFF'}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"⚡ FPS display: {'ON' if show_fps else 'OFF'}")
            elif key == ord('c'):
                # Lưu frame hiện tại
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📷 Đã lưu: {filename}")
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"⏯️ {'Paused' if paused else 'Resumed'}")
            
            # Hiển thị FPS trong console mỗi 60 frames
            if frame_count % 60 == 0 and frame_count > 0:
                elapsed_time = time.time() - start_time
                overall_fps = frame_count / elapsed_time
                print(f"📈 Frame {frame_count}: Overall FPS = {overall_fps:.1f}, "
                      f"Current FPS = {current_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n⌨️ Dừng bằng Ctrl+C")
    
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Hiển thị thống kê cuối
        if 'frame_count' in locals() and frame_count > 0:
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"\n📊 SESSION SUMMARY:")
            print(f"   - Total frames processed: {frame_count}")
            print(f"   - Total time: {total_time:.1f}s") 
            print(f"   - Average FPS: {avg_fps:.1f}")
            print(f"   - Average inference time: {avg_inference_time*1000:.1f}ms")
        
        print("✅ Cleanup hoàn thành!")

def check_system():
    """Kiểm tra system requirements"""
    print("🔍 Kiểm tra system...")
    
    # Kiểm tra Python version
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Kiểm tra các thư viện cần thiết
    try:
        import cv2
        print(f"   OpenCV: {cv2.__version__}")
    except ImportError:
        print("   ❌ OpenCV chưa cài đặt")
        return False
    
    try:
        from ultralytics import YOLO
        print("   ✅ Ultralytics YOLO available")
    except ImportError:
        print("   ❌ Ultralytics chưa cài đặt")
        print("   💡 Run: pip install ultralytics")
        return False
    
    # Kiểm tra camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✅ Camera accessible")
        cap.release()
    else:
        print("   ⚠️ Camera không thể mở")
    
    return True

if __name__ == "__main__":
    print("🔧 VIETNAM TRAFFIC AI - RASPBERRY PI SETUP")
    print("=" * 50)
    
    if not check_system():
        print("❌ System check failed!")
        print("\n💡 Setup instructions:")
        print("1. pip install -r requirements.txt")
        print("2. sudo usermod -a -G video $USER")
        print("3. Reboot để apply permissions")
        exit(1)
    
    print("✅ System check passed!")
    print()
    
    main()