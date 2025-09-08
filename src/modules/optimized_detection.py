#!/usr/bin/env python3
"""
Optimized Object Detection for Raspberry Pi Real-time Applications
Sử dụng YOLO11n với NCNN optimization cho hiệu suất tốt nhất trên RPi
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple, Optional
import yaml
import threading
from queue import Queue

class OptimizedDetector:
    """
    Detector tối ưu cho Raspberry Pi với YOLO11n + NCNN
    """
    
    def __init__(self, config_path: str = "configs/config.yaml", use_ncnn: bool = True):
        """
        Khởi tạo detector tối ưu
        
        Args:
            config_path: Đường dẫn file config
            use_ncnn: Sử dụng NCNN format (khuyên dùng cho RPi)
        """
        print("🚀 Khởi tạo Optimized Detector cho Raspberry Pi...")
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Cấu hình tối ưu cho RPi
        self.input_size = (416, 416)  # Giảm từ 640x640 để tăng tốc
        self.confidence = 0.4  # Tăng threshold để giảm false positives
        self.iou_threshold = 0.5
        self.use_ncnn = use_ncnn
        
        # Load model
        self._load_model()
        
        # Classes quan trọng cho giao thông VN
        self.traffic_classes = {
            0: 'person',        # Người đi bộ
            1: 'bicycle',       # Xe đạp  
            2: 'car',          # Ô tô
            3: 'motorcycle',    # Xe máy - quan trọng nhất ở VN
            5: 'bus',          # Xe buýt
            7: 'truck',        # Xe tải
            9: 'traffic_light' # Đèn giao thông
        }
        
        # Màu sắc dễ nhìn
        self.colors = {
            'person': (0, 255, 0),        # Xanh lá
            'bicycle': (255, 255, 0),     # Vàng
            'car': (255, 0, 0),          # Đỏ
            'motorcycle': (255, 0, 255),  # Tím - nổi bật cho xe máy
            'bus': (0, 255, 255),        # Cyan
            'truck': (128, 0, 128),      # Tím đậm
            'traffic_light': (0, 165, 255) # Cam
        }
        
        # Threading cho xử lý frame
        self.frame_queue = Queue(maxsize=2)  # Giới hạn queue size
        self.result_queue = Queue(maxsize=2)
        self.processing = False
        
        print("✅ Khởi tạo hoàn thành!")
    
    def _load_model(self):
        """Load model với optimization cho RPi"""
        try:
            model_path = self.config['model']['yolo_model']
            
            # Thử load model YOLO11n trước
            if "yolov8" in model_path:
                print("⚠️ Đang dùng YOLOv8, khuyên dùng YOLO11n cho RPi")
                model_path = "yolo11n.pt"  # Tự động chuyển sang YOLO11n
            
            self.model = YOLO(model_path)
            
            # Export sang NCNN nếu chưa có
            if self.use_ncnn:
                ncnn_path = model_path.replace('.pt', '_ncnn_model')
                try:
                    # Thử load NCNN model
                    self.model = YOLO(ncnn_path)
                    print("✅ Đã load NCNN model (tối ưu cho RPi)")
                except:
                    # Export sang NCNN nếu chưa có
                    print("🔄 Đang export model sang NCNN format...")
                    temp_model = YOLO(model_path)
                    temp_model.export(format="ncnn", imgsz=self.input_size[0])
                    self.model = YOLO(ncnn_path)
                    print("✅ Export NCNN hoàn thành!")
            else:
                print("📝 Sử dụng PyTorch model (chậm hơn trên RPi)")
                
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            # Fallback về YOLOv8n
            self.model = YOLO("yolov8n.pt")
    
    def detect_optimized(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects với optimization cho real-time
        
        Args:
            frame: Input frame
            
        Returns:
            List detections với bbox, confidence, class
        """
        # Resize frame để tăng tốc
        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, self.input_size)
        
        # Inference
        results = self.model(
            resized_frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            device='cpu'  # Chỉ định dùng CPU cho RPi
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Scale bounding box về kích thước gốc
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1 = int(x1 * width / self.input_size[0])
                    y1 = int(y1 * height / self.input_size[1])
                    x2 = int(x2 * width / self.input_size[0])
                    y2 = int(y2 * height / self.input_size[1])
                    
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Chỉ lấy objects quan trọng cho giao thông
                    if class_id in self.traffic_classes:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.traffic_classes[class_id]
                        })
        
        return detections
    
    def draw_detections_simple(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Vẽ detections đơn giản, tối ưu cho RPi
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Vẽ bounding box mỏng hơn để tiết kiệm computation
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
            
            # Label đơn giản
            label = f"{class_name}: {confidence:.1f}"  # Chỉ 1 chữ số thập phân
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def get_traffic_stats_fast(self, detections: List[Dict]) -> Dict:
        """Thống kê nhanh cho real-time display"""
        stats = {
            'total': len(detections),
            'vehicles': sum(1 for d in detections if d['class_name'] in ['car', 'motorcycle', 'bus', 'truck']),
            'people': sum(1 for d in detections if d['class_name'] == 'person'),
            'motorcycles': sum(1 for d in detections if d['class_name'] == 'motorcycle'),
        }
        return stats
    
    def process_frame_threaded(self, frame: np.ndarray):
        """Process frame trong thread riêng để không block main loop"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())
    
    def _processing_thread(self):
        """Thread xử lý frame"""
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                detections = self.detect_optimized(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put(detections)
            time.sleep(0.01)  # Ngắt ngủ ngắn
    
    def start_threading(self):
        """Bắt đầu threading mode"""
        self.processing = True
        self.thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.thread.start()
        print("🔄 Threading mode started")
    
    def stop_threading(self):
        """Dừng threading mode"""
        self.processing = False
        if hasattr(self, 'thread'):
            self.thread.join()
        print("⏹️ Threading mode stopped")
    
    def get_latest_results(self) -> Optional[List[Dict]]:
        """Lấy kết quả detection mới nhất"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

# Utility functions
def optimize_camera_settings(cap: cv2.VideoCapture) -> cv2.VideoCapture:
    """
    Tối ưu settings camera cho RPi
    """
    # Giảm resolution để tăng FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Buffer size = 1 để giảm latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("📹 Camera settings optimized for RPi")
    return cap

def measure_fps(func):
    """Decorator đo FPS"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        print(f"⚡ Current FPS: {fps:.1f}")
        return result
    return wrapper

# Demo real-time detection
if __name__ == "__main__":
    print("🚀 Demo Optimized Traffic Detection cho Raspberry Pi")
    
    # Khởi tạo detector
    detector = OptimizedDetector(use_ncnn=True)
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    cap = optimize_camera_settings(cap)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera")
        exit()
    
    print("📹 Bắt đầu real-time detection...")
    print("Nhấn 'q' để thoát, 's' để bật/tắt stats")
    
    show_stats = True
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            frame_start = time.time()
            detections = detector.detect_optimized(frame)
            detection_time = time.time() - frame_start
            
            # Vẽ detections
            annotated_frame = detector.draw_detections_simple(frame, detections)
            
            # Hiển thị thống kê
            if show_stats:
                stats = detector.get_traffic_stats_fast(detections)
                
                # FPS và thông tin
                cv2.putText(annotated_frame, f"FPS: {1.0/detection_time:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Objects: {stats['total']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Vehicles: {stats['vehicles']} | People: {stats['people']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Motorcycles: {stats['motorcycles']}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Optimized Traffic Detection - RPi', annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Stats display: {'ON' if show_stats else 'OFF'}")
            
            fps_counter += 1
            
            # Hiển thị FPS trung bình mỗi 30 frames
            if fps_counter % 30 == 0:
                avg_fps = fps_counter / (time.time() - start_time)
                print(f"📊 Average FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Dừng bởi người dùng")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup hoàn thành")