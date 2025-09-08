import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import yaml

class ObjectDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = YOLO(self.config['model']['yolo_model'])
        self.confidence = self.config['model']['confidence']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Vietnam traffic specific classes
        self.vietnam_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',  # Xe máy - rất quan trọng cho VN
            5: 'bus',
            7: 'truck',
            9: 'traffic_light'
        }
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, 
                           conf=self.confidence,
                           iou=self.iou_threshold,
                           verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id in self.vietnam_classes:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.vietnam_classes[class_id]
                        })
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Màu sắc cho từng loại đối tượng
            colors = {
                'person': (0, 255, 0),      # Xanh lá
                'bicycle': (255, 255, 0),   # Vàng
                'car': (255, 0, 0),         # Đỏ
                'motorcycle': (255, 0, 255), # Tím (xe máy)
                'bus': (0, 255, 255),       # Cyan
                'truck': (128, 0, 128),     # Tím đậm
                'traffic_light': (0, 165, 255) # Cam
            }
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Vẽ bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ label với confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 0), 2)
        
        return annotated_frame
    
    def get_traffic_statistics(self, detections: List[Dict]) -> Dict:
        stats = {
            'total_objects': len(detections),
            'vehicles': 0,
            'people': 0,
            'motorcycles': 0,
            'cars': 0,
            'by_class': {}
        }
        
        for detection in detections:
            class_name = detection['class_name']
            stats['by_class'][class_name] = stats['by_class'].get(class_name, 0) + 1
            
            if class_name in ['car', 'motorcycle', 'bus', 'truck']:
                stats['vehicles'] += 1
            if class_name == 'person':
                stats['people'] += 1
            if class_name == 'motorcycle':
                stats['motorcycles'] += 1
            if class_name == 'car':
                stats['cars'] += 1
        
        return stats