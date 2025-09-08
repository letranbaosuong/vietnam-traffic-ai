import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple
import yaml
import re

class TrafficSignDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize EasyOCR for Vietnamese and English
        self.reader = easyocr.Reader(['vi', 'en'], gpu=False)
        
        # Vietnam traffic sign keywords
        self.vietnam_signs = {
            'stop': ['stop', 'dừng', 'dung'],
            'yield': ['nhường đường', 'nhuong duong', 'yield'],
            'speed_limit': ['tốc độ', 'toc do', 'km/h', 'speed'],
            'no_entry': ['cấm', 'cam', 'no entry', 'no'],
            'one_way': ['một chiều', 'mot chieu', 'one way'],
            'parking': ['đỗ xe', 'do xe', 'parking', 'p'],
            'pedestrian_crossing': ['người đi bộ', 'nguoi di bo', 'pedestrian'],
            'school_zone': ['trường học', 'truong hoc', 'school'],
            'construction': ['thi công', 'thi cong', 'construction', 'work']
        }
        
        # Color ranges for traffic sign detection (HSV)
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (160, 50, 50), (180, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)]
        }
    
    def detect_sign_regions(self, frame: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sign_regions = []
        
        # Detect colored regions that might be signs
        for color, ranges in self.color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            if color == 'red':  # Red has two ranges
                mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges[0], ranges[1])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Filter by aspect ratio (signs are usually square-ish or rectangular)
                    if 0.3 < aspect_ratio < 3.0:
                        sign_regions.append({
                            'bbox': [x, y, x + w, y + h],
                            'color': color,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        # Sort by area (larger regions first)
        sign_regions.sort(key=lambda x: x['area'], reverse=True)
        return sign_regions[:10]  # Keep top 10 candidates
    
    def recognize_text(self, image_region: np.ndarray) -> List[str]:
        try:
            results = self.reader.readtext(image_region, detail=0)
            return [text.lower().strip() for text in results if len(text.strip()) > 1]
        except:
            return []
    
    def classify_sign(self, text_list: List[str]) -> Dict:
        best_match = {'type': 'unknown', 'confidence': 0.0, 'text': ''}
        
        all_text = ' '.join(text_list).lower()
        
        for sign_type, keywords in self.vietnam_signs.items():
            confidence = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in all_text:
                    # Exact match gets higher score
                    if keyword == all_text.strip():
                        confidence += 1.0
                    else:
                        confidence += 0.5
                    matched_keywords.append(keyword)
            
            # Special handling for speed limit signs
            if sign_type == 'speed_limit':
                speed_pattern = r'\d+\s*km/h|\d+\s*kmh|\d+\s*km'
                if re.search(speed_pattern, all_text):
                    confidence += 1.0
                    matched_keywords.append('speed detected')
            
            if confidence > best_match['confidence']:
                best_match = {
                    'type': sign_type,
                    'confidence': confidence / len(keywords),  # Normalize
                    'text': all_text,
                    'keywords': matched_keywords
                }
        
        return best_match
    
    def detect_traffic_signs(self, frame: np.ndarray) -> List[Dict]:
        sign_regions = self.detect_sign_regions(frame)
        traffic_signs = []
        
        for region in sign_regions:
            x1, y1, x2, y2 = region['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Enhance the region for better OCR
            roi_enhanced = self.enhance_for_ocr(roi)
            
            # Recognize text
            text_list = self.recognize_text(roi_enhanced)
            
            if text_list:
                # Classify the sign
                classification = self.classify_sign(text_list)
                
                if classification['confidence'] > 0.3:  # Confidence threshold
                    traffic_signs.append({
                        'bbox': region['bbox'],
                        'type': classification['type'],
                        'confidence': classification['confidence'],
                        'text': classification['text'],
                        'color': region['color'],
                        'keywords': classification.get('keywords', [])
                    })
        
        return traffic_signs
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        # Resize for better OCR if too small
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            scale_factor = max(100 / h, 100 / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast
        enhanced = cv2.equalizeHist(denoised)
        
        return enhanced
    
    def draw_signs(self, frame: np.ndarray, signs: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        colors = {
            'stop': (0, 0, 255),           # Red
            'yield': (0, 255, 255),        # Yellow
            'speed_limit': (255, 255, 0),  # Cyan
            'no_entry': (255, 0, 0),       # Blue
            'one_way': (255, 0, 255),      # Magenta
            'parking': (0, 255, 0),        # Green
            'pedestrian_crossing': (255, 165, 0),  # Orange
            'school_zone': (255, 192, 203),        # Pink
            'construction': (128, 0, 128),         # Purple
            'unknown': (128, 128, 128)             # Gray
        }
        
        for sign in signs:
            x1, y1, x2, y2 = sign['bbox']
            sign_type = sign['type']
            confidence = sign['confidence']
            text = sign['text']
            
            color = colors.get(sign_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{sign_type}: {confidence:.2f}"
            if text:
                label += f" ({text[:20]})"
            
            # Background for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Text
            cv2.putText(annotated_frame, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 0, 0), 2)
        
        return annotated_frame
    
    def get_sign_statistics(self, signs: List[Dict]) -> Dict:
        stats = {
            'total_signs': len(signs),
            'by_type': {},
            'by_color': {},
            'high_confidence': 0
        }
        
        for sign in signs:
            sign_type = sign['type']
            color = sign['color']
            confidence = sign['confidence']
            
            stats['by_type'][sign_type] = stats['by_type'].get(sign_type, 0) + 1
            stats['by_color'][color] = stats['by_color'].get(color, 0) + 1
            
            if confidence > 0.7:
                stats['high_confidence'] += 1
        
        return stats