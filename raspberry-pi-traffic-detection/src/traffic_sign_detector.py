#!/usr/bin/env python3
"""
Traffic Sign Detection cho Raspberry Pi 4
Sử dụng YOLO và custom classification cho biển báo Việt Nam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class TrafficSignDetector:
    def __init__(self, confidence=0.5):
        """
        Initialize traffic sign detector
        """
        self.confidence = confidence
        self.sign_model = self._load_sign_model()

        # Vietnamese traffic signs mapping
        self.vietnamese_signs = {
            # Biển cấm
            'stop_sign': 'Biển STOP',
            'no_entry': 'Cấm đi ngược chiều',
            'no_left_turn': 'Cấm rẽ trái',
            'no_right_turn': 'Cấm rẽ phải',
            'no_u_turn': 'Cấm quay đầu',
            'speed_limit_30': 'Tốc độ tối đa 30',
            'speed_limit_50': 'Tốc độ tối đa 50',
            'speed_limit_60': 'Tốc độ tối đa 60',
            'speed_limit_80': 'Tốc độ tối đa 80',

            # Biển báo hiệu
            'yield': 'Nhường đường',
            'priority_road': 'Đường ưu tiên',
            'crosswalk': 'Đường dành cho người đi bộ',
            'children_crossing': 'Khu vực trường học',
            'bicycle_crossing': 'Đường dành cho xe đạp',
            'road_work': 'Đang thi công',
            'traffic_light': 'Đèn giao thông',

            # Biển chỉ dẫn
            'turn_left': 'Rẽ trái',
            'turn_right': 'Rẽ phải',
            'go_straight': 'Đi thẳng',
            'roundabout': 'Bùng binh',
            'one_way': 'Đường một chiều',
        }

        # Color ranges for different sign types (HSV)
        self.sign_colors = {
            'red': [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],  # Red signs
            'blue': [(100, 100, 100), (130, 255, 255)],  # Blue signs
            'yellow': [(20, 100, 100), (30, 255, 255)],  # Yellow signs
            'white': [(0, 0, 200), (180, 30, 255)]  # White signs
        }

    def _load_sign_model(self):
        """
        Load traffic sign detection model
        """
        # Try to load custom Vietnamese traffic sign model first
        custom_model_path = "../models/vietnamese_traffic_signs.pt"

        if os.path.exists(custom_model_path):
            return YOLO(custom_model_path)
        else:
            # Fallback to general object detection for traffic lights and stop signs
            print("⚠️ Custom traffic sign model not found. Using general YOLO model.")
            print("📥 Download Vietnamese traffic sign model for better accuracy.")
            return YOLO('yolov8n.pt')

    def detect_by_color(self, image):
        """
        Detect potential traffic signs using color filtering
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        potential_signs = []

        for color_name, color_ranges in self.sign_colors.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            # Handle multiple ranges for red color
            if color_name == 'red':
                mask1 = cv2.inRange(hsv, np.array(color_ranges[0]), np.array(color_ranges[1]))
                mask2 = cv2.inRange(hsv, np.array(color_ranges[2]), np.array(color_ranges[3]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, np.array(color_ranges[0]), np.array(color_ranges[1]))

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Filter by aspect ratio (traffic signs are usually square or circular)
                    if 0.7 <= aspect_ratio <= 1.3:
                        potential_signs.append({
                            'bbox': (x, y, w, h),
                            'color': color_name,
                            'area': area,
                            'confidence': 0.7  # Color-based confidence
                        })

        return potential_signs

    def detect_shapes(self, image, bbox):
        """
        Detect traffic sign shapes within bounding box
        """
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return 'unknown'

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 'unknown'

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        vertices = len(approx)

        if vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices >= 8:
            return 'circle'
        else:
            return 'polygon'

    def classify_sign(self, image, bbox, color, shape):
        """
        Classify traffic sign based on color, shape and content
        """
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]

        # Basic classification based on color and shape
        if color == 'red':
            if shape == 'circle':
                return 'speed_limit_60', 0.8  # Default speed limit
            elif shape == 'triangle':
                return 'yield', 0.8
            elif shape == 'rectangle':
                return 'stop_sign', 0.9
        elif color == 'blue':
            if shape == 'circle':
                return 'go_straight', 0.7
            elif shape == 'rectangle':
                return 'one_way', 0.7
        elif color == 'yellow':
            if shape == 'triangle':
                return 'road_work', 0.8
        elif color == 'white':
            if shape == 'rectangle':
                return 'priority_road', 0.7

        return 'unknown_sign', 0.5

    def detect_with_yolo(self, image):
        """
        Detect traffic signs using YOLO model
        """
        results = self.sign_model(image, conf=self.confidence)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    # Check if detection is traffic light (class 9 in COCO)
                    if cls == 9:  # Traffic light
                        detections.append({
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'sign_type': 'traffic_light',
                            'confidence': conf,
                            'method': 'yolo'
                        })

        return detections

    def detect_signs(self, image):
        """
        Main traffic sign detection function combining multiple methods
        """
        all_detections = []

        # Method 1: YOLO detection
        yolo_detections = self.detect_with_yolo(image)
        all_detections.extend(yolo_detections)

        # Method 2: Color-based detection
        color_detections = self.detect_by_color(image)

        for detection in color_detections:
            bbox = detection['bbox']
            color = detection['color']

            # Detect shape
            shape = self.detect_shapes(image, bbox)

            # Classify sign
            sign_type, confidence = self.classify_sign(image, bbox, color, shape)

            all_detections.append({
                'bbox': bbox,
                'sign_type': sign_type,
                'confidence': confidence * detection['confidence'],
                'color': color,
                'shape': shape,
                'method': 'color_shape'
            })

        # Remove overlapping detections
        filtered_detections = self._filter_overlapping(all_detections)

        return filtered_detections

    def _filter_overlapping(self, detections, overlap_threshold=0.5):
        """
        Remove overlapping detections using NMS-like approach
        """
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        for detection in detections:
            bbox1 = detection['bbox']
            overlaps = False

            for existing in filtered:
                bbox2 = existing['bbox']
                if self._calculate_overlap(bbox1, bbox2) > overlap_threshold:
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(detection)

        return filtered

    def _calculate_overlap(self, bbox1, bbox2):
        """
        Calculate IoU overlap between two bounding boxes
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0

    def draw_detections(self, image, detections):
        """
        Draw traffic sign detections on image
        """
        result = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            sign_type = detection['sign_type']
            confidence = detection['confidence']

            x, y, w, h = bbox

            # Choose color based on sign type
            if 'stop' in sign_type or 'speed_limit' in sign_type:
                color = (0, 0, 255)  # Red
            elif 'traffic_light' in sign_type:
                color = (0, 255, 255)  # Yellow
            elif 'yield' in sign_type or 'road_work' in sign_type:
                color = (0, 165, 255)  # Orange
            else:
                color = (255, 0, 0)  # Blue

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Get Vietnamese name
            vn_name = self.vietnamese_signs.get(sign_type, sign_type)
            label = f"{vn_name}: {confidence:.2f}"

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)

            # Draw label text
            cv2.putText(result, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result

def main():
    """Test traffic sign detection"""
    detector = TrafficSignDetector(confidence=0.5)

    # Test with video
    cap = cv2.VideoCapture("video/sample.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Detect traffic signs
        detections = detector.detect_signs(frame)

        # Draw results
        result = detector.draw_detections(frame, detections)

        # Show result
        cv2.imshow('Traffic Sign Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()