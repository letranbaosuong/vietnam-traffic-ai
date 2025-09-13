#!/usr/bin/env python3
"""
Raspberry Pi 4 Traffic Object Detection
Tối ưu hóa cho Pi 4 với YOLOv8n và MobileNet SSD
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
import psutil
import os
from pathlib import Path

class TrafficDetector:
    def __init__(self, model_type="yolov8n", confidence=0.5):
        """
        Initialize traffic detector
        Args:
            model_type: "yolov8n", "yolov5n", hoặc "mobilenet"
            confidence: Ngưỡng confidence
        """
        self.model_type = model_type
        self.confidence = confidence
        self.model = self._load_model()

        # Traffic classes (COCO dataset) - Thêm person để dễ test
        self.traffic_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic light'
        }

        # Performance tracking
        self.fps_counter = 0
        self.start_time = time.time()

    def _load_model(self):
        """Load model tối ưu cho Pi 4"""
        if self.model_type == "yolov8n":
            model = YOLO('yolov8n.pt')
            # Convert to TensorRT if available
            try:
                model.export(format='engine', device=0, half=True)
                return YOLO('yolov8n.engine')
            except:
                return model

        elif self.model_type == "yolov5n":
            model = YOLO('yolov5n.pt')
            return model

        else:
            # MobileNet SSD fallback
            return cv2.dnn.readNetFromTensorflow(
                'frozen_inference_graph.pb',
                'ssd_mobilenet_v2_coco.pbtxt'
            )

    def detect_frame(self, frame):
        """
        Detect objects trong 1 frame
        Returns: frame với bounding boxes và FPS info
        """
        start_time = time.time()

        if self.model_type in ["yolov8n", "yolov5n"]:
            results = self.model(frame, conf=self.confidence, classes=list(self.traffic_classes.keys()))
            annotated_frame = self._draw_yolo_detections(frame, results)
        else:
            annotated_frame = self._detect_mobilenet(frame)

        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0

        # Add performance info
        self._add_performance_info(annotated_frame, fps)

        return annotated_frame

    def _draw_yolo_detections(self, frame, results):
        """Vẽ detections từ YOLO"""
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls in self.traffic_classes:
                        label = f"{self.traffic_classes[cls]}: {conf:.2f}"

                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated_frame

    def _add_performance_info(self, frame, fps):
        """Thêm thông tin performance lên frame"""
        # System info
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        temp = self._get_pi_temperature()

        # Add text overlay
        info_text = [
            f"FPS: {fps:.1f}",
            f"CPU: {cpu_percent:.1f}%",
            f"RAM: {memory_percent:.1f}%",
            f"Temp: {temp:.1f}°C",
            f"Model: {self.model_type}"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

    def _get_pi_temperature(self):
        """Lấy nhiệt độ Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0

    def process_video(self, video_path, output_path=None):
        """
        Process video file
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer nếu cần save
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            result_frame = self.detect_frame(frame)

            # Save frame nếu cần
            if writer:
                writer.write(result_frame)

            # Display frame (comment out for headless Pi)
            cv2.imshow('Traffic Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                print(f"Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"Processing complete: {frame_count} frames in {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

def main():
    """Demo function"""
    # Create detector
    detector = TrafficDetector(model_type="yolov8n", confidence=0.5)

    # Test với webcam hoặc video file
    # video_source = 0  # Webcam
    video_source = "video/sample.mp4"  # Video file

    if isinstance(video_source, str):
        detector.process_video(video_source, "outputs/detected_traffic.mp4")
    else:
        # Real-time detection
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result_frame = detector.detect_frame(frame)
            cv2.imshow('Traffic Detection', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()