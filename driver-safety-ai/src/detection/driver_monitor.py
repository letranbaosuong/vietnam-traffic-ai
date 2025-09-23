"""
Driver Behavior Detection Module
Phát hiện hành vi lái xe nguy hiểm sử dụng AI
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# TensorFlow Lite for inference
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class BehaviorType(Enum):
    """Các loại hành vi được phát hiện"""
    NORMAL = "normal"
    DROWSY = "drowsy"
    YAWNING = "yawning"
    PHONE_CALL = "phone_call"
    TEXTING = "texting"
    DISTRACTED = "distracted"
    DRINKING = "drinking"
    REACHING_BEHIND = "reaching_behind"
    NO_SEATBELT = "no_seatbelt"


@dataclass
class Detection:
    """Kết quả phát hiện"""
    behavior: BehaviorType
    confidence: float
    timestamp: float
    frame_id: int
    bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def is_dangerous(self) -> bool:
        """Kiểm tra xem hành vi có nguy hiểm không"""
        safe_behaviors = [BehaviorType.NORMAL]
        return self.behavior not in safe_behaviors


class DriverMonitor:
    """
    Lớp chính để giám sát hành vi người lái xe
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.7,
        danger_duration_threshold: float = 3.0
    ):
        """
        Khởi tạo Driver Monitor

        Args:
            model_path: Đường dẫn đến file model TFLite
            input_size: Kích thước input của model
            confidence_threshold: Ngưỡng confidence để chấp nhận detection
            danger_duration_threshold: Thời gian (giây) hành vi nguy hiểm kéo dài để cảnh báo
        """
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.danger_duration_threshold = danger_duration_threshold

        # Load model
        self.interpreter = self._load_model()

        # Tracking
        self.current_behavior = BehaviorType.NORMAL
        self.behavior_start_time = time.time()
        self.frame_count = 0
        self.detections_history: List[Detection] = []

        # Threading for async processing
        self.detection_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.is_running = False

    def _load_model(self) -> tflite.Interpreter:
        """Load TFLite model"""
        interpreter = tflite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        return interpreter

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý frame cho model

        Args:
            frame: Frame từ camera (BGR)

        Returns:
            Preprocessed frame
        """
        # Resize
        resized = cv2.resize(frame, self.input_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def predict(self, frame: np.ndarray) -> Detection:
        """
        Dự đoán hành vi từ frame

        Args:
            frame: Input frame

        Returns:
            Detection result
        """
        # Preprocess
        input_data = self.preprocess_frame(frame)

        # Get input/output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        predictions = output_data[0]

        # Get top prediction
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]

        # Map to BehaviorType
        behavior_map = list(BehaviorType)
        behavior = behavior_map[class_id] if class_id < len(behavior_map) else BehaviorType.NORMAL

        # Create detection
        detection = Detection(
            behavior=behavior,
            confidence=float(confidence),
            timestamp=time.time(),
            frame_id=self.frame_count
        )

        self.frame_count += 1
        return detection

    def update_behavior_tracking(self, detection: Detection) -> bool:
        """
        Cập nhật tracking hành vi và kiểm tra cảnh báo

        Args:
            detection: Kết quả detection mới

        Returns:
            True nếu cần cảnh báo
        """
        should_alert = False

        # Check confidence threshold
        if detection.confidence < self.confidence_threshold:
            return False

        # Update history
        self.detections_history.append(detection)
        if len(self.detections_history) > 100:
            self.detections_history.pop(0)

        # Check behavior change
        if detection.behavior != self.current_behavior:
            self.current_behavior = detection.behavior
            self.behavior_start_time = time.time()

        # Check dangerous behavior duration
        if detection.is_dangerous:
            duration = time.time() - self.behavior_start_time
            if duration >= self.danger_duration_threshold:
                should_alert = True

        return should_alert

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Phân tích frame và trả về kết quả

        Args:
            frame: Input frame

        Returns:
            Dictionary chứa kết quả phân tích
        """
        # Detect behavior
        detection = self.predict(frame)

        # Update tracking
        should_alert = self.update_behavior_tracking(detection)

        # Calculate statistics
        stats = self.calculate_statistics()

        result = {
            'detection': detection,
            'should_alert': should_alert,
            'current_behavior': self.current_behavior.value,
            'behavior_duration': time.time() - self.behavior_start_time,
            'statistics': stats
        }

        return result

    def calculate_statistics(self) -> Dict:
        """
        Tính toán thống kê từ lịch sử detections

        Returns:
            Dictionary chứa các thống kê
        """
        if not self.detections_history:
            return {}

        # Count behaviors in last 30 seconds
        current_time = time.time()
        recent_detections = [
            d for d in self.detections_history
            if current_time - d.timestamp <= 30
        ]

        if not recent_detections:
            return {}

        # Calculate behavior distribution
        behavior_counts = {}
        for detection in recent_detections:
            behavior = detection.behavior.value
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        # Calculate percentages
        total = len(recent_detections)
        behavior_percentages = {
            behavior: (count / total) * 100
            for behavior, count in behavior_counts.items()
        }

        # Calculate average confidence
        avg_confidence = np.mean([d.confidence for d in recent_detections])

        # Danger score (0-100)
        danger_behaviors = [d for d in recent_detections if d.is_dangerous]
        danger_score = (len(danger_behaviors) / total) * 100 if total > 0 else 0

        return {
            'behavior_distribution': behavior_percentages,
            'average_confidence': float(avg_confidence),
            'danger_score': float(danger_score),
            'total_detections': total
        }

    def draw_overlay(
        self,
        frame: np.ndarray,
        detection: Detection,
        should_alert: bool
    ) -> np.ndarray:
        """
        Vẽ overlay lên frame

        Args:
            frame: Input frame
            detection: Detection result
            should_alert: Alert status

        Returns:
            Frame với overlay
        """
        height, width = frame.shape[:2]
        overlay = frame.copy()

        # Alert background
        if should_alert:
            # Red border for alert
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), 10)

        # Behavior text
        behavior_text = f"{detection.behavior.value.upper()}"
        confidence_text = f"Confidence: {detection.confidence:.2%}"

        # Text position
        text_y = 40

        # Background for text
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255) if detection.is_dangerous else (0, 255, 0)
        cv2.putText(overlay, behavior_text, (20, text_y),
                    font, 0.8, color, 2)
        cv2.putText(overlay, confidence_text, (20, text_y + 30),
                    font, 0.6, (255, 255, 255), 1)

        # Alert message
        if should_alert:
            alert_text = "DANGER! PLEASE FOCUS ON DRIVING!"
            text_size = cv2.getTextSize(alert_text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50

            # Background for alert
            cv2.rectangle(overlay, (text_x - 10, text_y - 35),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 255), -1)
            cv2.putText(overlay, alert_text, (text_x, text_y),
                       font, 1, (255, 255, 255), 2)

        # Blend with original
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def process_video_stream(
        self,
        video_source: int = 0,
        display: bool = True
    ):
        """
        Process video stream từ camera

        Args:
            video_source: Camera index hoặc video file path
            display: Hiển thị video output
        """
        cap = cv2.VideoCapture(video_source)

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting driver monitoring...")
        print("Press 'q' to quit")

        fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # Analyze frame
                result = self.analyze_frame(frame)
                detection = result['detection']
                should_alert = result['should_alert']

                # Draw overlay
                if display:
                    display_frame = self.draw_overlay(frame, detection, should_alert)

                    # Add FPS
                    cv2.putText(display_frame, f"FPS: {current_fps}",
                              (10, frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                              (255, 255, 255), 1)

                    # Show frame
                    cv2.imshow('Driver Monitoring System', display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Log dangerous behavior
                if should_alert:
                    print(f"[ALERT] Dangerous behavior detected: {detection.behavior.value}")
                    print(f"Duration: {result['behavior_duration']:.1f}s")

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark model performance

        Args:
            num_iterations: Số lần inference để test

        Returns:
            Benchmark results
        """
        # Create random input
        dummy_frame = np.random.randint(0, 255,
                                       (480, 640, 3),
                                       dtype=np.uint8)

        # Warm up
        for _ in range(10):
            self.predict(dummy_frame)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(dummy_frame)
            times.append(time.time() - start)

        # Calculate statistics
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time

        return {
            'average_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'fps': fps,
            'total_iterations': num_iterations
        }


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Driver Monitoring System')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to TFLite model')
    parser.add_argument('--source', type=int, default=0,
                       help='Video source (default: 0 for webcam)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')

    args = parser.parse_args()

    # Initialize monitor
    monitor = DriverMonitor(
        model_path=args.model,
        confidence_threshold=0.7,
        danger_duration_threshold=3.0
    )

    if args.benchmark:
        print("Running benchmark...")
        results = monitor.benchmark(100)
        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.2f}")
    else:
        # Process video stream
        monitor.process_video_stream(args.source)


if __name__ == "__main__":
    main()