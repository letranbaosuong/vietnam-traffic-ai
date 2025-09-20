#!/usr/bin/env python3
"""
Raspberry Pi Driver Monitoring System
Lightweight version optimized for Raspberry Pi
"""

import cv2
import numpy as np
import time
import argparse
import yaml
import sys
import os
from datetime import datetime

# Import lightweight modules
from modules.face_detector_lite import FaceDetectorLite
from modules.drowsiness_detector_lite import DrowsinessDetectorLite
from modules.head_pose_lite import HeadPoseLite

# Try to import Pi camera
try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False
    print("PiCamera2 not found, will use USB camera")

class PiDriverMonitor:
    def __init__(self, config_path="config_pi.yaml"):
        """Initialize the driver monitoring system"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize detectors
        self.face_detector = FaceDetectorLite(
            model_path=self.config['model'].get('tflite_model_path'),
            use_coral=self.config['model'].get('use_coral', False)
        )

        self.drowsiness_detector = DrowsinessDetectorLite()
        self.head_pose_detector = HeadPoseLite()

        # Initialize camera
        self.camera = None
        self.setup_camera()

        # Performance settings
        self.skip_frames = self.config['performance']['skip_frames']
        self.frame_counter = 0

        # Alert system
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds

        # Statistics
        self.fps = 0
        self.frame_times = []

    def setup_camera(self):
        """Setup camera based on configuration"""
        cam_type = self.config['camera']['type']
        resolution = tuple(self.config['camera']['resolution'])

        if cam_type == "pi" and HAS_PICAMERA:
            # Use Raspberry Pi Camera
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            print(f"Initialized Pi Camera with resolution {resolution}")
        else:
            # Use USB camera
            device_id = self.config['camera']['device_id']
            self.camera = cv2.VideoCapture(device_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
            print(f"Initialized USB Camera {device_id} with resolution {resolution}")

    def get_frame(self):
        """Get frame from camera"""
        if HAS_PICAMERA and isinstance(self.camera, Picamera2):
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                return None

        # Apply flips if configured
        if self.config['camera']['flip_horizontal']:
            frame = cv2.flip(frame, 1)
        if self.config['camera']['flip_vertical']:
            frame = cv2.flip(frame, 0)

        return frame

    def process_frame(self, frame):
        """Process single frame for driver monitoring"""
        self.frame_counter += 1

        # Skip frames for performance
        if self.frame_counter % self.skip_frames != 0:
            return frame, None, None

        results = {
            'timestamp': datetime.now(),
            'face_detected': False,
            'drowsiness_status': 'Unknown',
            'distraction_status': 'Unknown',
            'alerts': []
        }

        # Detect faces
        faces = self.face_detector.detect(frame)

        if faces and len(faces) > 0:
            results['face_detected'] = True

            # Get the first (largest) face
            face = faces[0]
            face_roi = self.face_detector.get_face_roi(frame, face)

            if face_roi is not None and face_roi.size > 0:
                # Get face bounding box for drawing
                h, w = frame.shape[:2]
                bbox = face['bbox']

                if self.face_detector.use_tflite:
                    x = int(bbox[1] * w)
                    y = int(bbox[0] * h)
                    face_w = int((bbox[3] - bbox[1]) * w)
                    face_h = int((bbox[2] - bbox[0]) * h)
                else:
                    x = int(bbox[0] * w)
                    y = int(bbox[1] * h)
                    face_w = int((bbox[2] - bbox[0]) * w)
                    face_h = int((bbox[3] - bbox[1]) * h)

                face_bbox = (x, y, face_w, face_h)

                # Drowsiness detection
                if self.config['detection']['drowsiness']['enabled']:
                    drowsy_status, eye_status = self.drowsiness_detector.detect_drowsiness(face_roi)
                    results['drowsiness_status'] = drowsy_status

                    if drowsy_status == "DROWSY":
                        results['alerts'].append("CẢNH BÁO: Ngủ gật!")
                        self.trigger_alert("Ngủ gật phát hiện!")

                    # Draw eye indicators
                    frame = self.drowsiness_detector.draw_eyes(frame, face_roi, (x, y))

                # Head pose detection (distraction)
                if self.config['detection']['distraction']['enabled']:
                    yaw, pitch, direction = self.head_pose_detector.estimate_head_pose(face_roi)
                    distraction_status, detail = self.head_pose_detector.detect_distraction()
                    results['distraction_status'] = distraction_status

                    if distraction_status == "DISTRACTED":
                        results['alerts'].append(f"CẢNH BÁO: {detail}")
                        self.trigger_alert(detail)

                    # Draw head pose indicators
                    frame = self.head_pose_detector.draw_indicators(frame, face_bbox, yaw, pitch, direction)

                # Draw face bounding box
                color = (0, 0, 255) if results['alerts'] else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + face_w, y + face_h), color, 2)

        # Draw alerts on frame
        self.draw_alerts(frame, results['alerts'])

        # Draw FPS if enabled
        if self.config['display']['show_fps']:
            self.draw_fps(frame)

        return frame, results, faces

    def trigger_alert(self, message):
        """Trigger alert with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            self.last_alert_time = current_time
            self.alert_active = True

            # Console output
            if self.config['alerts']['console_output']:
                print(f"⚠️  ALERT: {message}")

            # Audio alert (beep)
            if self.config['alerts']['audio_enabled']:
                # Simple beep using system bell
                print('\a')

    def draw_alerts(self, frame, alerts):
        """Draw alert messages on frame"""
        if not alerts:
            return

        # Draw red overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
        frame[:80] = cv2.addWeighted(overlay[:80], 0.3, frame[:80], 0.7, 0)

        # Draw alert text
        y_offset = 30
        for alert in alerts[:2]:  # Show max 2 alerts
            cv2.putText(frame, alert, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35

    def draw_fps(self, frame):
        """Draw FPS counter on frame"""
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def calculate_fps(self):
        """Calculate rolling average FPS"""
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only recent times (last second)
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]

        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times)

    def run(self):
        """Main monitoring loop"""
        print("\n🚗 Raspberry Pi Driver Monitor Started")
        print("Press 'q' to quit, 's' for statistics, 'r' to reset\n")

        try:
            while True:
                # Get frame
                frame = self.get_frame()
                if frame is None:
                    continue

                # Process frame
                processed_frame, results, faces = self.process_frame(frame)

                # Calculate FPS
                self.calculate_fps()

                # Display if enabled
                if self.config['display']['show_video']:
                    cv2.imshow(self.config['display']['window_name'], processed_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.print_statistics()
                elif key == ord('r'):
                    self.reset_detectors()
                    print("Detectors reset")

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

        finally:
            self.cleanup()

    def print_statistics(self):
        """Print monitoring statistics"""
        print("\n" + "="*50)
        print("📊 MONITORING STATISTICS")
        print("="*50)

        # Drowsiness stats
        drowsy_stats = self.drowsiness_detector.get_statistics()
        print(f"Drowsiness Events: {drowsy_stats['drowsy_count']}")
        print(f"Total Drowsy Duration: {drowsy_stats['total_drowsy_duration']:.1f}s")
        print(f"Current Status: {drowsy_stats['current_status']}")

        # Distraction stats
        distraction_stats = self.head_pose_detector.get_statistics()
        print(f"\nDistracted: {distraction_stats['distracted']}")
        print(f"Distraction Type: {distraction_stats['distraction_type']}")
        print(f"Direction Counters - L:{distraction_stats['left_counter']} "
              f"R:{distraction_stats['right_counter']} "
              f"U:{distraction_stats['up_counter']} "
              f"D:{distraction_stats['down_counter']}")

        print(f"\nCurrent FPS: {self.fps:.1f}")
        print("="*50 + "\n")

    def reset_detectors(self):
        """Reset all detector states"""
        self.drowsiness_detector.reset()
        self.head_pose_detector.reset()
        self.alert_active = False

    def cleanup(self):
        """Clean up resources"""
        if HAS_PICAMERA and isinstance(self.camera, Picamera2):
            self.camera.stop()
        elif self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up")

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Driver Monitor')
    parser.add_argument('--config', type=str, default='config_pi.yaml',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=str, choices=['pi', 'usb'], default='pi',
                       help='Camera type to use')
    parser.add_argument('--device', type=int, default=0,
                       help='USB camera device ID')
    parser.add_argument('--use-coral', action='store_true',
                       help='Use Coral USB Accelerator')
    parser.add_argument('--headless', action='store_true',
                       help='Run without display output')

    args = parser.parse_args()

    # Override config with command line arguments
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        config['camera']['type'] = args.camera
        config['camera']['device_id'] = args.device
        config['model']['use_coral'] = args.use_coral
        config['display']['show_video'] = not args.headless

        # Save updated config to temp file
        temp_config = '/tmp/pi_monitor_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)

        monitor = PiDriverMonitor(temp_config)
    else:
        monitor = PiDriverMonitor()

    # Run the monitor
    monitor.run()

if __name__ == "__main__":
    main()