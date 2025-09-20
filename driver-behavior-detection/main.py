import cv2
import numpy as np
import time
import argparse
from datetime import datetime
import os
import sys

from modules.drowsiness_detector import DrowsinessDetector
from modules.yawn_detector import YawnDetector
from modules.distraction_detector import DistractionDetector
from modules.phone_detector import PhoneDetector
from modules.smoking_detector import SmokingDetector
from utils.alert_system import AlertSystem
from utils.video_recorder import VideoRecorder, EventLogger

class DriverBehaviorMonitor:
    def __init__(self, config=None):
        self.drowsiness_detector = DrowsinessDetector()
        self.yawn_detector = YawnDetector()
        self.distraction_detector = DistractionDetector()
        self.phone_detector = PhoneDetector()
        self.smoking_detector = SmokingDetector()

        self.alert_system = AlertSystem()
        self.video_recorder = VideoRecorder()
        self.event_logger = EventLogger()

        self.enable_recording = config.get('enable_recording', False) if config else False
        self.show_stats = config.get('show_stats', True) if config else True
        self.debug_mode = config.get('debug_mode', False) if config else False

        self.session_start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()

    def draw_status_panel(self, frame, statuses):
        h, w = frame.shape[:2]
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        cv2.putText(panel, "DRIVER BEHAVIOR MONITORING SYSTEM",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(panel, f"FPS: {self.fps:.1f}",
                   (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        y_offset = 70
        for detector_name, status_info in statuses.items():
            status = status_info.get('status', 'Unknown')
            value = status_info.get('value', 0)
            details = status_info.get('details', '')

            if status != "Normal":
                color = self.alert_system.get_alert_color(status)
            else:
                color = (0, 255, 0)

            status_text = f"{detector_name}: {status}"
            if value > 0:
                status_text += f" ({value:.2f})"
            if details:
                status_text += f" - {details}"

            cv2.putText(panel, status_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25

        combined = np.vstack([panel, frame])
        return combined

    def draw_alert_overlay(self, frame, alert_message, alert_type):
        if alert_message:
            h, w = frame.shape[:2]
            overlay = frame.copy()

            color = self.alert_system.get_alert_color(alert_type)
            cv2.rectangle(overlay, (0, 0), (w, 100), color, -1)

            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

            text_size = cv2.getTextSize(alert_message,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = 60

            cv2.putText(frame, alert_message,
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)

            cv2.rectangle(frame, (0, 0), (w, 100), color, 3)

        return frame

    def process_frame(self, frame):
        statuses = {}
        alert_triggered = False
        alert_message = ""
        alert_type = ""

        drowsy_status, ear_value, frame = self.drowsiness_detector.detect(frame)
        statuses['Drowsiness'] = {
            'status': drowsy_status,
            'value': ear_value,
            'details': f"EAR: {ear_value:.2f}"
        }
        if drowsy_status == "DROWSY":
            alert_message = self.alert_system.trigger_alert('DROWSY')
            alert_type = 'DROWSY'
            alert_triggered = True
            self.event_logger.log_event('DROWSY', severity='CRITICAL')

        yawn_status, mar_value, frame = self.yawn_detector.detect(frame)
        statuses['Yawning'] = {
            'status': yawn_status,
            'value': mar_value,
            'details': f"MAR: {mar_value:.2f}"
        }
        if yawn_status == "YAWNING" and not alert_triggered:
            alert_message = self.alert_system.trigger_alert('YAWNING')
            alert_type = 'YAWNING'
            alert_triggered = True
            self.event_logger.log_event('YAWNING', severity='LOW')

        distraction_status, distraction_type, head_angles, frame = self.distraction_detector.detect(frame)
        statuses['Attention'] = {
            'status': distraction_status,
            'value': 0,
            'details': distraction_type
        }
        if distraction_status in ["DISTRACTED", "GAZE DISTRACTED"] and not alert_triggered:
            alert_message = self.alert_system.trigger_alert(distraction_status)
            alert_type = distraction_status
            alert_triggered = True
            self.event_logger.log_event(distraction_status, severity='MEDIUM', details=distraction_type)

        phone_status, phone_type, frame = self.phone_detector.detect(frame)
        statuses['Phone'] = {
            'status': phone_status,
            'value': 0,
            'details': phone_type
        }
        if phone_status == "PHONE USAGE" and not alert_triggered:
            alert_message = self.alert_system.trigger_alert('PHONE USAGE')
            alert_type = 'PHONE USAGE'
            alert_triggered = True
            self.event_logger.log_event('PHONE USAGE', severity='HIGH', details=phone_type)

        smoking_status, smoking_type, frame = self.smoking_detector.detect(frame)
        statuses['Smoking'] = {
            'status': smoking_status,
            'value': 0,
            'details': smoking_type
        }
        if smoking_status == "SMOKING DETECTED" and not alert_triggered:
            alert_message = self.alert_system.trigger_alert('SMOKING DETECTED')
            alert_type = 'SMOKING DETECTED'
            alert_triggered = True
            self.event_logger.log_event('SMOKING DETECTED', severity='HIGH')

        if alert_triggered and self.enable_recording:
            if not self.video_recorder.is_recording:
                h, w = frame.shape[:2]
                self.video_recorder.start_recording(w, h + 200, 20, alert_type)

        if self.video_recorder.is_recording:
            self.video_recorder.add_frame(frame)

        if alert_message:
            frame = self.draw_alert_overlay(frame, alert_message, alert_type)

        if self.show_stats:
            frame = self.draw_status_panel(frame, statuses)

        return frame, statuses, alert_triggered

    def update_fps(self):
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_update > 1:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def run(self, source=0):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Error: Cannot open video source {source}")
            return

        print("Driver Behavior Monitoring System Started")
        print("Press 'q' to quit")
        print("Press 's' to save snapshot")
        print("Press 'r' to toggle recording")
        print("Press 'd' to toggle debug mode")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read frame")
                    break

                self.update_fps()

                processed_frame, statuses, alert_triggered = self.process_frame(frame)

                cv2.imshow('Driver Behavior Monitor', processed_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = self.video_recorder.save_snapshot(processed_frame)
                    print(f"Snapshot saved: {filename}")
                elif key == ord('r'):
                    self.enable_recording = not self.enable_recording
                    print(f"Recording: {'ON' if self.enable_recording else 'OFF'}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

        finally:
            if self.video_recorder.is_recording:
                saved_file = self.video_recorder.stop_recording()
                print(f"Recording saved: {saved_file}")

            cap.release()
            cv2.destroyAllWindows()
            self.alert_system.cleanup()

            stats = self.event_logger.get_statistics()
            print("\n=== Session Statistics ===")
            print(f"Total events: {stats.get('total_events', 0)}")
            for event_type, count in stats.get('events_by_type', {}).items():
                print(f"  {event_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Driver Behavior Monitoring System')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam, or video file path)')
    parser.add_argument('--record', action='store_true',
                       help='Enable video recording for alerts')
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable statistics display')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')

    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    config = {
        'enable_recording': args.record,
        'show_stats': not args.no_stats,
        'debug_mode': args.debug
    }

    monitor = DriverBehaviorMonitor(config)
    monitor.run(source)

if __name__ == "__main__":
    main()