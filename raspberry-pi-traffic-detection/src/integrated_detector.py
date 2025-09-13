#!/usr/bin/env python3
"""
Integrated Traffic Detection System
Kết hợp vehicle detection, lane detection và traffic sign detection
"""

import cv2
import numpy as np
import time
from traffic_detector import TrafficDetector
from lane_detector import LaneDetector
from traffic_sign_detector import TrafficSignDetector
from hand_gesture_detector import HandGestureDetector

class IntegratedTrafficSystem:
    def __init__(self, model_type="yolov8n", confidence=0.5):
        """
        Initialize integrated traffic detection system
        """
        self.traffic_detector = TrafficDetector(model_type, confidence)
        self.lane_detector = LaneDetector()
        self.sign_detector = TrafficSignDetector(confidence)
        self.gesture_detector = HandGestureDetector(confidence)

        # Performance tracking
        self.fps_counter = 0
        self.start_time = time.time()

        # Detection modes
        self.enable_vehicles = True
        self.enable_lanes = True
        self.enable_signs = True
        self.enable_gestures = True

        # Alert system
        self.alerts = []

    def detect_all(self, frame):
        """
        Perform all detections on a single frame
        """
        results = {
            'vehicles': None,
            'lanes': None,
            'signs': None,
            'gestures': None,
            'alerts': [],
            'performance': {}
        }

        start_time = time.time()

        # Vehicle Detection
        if self.enable_vehicles:
            vehicle_start = time.time()
            # Use existing traffic detector but get raw results
            vehicle_frame = self.traffic_detector.detect_frame(frame)
            results['vehicles'] = {
                'frame': vehicle_frame,
                'processing_time': time.time() - vehicle_start
            }

        # Lane Detection
        if self.enable_lanes:
            lane_start = time.time()
            lane_data = self.lane_detector.detect_lanes(frame)
            results['lanes'] = {
                'data': lane_data,
                'processing_time': time.time() - lane_start
            }

        # Traffic Sign Detection
        if self.enable_signs:
            sign_start = time.time()
            sign_detections = self.sign_detector.detect_signs(frame)
            results['signs'] = {
                'detections': sign_detections,
                'processing_time': time.time() - sign_start
            }

        # Hand Gesture Detection
        if self.enable_gestures:
            gesture_start = time.time()
            gestures, mp_results = self.gesture_detector.detect_gestures(frame)
            results['gestures'] = {
                'detections': gestures,
                'mp_results': mp_results,
                'processing_time': time.time() - gesture_start
            }

        # Generate alerts
        results['alerts'] = self.generate_alerts(results)

        # Performance metrics
        total_time = time.time() - start_time
        results['performance'] = {
            'total_time': total_time,
            'fps': 1.0 / total_time if total_time > 0 else 0
        }

        return results

    def generate_alerts(self, results):
        """
        Generate safety alerts based on detections
        """
        alerts = []

        # Lane departure warning
        if self.enable_lanes and results['lanes']:
            lane_data = results['lanes']['data']
            if lane_data['direction'] != 'Straight':
                alerts.append({
                    'type': 'lane_departure',
                    'message': f"⚠️ {lane_data['direction']}",
                    'severity': 'warning'
                })

        # Traffic sign alerts
        if self.enable_signs and results['signs']:
            for detection in results['signs']['detections']:
                sign_type = detection['sign_type']

                if 'stop' in sign_type:
                    alerts.append({
                        'type': 'stop_sign',
                        'message': '🛑 STOP - Dừng lại!',
                        'severity': 'critical'
                    })
                elif 'speed_limit' in sign_type:
                    alerts.append({
                        'type': 'speed_limit',
                        'message': f'🚫 {self.sign_detector.vietnamese_signs.get(sign_type, sign_type)}',
                        'severity': 'warning'
                    })
                elif 'yield' in sign_type:
                    alerts.append({
                        'type': 'yield',
                        'message': '⚠️ Nhường đường',
                        'severity': 'warning'
                    })

        # Hand gesture alerts
        if self.enable_gestures and results['gestures']:
            gesture_alerts = self.gesture_detector.generate_gesture_alerts(results['gestures']['detections'])
            alerts.extend(gesture_alerts)

        return alerts

    def draw_integrated_results(self, frame, results):
        """
        Draw all detection results on a single frame
        """
        result_frame = frame.copy()

        # Draw vehicle detections (if available)
        if results['vehicles']:
            vehicle_frame = results['vehicles']['frame']
            # Extract just the bounding boxes from the vehicle frame
            # This is a simplified approach - in practice you'd want to overlay properly
            result_frame = vehicle_frame

        # Draw lane detection
        if results['lanes']:
            lane_data = results['lanes']['data']
            result_frame = self.lane_detector.draw_lanes(result_frame, lane_data)

        # Draw traffic sign detections
        if results['signs']:
            sign_detections = results['signs']['detections']
            result_frame = self.sign_detector.draw_detections(result_frame, sign_detections)

        # Draw hand gesture detections
        if results['gestures']:
            gesture_detections = results['gestures']['detections']
            mp_results = results['gestures']['mp_results']
            result_frame = self.gesture_detector.draw_gestures(result_frame, gesture_detections, mp_results)

        # Draw alerts
        self.draw_alerts(result_frame, results['alerts'])

        # Draw performance info
        self.draw_performance_info(result_frame, results['performance'])

        return result_frame

    def draw_alerts(self, frame, alerts):
        """
        Draw safety alerts on frame
        """
        y_offset = 100
        for alert in alerts:
            color = (0, 0, 255) if alert['severity'] == 'critical' else (0, 165, 255)

            # Draw alert background
            text_size = cv2.getTextSize(alert['message'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (10, y_offset - 25), (20 + text_size[0], y_offset + 5), color, -1)

            # Draw alert text
            cv2.putText(frame, alert['message'], (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            y_offset += 40

    def draw_performance_info(self, frame, performance):
        """
        Draw performance metrics
        """
        height, width = frame.shape[:2]

        # Performance info
        fps = performance.get('fps', 0)
        total_time = performance.get('total_time', 0)

        info_text = [
            f"Integrated FPS: {fps:.1f}",
            f"Total Time: {total_time*1000:.1f}ms"
        ]

        # Draw on top right
        for i, text in enumerate(info_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_pos = width - text_size[0] - 10
            y_pos = 30 + i * 25

            cv2.putText(frame, text, (x_pos, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def process_video(self, video_path, output_path=None):
        """
        Process video with integrated detection
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"🎬 Processing integrated detection: {video_path}")
        print(f"📺 Resolution: {width}x{height}, FPS: {fps}")
        print(f"🚗 Vehicle Detection: {'ON' if self.enable_vehicles else 'OFF'}")
        print(f"🛣️  Lane Detection: {'ON' if self.enable_lanes else 'OFF'}")
        print(f"🚦 Sign Detection: {'ON' if self.enable_signs else 'OFF'}")
        print(f"🤚 Gesture Detection: {'ON' if self.enable_gestures else 'OFF'}")

        frame_count = 0
        total_processing_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with all detections
            results = self.detect_all(frame)

            # Draw integrated results
            result_frame = self.draw_integrated_results(frame, results)

            # Save frame if needed
            if writer:
                writer.write(result_frame)

            # Display frame
            cv2.imshow('Integrated Traffic Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            total_processing_time += results['performance']['total_time']

            # Progress update
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
                print(f"📊 Processed {frame_count} frames, Avg FPS: {avg_fps:.2f}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Final statistics
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        print(f"\n✅ Processing complete!")
        print(f"📈 Frames processed: {frame_count}")
        print(f"⚡ Average FPS: {avg_fps:.2f}")
        print(f"⏱️  Total time: {total_processing_time:.2f}s")

    def toggle_detection(self, detection_type, enable=None):
        """
        Enable/disable specific detection types
        """
        if detection_type == 'vehicles':
            self.enable_vehicles = enable if enable is not None else not self.enable_vehicles
            status = "ON" if self.enable_vehicles else "OFF"
            print(f"🚗 Vehicle Detection: {status}")

        elif detection_type == 'lanes':
            self.enable_lanes = enable if enable is not None else not self.enable_lanes
            status = "ON" if self.enable_lanes else "OFF"
            print(f"🛣️  Lane Detection: {status}")

        elif detection_type == 'signs':
            self.enable_signs = enable if enable is not None else not self.enable_signs
            status = "ON" if self.enable_signs else "OFF"
            print(f"🚦 Sign Detection: {status}")

        elif detection_type == 'gestures':
            self.enable_gestures = enable if enable is not None else not self.enable_gestures
            status = "ON" if self.enable_gestures else "OFF"
            print(f"🤚 Gesture Detection: {status}")

def main():
    """Main function for integrated traffic detection"""
    # Initialize integrated system
    system = IntegratedTrafficSystem(model_type="yolov8n", confidence=0.5)

    # Test with sample video
    # video_path = "video/sample.mp4"
    # video_path = "video/car-driver.mp4"
    video_path = "video/mobycle-driver.mp4"
    output_path = "outputs/integrated_detection.mp4"

    # Process video
    system.process_video(video_path, output_path)

if __name__ == "__main__":
    main()