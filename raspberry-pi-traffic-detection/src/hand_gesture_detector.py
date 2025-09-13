#!/usr/bin/env python3
"""
Hand Gesture Detection cho Traffic System
Phát hiện cử chỉ tay người lái xe (xi nhan, dừng, báo hiệu)
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from pathlib import Path

class HandGestureDetector:
    def __init__(self, confidence=0.7):
        """Initialize hand gesture detector"""
        self.confidence = confidence

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Gesture recognition
        self.gesture_history = []
        self.history_length = 10

        # Vietnamese traffic gestures
        self.traffic_gestures = {
            'xi_nhan_trai': 'Xi nhan trái - Turn Left Signal',
            'xi_nhan_phai': 'Xi nhan phải - Turn Right Signal',
            'dung_lai': 'Dừng lại - Stop Signal',
            'di_thang': 'Đi thẳng - Go Straight',
            'cham_lai': 'Chậm lại - Slow Down',
            'nguy_hiem': 'Nguy hiểm - Warning',
            'khong_co_cu_chi': 'Không có cử chỉ - No Gesture'
        }

    def detect_hands(self, frame):
        """Detect hands in frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand side (left/right)
                hand_side = handedness.classification[0].label

                # Extract landmark positions
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                hand_data.append({
                    'side': hand_side,
                    'landmarks': landmarks,
                    'raw_landmarks': hand_landmarks
                })

        return hand_data, results

    def classify_gesture(self, landmarks):
        """Classify hand gesture based on landmark positions"""
        if not landmarks or len(landmarks) < 21:
            return 'khong_co_cu_chi'

        # Convert to numpy array for easier calculation
        points = np.array(landmarks)

        # Get key landmark indices
        # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
        # Wrist: 0, Palm base: 9

        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        wrist = points[0]
        palm_base = points[9]

        # Calculate finger states (extended or folded)
        fingers_up = []

        # Thumb (check x-coordinate relative to palm)
        if thumb_tip[0] > points[3][0]:  # Thumb tip vs thumb joint
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # Other fingers (check y-coordinate - lower y means higher in image)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_joints = [6, 10, 14, 18]  # PIP joints

        for tip, joint in zip(finger_tips, finger_joints):
            if points[tip][1] < points[joint][1]:  # Tip is higher than joint
                fingers_up.append(1)
            else:
                fingers_up.append(0)

        # Gesture classification based on finger states
        total_fingers = sum(fingers_up)

        # Hand pointing gestures
        if fingers_up == [0, 1, 0, 0, 0]:  # Only index finger up
            # Determine direction based on hand position
            hand_center_x = np.mean([p[0] for p in points])
            if hand_center_x < 0.3:  # Left side of frame
                return 'xi_nhan_trai'
            elif hand_center_x > 0.7:  # Right side of frame
                return 'xi_nhan_phai'
            else:
                return 'di_thang'

        elif fingers_up == [1, 1, 1, 1, 1]:  # All fingers up (open palm)
            # Check if palm is facing camera (stop gesture)
            return 'dung_lai'

        elif fingers_up == [1, 1, 0, 0, 0]:  # Thumb and index up (peace sign)
            return 'cham_lai'

        elif total_fingers == 0:  # Closed fist
            return 'nguy_hiem'

        elif fingers_up == [0, 1, 1, 0, 0]:  # Index and middle up
            return 'di_thang'

        else:
            return 'khong_co_cu_chi'

    def smooth_gesture_detection(self, current_gesture):
        """Smooth gesture detection using history"""
        self.gesture_history.append(current_gesture)

        # Keep only recent history
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)

        # Find most common gesture in recent history
        if len(self.gesture_history) >= 3:
            gesture_counts = {}
            for gesture in self.gesture_history:
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

            # Return most common gesture if it appears frequently enough
            most_common = max(gesture_counts, key=gesture_counts.get)
            if gesture_counts[most_common] >= len(self.gesture_history) // 2:
                return most_common

        return current_gesture

    def detect_gestures(self, frame):
        """Main function to detect and classify gestures"""
        hand_data, mp_results = self.detect_hands(frame)

        detected_gestures = []

        for hand_info in hand_data:
            side = hand_info['side']
            landmarks = hand_info['landmarks']

            # Classify gesture
            gesture = self.classify_gesture(landmarks)

            # Smooth detection
            smoothed_gesture = self.smooth_gesture_detection(gesture)

            detected_gestures.append({
                'hand_side': side,
                'gesture': smoothed_gesture,
                'gesture_name': self.traffic_gestures.get(smoothed_gesture, smoothed_gesture),
                'landmarks': landmarks,
                'raw_landmarks': hand_info['raw_landmarks']
            })

        return detected_gestures, mp_results

    def draw_gestures(self, frame, gestures, mp_results):
        """Draw gesture detection results on frame"""
        result_frame = frame.copy()

        # Draw hand landmarks
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    result_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        # Draw gesture information
        y_offset = 30
        for i, gesture_info in enumerate(gestures):
            hand_side = gesture_info['hand_side']
            gesture = gesture_info['gesture']
            gesture_name = gesture_info['gesture_name']

            # Choose color based on gesture type
            if 'xi_nhan' in gesture:
                color = (0, 255, 255)  # Yellow for turn signals
            elif gesture == 'dung_lai':
                color = (0, 0, 255)    # Red for stop
            elif gesture == 'nguy_hiem':
                color = (0, 100, 255)  # Orange for warning
            elif gesture == 'cham_lai':
                color = (0, 165, 255)  # Orange for slow down
            else:
                color = (255, 255, 255) # White for others

            # Draw gesture text
            text = f"{hand_side}: {gesture_name}"

            # Background rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_frame, (10, y_offset - 25),
                         (20 + text_size[0], y_offset + 5), color, -1)

            # Gesture text
            cv2.putText(result_frame, text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            y_offset += 40

        # Add instructions
        instructions = [
            "Cu chi tay - Hand Gestures:",
            "1 ngon: Xi nhan | Open palm: Dung",
            "Closed fist: Nguy hiem | 2 ngon: Cham"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(result_frame, instruction, (10, frame.shape[0] - 60 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return result_frame

    def generate_gesture_alerts(self, gestures):
        """Generate alerts based on detected gestures"""
        alerts = []

        for gesture_info in gestures:
            gesture = gesture_info['gesture']
            hand_side = gesture_info['hand_side']

            if gesture == 'xi_nhan_trai':
                alerts.append({
                    'type': 'turn_signal',
                    'message': f'🤚 Xi nhan TRÁI ({hand_side} hand)',
                    'severity': 'info',
                    'action': 'turning_left'
                })
            elif gesture == 'xi_nhan_phai':
                alerts.append({
                    'type': 'turn_signal',
                    'message': f'🤚 Xi nhan PHẢI ({hand_side} hand)',
                    'severity': 'info',
                    'action': 'turning_right'
                })
            elif gesture == 'dung_lai':
                alerts.append({
                    'type': 'stop_signal',
                    'message': f'✋ DỪNG LẠI ({hand_side} hand)',
                    'severity': 'warning',
                    'action': 'stopping'
                })
            elif gesture == 'nguy_hiem':
                alerts.append({
                    'type': 'warning_signal',
                    'message': f'⚠️ NGUY HIỂM ({hand_side} hand)',
                    'severity': 'critical',
                    'action': 'warning'
                })
            elif gesture == 'cham_lai':
                alerts.append({
                    'type': 'slow_signal',
                    'message': f'🐌 CHẬM LẠI ({hand_side} hand)',
                    'severity': 'info',
                    'action': 'slowing'
                })

        return alerts

def main():
    """Test hand gesture detection"""
    detector = HandGestureDetector()

    # Test with webcam or video
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or video file path

    if not cap.isOpened():
        print("❌ Could not open camera/video")
        return

    print("🤚 Hand Gesture Detection Started")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect gestures
        gestures, mp_results = detector.detect_gestures(frame)

        # Draw results
        result_frame = detector.draw_gestures(frame, gestures, mp_results)

        # Generate alerts
        alerts = detector.generate_gesture_alerts(gestures)
        for alert in alerts:
            print(f"🚨 {alert['message']}")

        # Display frame
        cv2.imshow('Hand Gesture Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()