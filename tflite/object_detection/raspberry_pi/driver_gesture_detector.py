"""
Driver Gesture Detection Module for Raspberry Pi
Detects dangerous driving behaviors and gestures
Optimized for real-time performance on Raspberry Pi

Reuses existing MediaPipe patterns from driver-behavior-detection project
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from typing import Dict, Tuple, Optional, List


class DriverGestureDetector:
    """
    Lightweight driver gesture detector for Raspberry Pi
    Detects dangerous behaviors: phone usage, distraction, hands off wheel
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize gesture detector

        Args:
            config: Optional configuration dict with thresholds
        """
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        # Optimized for Raspberry Pi - lower complexity for better FPS
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,  # Balanced for Pi
            min_tracking_confidence=0.5,
            model_complexity=0  # Lightweight model
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightweight model for Pi
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Faster on Pi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Visualization
        self.mp_drawing = mp.solutions.drawing_utils

        # Detection thresholds
        config = config or {}
        self.PHONE_THRESH = config.get('phone_thresh', 0.12)
        self.PHONE_FRAMES = config.get('phone_frames', 15)
        self.DISTRACTION_YAW_THRESH = config.get('distraction_yaw', 25)
        self.DISTRACTION_PITCH_THRESH = config.get('distraction_pitch', 20)
        self.DISTRACTION_FRAMES = config.get('distraction_frames', 12)
        self.HANDS_OFF_FRAMES = config.get('hands_off_frames', 20)

        # State tracking
        self.phone_counter = 0
        self.distraction_counter = 0
        self.hands_off_counter = 0

        # Statistics
        self.detection_stats = {
            'phone_usage': 0,
            'distraction': 0,
            'hands_off_wheel': 0,
            'dangerous_gesture': 0
        }

        self.current_warnings = []

    def _calculate_head_pose(self, landmarks, img_shape: Tuple) -> Tuple[float, float, float]:
        """
        Calculate head pose angles (pitch, yaw, roll)
        Reused from distraction_detector.py pattern
        """
        face_2d = []
        face_3d = []

        # Key facial points for pose estimation
        key_points = [1, 152, 33, 263, 61, 291]  # Nose, chin, eyes, mouth

        for idx in key_points:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * img_shape[1]), int(lm.y * img_shape[0])
            face_2d.append([x, y])
            face_3d.append([lm.x * img_shape[1], lm.y * img_shape[0], lm.z * img_shape[1]])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera parameters
        focal_length = img_shape[1]
        center = (img_shape[1] / 2, img_shape[0] / 2)
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0, 0, 0

        # Get rotation matrix and angles
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Calculate Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

        pitch = x * 180.0 / math.pi
        yaw = y * 180.0 / math.pi
        roll = z * 180.0 / math.pi

        return pitch, yaw, roll

    def _detect_phone_usage(self, hand_landmarks, pose_landmarks) -> Tuple[bool, str]:
        """
        Detect phone usage patterns
        Reused from phone_detector.py pattern
        """
        if not hand_landmarks or not pose_landmarks:
            return False, ""

        # Get hand center position
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        hand_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
        hand_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3

        # Get face landmarks
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]

        # Check hand near ear (phone call)
        dist_left_ear = np.sqrt((hand_x - left_ear.x)**2 + (hand_y - left_ear.y)**2)
        dist_right_ear = np.sqrt((hand_x - right_ear.x)**2 + (hand_y - right_ear.y)**2)

        if dist_left_ear < self.PHONE_THRESH or dist_right_ear < self.PHONE_THRESH:
            return True, "Đang gọi điện thoại"

        # Check hand in front of face (texting/viewing)
        dist_nose = np.sqrt((hand_x - nose.x)**2 + (hand_y - nose.y)**2)
        if 0.15 < dist_nose < 0.35 and hand_y < nose.y + 0.15:
            return True, "Đang xem điện thoại"

        return False, ""

    def _detect_distraction(self, face_landmarks, img_shape: Tuple) -> Tuple[bool, str]:
        """
        Detect head distraction (looking away)
        Reused from distraction_detector.py pattern
        """
        if not face_landmarks:
            return False, ""

        pitch, yaw, roll = self._calculate_head_pose(face_landmarks, img_shape)

        # Check for dangerous head angles
        if yaw < -self.DISTRACTION_YAW_THRESH:
            return True, f"Đang nhìn sang TRÁI ({abs(yaw):.0f}°)"
        elif yaw > self.DISTRACTION_YAW_THRESH:
            return True, f"Đang nhìn sang PHẢI ({yaw:.0f}°)"
        elif pitch < -self.DISTRACTION_PITCH_THRESH:
            return True, f"Đang nhìn XUỐNG ({abs(pitch):.0f}°)"
        elif pitch > self.DISTRACTION_PITCH_THRESH:
            return True, f"Đang nhìn LÊN ({pitch:.0f}°)"

        return False, ""

    def _detect_hands_off_wheel(self, hand_results, pose_landmarks) -> Tuple[bool, str]:
        """
        Detect if hands are off the steering wheel
        """
        if not pose_landmarks:
            return False, ""

        # If no hands detected, likely hands are down/off wheel
        if not hand_results.multi_hand_landmarks:
            return True, "Tay rời vô lăng"

        # Check if hands are in driving position
        # Steering wheel should be around chest level
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        hands_in_position = False
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            # Check if hand is at reasonable driving height
            if wrist.y < shoulder_y + 0.15 and wrist.y > shoulder_y - 0.15:
                hands_in_position = True
                break

        if not hands_in_position:
            return True, "Tay không ở vị trí lái"

        return False, ""

    def detect(self, frame: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Main detection method

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (warnings_list, annotated_frame)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run all detections
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        warnings = []

        # Phone detection
        if hand_results.multi_hand_landmarks and pose_results.pose_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                is_phone, phone_msg = self._detect_phone_usage(
                    hand_landmarks, pose_results.pose_landmarks
                )
                if is_phone:
                    self.phone_counter += 1
                    if self.phone_counter >= self.PHONE_FRAMES:
                        warnings.append(f"⚠️ NGUY HIỂM: {phone_msg}!")
                        self.detection_stats['phone_usage'] += 1
                    break
            else:
                self.phone_counter = max(0, self.phone_counter - 1)
        else:
            self.phone_counter = max(0, self.phone_counter - 1)

        # Distraction detection
        if face_results.multi_face_landmarks:
            is_distracted, distraction_msg = self._detect_distraction(
                face_results.multi_face_landmarks[0], frame.shape
            )
            if is_distracted:
                self.distraction_counter += 1
                if self.distraction_counter >= self.DISTRACTION_FRAMES:
                    warnings.append(f"⚠️ MẤT TẬP TRUNG: {distraction_msg}!")
                    self.detection_stats['distraction'] += 1
            else:
                self.distraction_counter = max(0, self.distraction_counter - 1)
        else:
            self.distraction_counter = max(0, self.distraction_counter - 1)

        # Hands off wheel detection
        if pose_results.pose_landmarks:
            hands_off, hands_msg = self._detect_hands_off_wheel(
                hand_results, pose_results.pose_landmarks
            )
            if hands_off:
                self.hands_off_counter += 1
                if self.hands_off_counter >= self.HANDS_OFF_FRAMES:
                    warnings.append(f"⚠️ CẢNH BÁO: {hands_msg}!")
                    self.detection_stats['hands_off_wheel'] += 1
            else:
                self.hands_off_counter = max(0, self.hands_off_counter - 1)

        # Update current warnings
        self.current_warnings = warnings

        # Update dangerous gesture count
        if warnings:
            self.detection_stats['dangerous_gesture'] += 1

        return warnings, frame

    def visualize(self, frame: np.ndarray, warnings: List[str],
                  show_landmarks: bool = False) -> np.ndarray:
        """
        Visualize warnings on frame

        Args:
            frame: Input frame
            warnings: List of warning messages
            show_landmarks: Whether to show MediaPipe landmarks

        Returns:
            Annotated frame
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # Draw warnings
        if warnings:
            # Create alert overlay
            overlay = output.copy()
            alert_height = min(100, h // 4)

            # Red overlay for danger
            cv2.rectangle(overlay, (0, 0), (w, alert_height), (0, 0, 255), -1)
            cv2.addWeighted(overlay[:alert_height], 0.3,
                          output[:alert_height], 0.7, 0,
                          output[:alert_height])

            # Draw warning text
            y_offset = 30
            for warning in warnings:
                # Draw text with outline for readability
                cv2.putText(output, warning, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(output, warning, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

        # Draw status indicator
        status_text = f"Gestures: {len(warnings)} warning(s)"
        status_color = (0, 0, 255) if warnings else (0, 255, 0)
        cv2.putText(output, status_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return output

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'phone_usage_detected': self.detection_stats['phone_usage'],
            'distraction_detected': self.detection_stats['distraction'],
            'hands_off_detected': self.detection_stats['hands_off_wheel'],
            'total_dangerous_gestures': self.detection_stats['dangerous_gesture'],
            'current_warnings': len(self.current_warnings),
            'phone_counter': self.phone_counter,
            'distraction_counter': self.distraction_counter,
            'hands_off_counter': self.hands_off_counter
        }

    def reset_statistics(self):
        """Reset all statistics"""
        self.detection_stats = {
            'phone_usage': 0,
            'distraction': 0,
            'hands_off_wheel': 0,
            'dangerous_gesture': 0
        }
        self.phone_counter = 0
        self.distraction_counter = 0
        self.hands_off_counter = 0
        self.current_warnings = []
