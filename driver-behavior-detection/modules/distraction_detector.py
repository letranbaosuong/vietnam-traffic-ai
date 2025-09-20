import cv2
import mediapipe as mp
import numpy as np
import time
import math

class DistractionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,  # Increased for better accuracy
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Optimized thresholds for better detection
        self.HEAD_POSE_THRESH_YAW = 25  # Left/Right threshold (reduced for sensitivity)
        self.HEAD_POSE_THRESH_PITCH = 20  # Up/Down threshold
        self.HEAD_POSE_CONSEC_FRAMES = 15  # Reduced for quicker response
        self.head_pose_counter = 0

        # Quick glance detection (for brief distractions)
        self.QUICK_GLANCE_THRESH = 35  # Larger angle for quick looks
        self.QUICK_GLANCE_FRAMES = 5  # Detect quick glances

        self.GAZE_DEVIATION_THRESH = 30  # More sensitive
        self.GAZE_CONSEC_FRAMES = 20
        self.gaze_counter = 0

        self.distracted = False
        self.distraction_start_time = None
        self.total_distraction_duration = 0
        self.distraction_count = 0

        # Track distraction history for pattern analysis
        self.angle_history = []
        self.MAX_HISTORY = 60  # Last 2-3 seconds

        # Counters for different directions
        self.left_counter = 0
        self.right_counter = 0
        self.up_counter = 0
        self.down_counter = 0

    def calculate_head_pose(self, landmarks, img_shape):
        # Using more stable landmarks for pose estimation
        # Nose tip (1), Chin (152), Left eye (33), Right eye (263),
        # Left mouth corner (61), Right mouth corner (291)

        face_2d = []
        face_3d = []

        # Key facial points for better pose estimation
        key_points = {
            1: "nose_tip",
            152: "chin",
            33: "left_eye",
            263: "right_eye",
            61: "left_mouth",
            291: "right_mouth",
            10: "forehead",  # Additional point for stability
            234: "right_cheek",
            454: "left_cheek"
        }

        for idx in key_points.keys():
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * img_shape[1]), int(lm.y * img_shape[0])
            face_2d.append([x, y])
            # Use actual 3D coordinates with proper scaling
            face_3d.append([lm.x * img_shape[1], lm.y * img_shape[0], lm.z * img_shape[1]])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera parameters (adjusted for better accuracy)
        focal_length = img_shape[1]
        center = (img_shape[1] / 2, img_shape[0] / 2)
        cam_matrix = np.array([[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP for rotation and translation vectors
        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # Get rotation matrix and angles
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Calculate Euler angles
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = x * 180.0 / math.pi  # Up/Down
        yaw = y * 180.0 / math.pi    # Left/Right
        roll = z * 180.0 / math.pi   # Tilt

        return pitch, yaw, roll

    def calculate_gaze_direction(self, landmarks, img_shape):
        left_eye_center = np.array([
            landmarks.landmark[33].x * img_shape[1],
            landmarks.landmark[33].y * img_shape[0]
        ])
        right_eye_center = np.array([
            landmarks.landmark[263].x * img_shape[1],
            landmarks.landmark[263].y * img_shape[0]
        ])

        left_pupil = np.array([
            landmarks.landmark[468].x * img_shape[1],
            landmarks.landmark[468].y * img_shape[0]
        ])
        right_pupil = np.array([
            landmarks.landmark[473].x * img_shape[1],
            landmarks.landmark[473].y * img_shape[0]
        ])

        left_gaze = left_pupil - left_eye_center
        right_gaze = right_pupil - right_eye_center

        avg_gaze = (left_gaze + right_gaze) / 2
        gaze_angle = math.degrees(math.atan2(avg_gaze[1], avg_gaze[0]))

        return gaze_angle, avg_gaze

    def detect_looking_away(self, pitch, yaw):
        # Separate thresholds for different directions
        looking_left = yaw < -self.HEAD_POSE_THRESH_YAW
        looking_right = yaw > self.HEAD_POSE_THRESH_YAW
        looking_up = pitch < -self.HEAD_POSE_THRESH_PITCH
        looking_down = pitch > self.HEAD_POSE_THRESH_PITCH

        # Quick glance detection
        quick_glance = abs(yaw) > self.QUICK_GLANCE_THRESH or abs(pitch) > self.QUICK_GLANCE_THRESH

        return looking_left, looking_right, looking_up, looking_down, quick_glance

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        distraction_type = ""
        head_angles = (0, 0, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            img_h, img_w = frame.shape[:2]

            # Calculate head pose angles
            pitch, yaw, roll = self.calculate_head_pose(landmarks, frame.shape)
            head_angles = (pitch, yaw, roll)

            # Store angle history for analysis
            self.angle_history.append((pitch, yaw, time.time()))
            if len(self.angle_history) > self.MAX_HISTORY:
                self.angle_history.pop(0)

            # Detect looking directions
            looking_left, looking_right, looking_up, looking_down, quick_glance = self.detect_looking_away(pitch, yaw)

            # Calculate gaze direction
            gaze_angle, gaze_vector = self.calculate_gaze_direction(landmarks, frame.shape)
            gaze_distracted = abs(gaze_angle) > self.GAZE_DEVIATION_THRESH

            # Update direction-specific counters
            if looking_left:
                self.left_counter += 1
                self.right_counter = max(0, self.right_counter - 1)
            elif looking_right:
                self.right_counter += 1
                self.left_counter = max(0, self.left_counter - 1)
            else:
                self.left_counter = max(0, self.left_counter - 1)
                self.right_counter = max(0, self.right_counter - 1)

            if looking_up:
                self.up_counter += 1
                self.down_counter = max(0, self.down_counter - 1)
            elif looking_down:
                self.down_counter += 1
                self.up_counter = max(0, self.up_counter - 1)
            else:
                self.up_counter = max(0, self.up_counter - 1)
                self.down_counter = max(0, self.down_counter - 1)

            # Check for sustained distraction
            if (self.left_counter >= self.HEAD_POSE_CONSEC_FRAMES or
                self.right_counter >= self.HEAD_POSE_CONSEC_FRAMES or
                self.up_counter >= self.HEAD_POSE_CONSEC_FRAMES or
                self.down_counter >= self.HEAD_POSE_CONSEC_FRAMES):

                status = "DISTRACTED"

                # Determine specific distraction type
                if self.left_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    distraction_type = "Nhìn sang TRÁI quá lâu!"
                elif self.right_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    distraction_type = "Nhìn sang PHẢI quá lâu!"
                elif self.down_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    distraction_type = "Nhìn XUỐNG quá lâu!"
                elif self.up_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    distraction_type = "Nhìn LÊN quá lâu!"

                if not self.distracted:
                    self.distracted = True
                    self.distraction_start_time = time.time()
                    self.distraction_count += 1

            # Check for quick glances (brief but extreme angles)
            elif quick_glance:
                self.head_pose_counter += 1
                if self.head_pose_counter >= self.QUICK_GLANCE_FRAMES:
                    status = "DISTRACTED"
                    if yaw < -self.QUICK_GLANCE_THRESH:
                        distraction_type = "Liếc nhanh sang TRÁI!"
                    elif yaw > self.QUICK_GLANCE_THRESH:
                        distraction_type = "Liếc nhanh sang PHẢI!"
                    elif pitch < -self.QUICK_GLANCE_THRESH:
                        distraction_type = "Nhìn lên nhanh!"
                    elif pitch > self.QUICK_GLANCE_THRESH:
                        distraction_type = "Nhìn xuống nhanh!"
            else:
                self.head_pose_counter = max(0, self.head_pose_counter - 1)

            # Check gaze distraction
            if gaze_distracted and status == "Normal":
                self.gaze_counter += 1
                if self.gaze_counter >= self.GAZE_CONSEC_FRAMES:
                    status = "GAZE DISTRACTED"
                    distraction_type = "Mắt không nhìn đường!"
            else:
                self.gaze_counter = max(0, self.gaze_counter - 1)

            # Reset if back to normal
            if status == "Normal":
                if self.distracted and self.distraction_start_time:
                    self.total_distraction_duration += time.time() - self.distraction_start_time
                    self.distraction_start_time = None
                self.distracted = False

            # Visual indicators
            nose_tip = landmarks.landmark[1]
            nose_x = int(nose_tip.x * img_w)
            nose_y = int(nose_tip.y * img_h)

            # Draw head direction arrow
            arrow_scale = 3
            p1 = (nose_x, nose_y)
            p2 = (nose_x + int(yaw * arrow_scale), nose_y - int(pitch * arrow_scale))

            # Color based on status
            arrow_color = (0, 255, 0) if status == "Normal" else (0, 0, 255)
            cv2.arrowedLine(frame, p1, p2, arrow_color, 3)

            # Draw angle text
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw face landmarks
            key_points = [1, 152, 33, 263, 61, 291]
            for idx in key_points:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return status, distraction_type, head_angles, frame

    def get_statistics(self):
        return {
            'total_distraction_count': self.distraction_count,
            'total_distraction_duration': self.total_distraction_duration,
            'left_counter': self.left_counter,
            'right_counter': self.right_counter,
            'up_counter': self.up_counter,
            'down_counter': self.down_counter,
            'gaze_counter': self.gaze_counter
        }