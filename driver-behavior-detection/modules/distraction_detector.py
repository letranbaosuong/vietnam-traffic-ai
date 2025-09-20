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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.HEAD_POSE_THRESH = 30
        self.HEAD_POSE_CONSEC_FRAMES = 20
        self.head_pose_counter = 0

        self.GAZE_DEVIATION_THRESH = 35
        self.GAZE_CONSEC_FRAMES = 25
        self.gaze_counter = 0

        self.distracted = False
        self.distraction_start_time = None
        self.total_distraction_duration = 0
        self.distraction_count = 0

        self.face_3d = []
        self.face_2d = []

    def calculate_head_pose(self, landmarks, img_shape):
        face_2d = []
        face_3d = []

        for idx, lm in enumerate(landmarks.landmark):
            if idx in [33, 263, 1, 61, 291, 199]:
                x, y = int(lm.x * img_shape[1]), int(lm.y * img_shape[0])
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_shape[1]
        cam_matrix = np.array([[focal_length, 0, img_shape[1] / 2],
                                [0, focal_length, img_shape[0] / 2],
                                [0, 0, 1]])

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360

        return x_angle, y_angle, z_angle

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

    def detect_looking_away(self, x_angle, y_angle):
        if abs(y_angle) > self.HEAD_POSE_THRESH or abs(x_angle) > self.HEAD_POSE_THRESH:
            return True
        return False

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        status = "Normal"
        distraction_type = ""
        head_angles = (0, 0, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            img_h, img_w = frame.shape[:2]

            x_angle, y_angle, z_angle = self.calculate_head_pose(landmarks, frame.shape)
            head_angles = (x_angle, y_angle, z_angle)

            gaze_angle, gaze_vector = self.calculate_gaze_direction(landmarks, frame.shape)

            looking_away = self.detect_looking_away(x_angle, y_angle)
            gaze_distracted = abs(gaze_angle) > self.GAZE_DEVIATION_THRESH

            if looking_away:
                self.head_pose_counter += 1
                if self.head_pose_counter >= self.HEAD_POSE_CONSEC_FRAMES:
                    status = "DISTRACTED"
                    if y_angle > self.HEAD_POSE_THRESH:
                        distraction_type = "Looking Right"
                    elif y_angle < -self.HEAD_POSE_THRESH:
                        distraction_type = "Looking Left"
                    elif x_angle > self.HEAD_POSE_THRESH:
                        distraction_type = "Looking Down"
                    elif x_angle < -self.HEAD_POSE_THRESH:
                        distraction_type = "Looking Up"

                    if not self.distracted:
                        self.distracted = True
                        self.distraction_start_time = time.time()
                        self.distraction_count += 1
            else:
                self.head_pose_counter = 0

            if gaze_distracted and status == "Normal":
                self.gaze_counter += 1
                if self.gaze_counter >= self.GAZE_CONSEC_FRAMES:
                    status = "GAZE DISTRACTED"
                    distraction_type = "Not Looking at Road"
            else:
                self.gaze_counter = 0

            if status == "Normal":
                if self.distracted and self.distraction_start_time:
                    self.total_distraction_duration += time.time() - self.distraction_start_time
                    self.distraction_start_time = None
                self.distracted = False

            nose_tip = landmarks.landmark[1]
            nose_x = int(nose_tip.x * img_w)
            nose_y = int(nose_tip.y * img_h)

            p1 = (nose_x, nose_y)
            p2 = (nose_x + int(y_angle * 5), nose_y - int(x_angle * 5))

            cv2.arrowedLine(frame, p1, p2, (255, 0, 255), 3)

            for idx in [33, 263, 1, 61, 291, 199]:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return status, distraction_type, head_angles, frame

    def get_statistics(self):
        return {
            'total_distraction_count': self.distraction_count,
            'total_distraction_duration': self.total_distraction_duration,
            'head_pose_counter': self.head_pose_counter,
            'gaze_counter': self.gaze_counter
        }