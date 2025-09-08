import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import yaml
import math

class PoseAnalyzer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        pose_config = self.config.get('pose', {})
        self.pose = self.mp_pose.Pose(
            model_complexity=pose_config.get('model_complexity', 1),
            min_detection_confidence=pose_config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=pose_config.get('min_tracking_confidence', 0.5)
        )
        
        # Define key body landmarks for traffic analysis
        self.key_landmarks = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
    
    def analyze_pose(self, frame: np.ndarray) -> List[Dict]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        poses = []
        if results.pose_landmarks:
            landmarks = self._extract_landmarks(results.pose_landmarks)
            pose_data = self._analyze_pose_data(landmarks, frame.shape)
            poses.append(pose_data)
        
        return poses
    
    def _extract_landmarks(self, pose_landmarks) -> Dict:
        landmarks = {}
        for name, idx in self.key_landmarks.items():
            landmark = pose_landmarks.landmark[idx]
            landmarks[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        return landmarks
    
    def _analyze_pose_data(self, landmarks: Dict, frame_shape: Tuple) -> Dict:
        h, w = frame_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        pixel_landmarks = {}
        for name, landmark in landmarks.items():
            pixel_landmarks[name] = {
                'x': int(landmark['x'] * w),
                'y': int(landmark['y'] * h),
                'visibility': landmark['visibility']
            }
        
        # Analyze pose characteristics
        pose_analysis = {
            'landmarks': pixel_landmarks,
            'head_direction': self._analyze_head_direction(landmarks),
            'body_orientation': self._analyze_body_orientation(landmarks),
            'arm_position': self._analyze_arm_position(landmarks),
            'walking_direction': self._analyze_walking_direction(landmarks),
            'pose_confidence': self._calculate_pose_confidence(landmarks),
            'traffic_behavior': self._analyze_traffic_behavior(landmarks)
        }
        
        return pose_analysis
    
    def _analyze_head_direction(self, landmarks: Dict) -> Dict:
        nose = landmarks['nose']
        left_ear = landmarks['left_ear']
        right_ear = landmarks['right_ear']
        
        if (left_ear['visibility'] < 0.5 or 
            right_ear['visibility'] < 0.5 or 
            nose['visibility'] < 0.5):
            return {'direction': 'unknown', 'angle': 0, 'confidence': 0}
        
        # Calculate head rotation angle
        ear_center_x = (left_ear['x'] + right_ear['x']) / 2
        nose_x = nose['x']
        
        # Horizontal head direction
        if nose_x < ear_center_x - 0.02:  # Looking right
            direction = 'right'
        elif nose_x > ear_center_x + 0.02:  # Looking left
            direction = 'left'
        else:
            direction = 'forward'
        
        # Calculate angle
        angle = math.degrees(math.atan2(nose_x - ear_center_x, 0.1))
        
        return {
            'direction': direction,
            'angle': angle,
            'confidence': min(left_ear['visibility'], right_ear['visibility'], nose['visibility'])
        }
    
    def _analyze_body_orientation(self, landmarks: Dict) -> Dict:
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        
        if (left_shoulder['visibility'] < 0.5 or 
            right_shoulder['visibility'] < 0.5):
            return {'orientation': 'unknown', 'angle': 0, 'confidence': 0}
        
        # Calculate shoulder angle
        dx = right_shoulder['x'] - left_shoulder['x']
        dy = right_shoulder['y'] - left_shoulder['y']
        angle = math.degrees(math.atan2(dy, dx))
        
        # Determine orientation
        if abs(angle) < 15:
            orientation = 'facing_camera'
        elif angle > 15:
            orientation = 'rotated_clockwise'
        else:
            orientation = 'rotated_counterclockwise'
        
        return {
            'orientation': orientation,
            'angle': angle,
            'confidence': min(left_shoulder['visibility'], right_shoulder['visibility'])
        }
    
    def _analyze_arm_position(self, landmarks: Dict) -> Dict:
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_wrist = landmarks['left_wrist']
        right_wrist = landmarks['right_wrist']
        
        arm_actions = []
        
        # Check if arms are raised (possible signaling)
        if (left_wrist['visibility'] > 0.5 and left_shoulder['visibility'] > 0.5):
            if left_wrist['y'] < left_shoulder['y']:
                arm_actions.append('left_arm_raised')
        
        if (right_wrist['visibility'] > 0.5 and right_shoulder['visibility'] > 0.5):
            if right_wrist['y'] < right_shoulder['y']:
                arm_actions.append('right_arm_raised')
        
        # Check for pointing or signaling gestures
        if len(arm_actions) == 1:
            if 'left_arm_raised' in arm_actions:
                gesture = 'possible_left_signal'
            else:
                gesture = 'possible_right_signal'
        elif len(arm_actions) == 2:
            gesture = 'both_arms_raised'
        else:
            gesture = 'normal'
        
        return {
            'actions': arm_actions,
            'gesture': gesture,
            'confidence': min(left_wrist['visibility'], right_wrist['visibility'])
        }
    
    def _analyze_walking_direction(self, landmarks: Dict) -> Dict:
        left_ankle = landmarks['left_ankle']
        right_ankle = landmarks['right_ankle']
        left_knee = landmarks['left_knee']
        right_knee = landmarks['right_knee']
        
        if (left_ankle['visibility'] < 0.5 or right_ankle['visibility'] < 0.5 or
            left_knee['visibility'] < 0.5 or right_knee['visibility'] < 0.5):
            return {'direction': 'unknown', 'confidence': 0, 'is_walking': False}
        
        # Calculate leg positions
        left_leg_angle = math.degrees(math.atan2(
            left_ankle['y'] - left_knee['y'],
            left_ankle['x'] - left_knee['x']
        ))
        
        right_leg_angle = math.degrees(math.atan2(
            right_ankle['y'] - right_knee['y'],
            right_ankle['x'] - right_knee['x']
        ))
        
        # Simple walking detection based on leg asymmetry
        leg_asymmetry = abs(left_leg_angle - right_leg_angle)
        is_walking = leg_asymmetry > 10
        
        # Determine approximate walking direction
        ankle_center_x = (left_ankle['x'] + right_ankle['x']) / 2
        knee_center_x = (left_knee['x'] + right_knee['x']) / 2
        
        if ankle_center_x < knee_center_x - 0.01:
            direction = 'moving_right'
        elif ankle_center_x > knee_center_x + 0.01:
            direction = 'moving_left'
        else:
            direction = 'stationary'
        
        return {
            'direction': direction,
            'is_walking': is_walking,
            'leg_asymmetry': leg_asymmetry,
            'confidence': min(left_ankle['visibility'], right_ankle['visibility'])
        }
    
    def _calculate_pose_confidence(self, landmarks: Dict) -> float:
        visible_landmarks = [lm for lm in landmarks.values() if lm['visibility'] > 0.5]
        if not visible_landmarks:
            return 0.0
        
        total_visibility = sum(lm['visibility'] for lm in visible_landmarks)
        return total_visibility / len(visible_landmarks)
    
    def _analyze_traffic_behavior(self, landmarks: Dict) -> Dict:
        head_dir = self._analyze_head_direction(landmarks)
        arm_pos = self._analyze_arm_position(landmarks)
        walking = self._analyze_walking_direction(landmarks)
        
        behaviors = []
        risk_level = 'low'
        
        # Analyze traffic-specific behaviors
        if head_dir['direction'] == 'left' or head_dir['direction'] == 'right':
            behaviors.append('looking_around')
        
        if arm_pos['gesture'] in ['possible_left_signal', 'possible_right_signal']:
            behaviors.append('signaling')
            risk_level = 'medium'
        
        if walking['is_walking'] and walking['direction'] != 'stationary':
            behaviors.append('in_motion')
            if head_dir['direction'] != 'forward':
                behaviors.append('distracted_walking')
                risk_level = 'high'
        
        return {
            'behaviors': behaviors,
            'risk_level': risk_level,
            'attention_level': 'high' if head_dir['direction'] == 'forward' else 'low'
        }
    
    def draw_pose(self, frame: np.ndarray, poses: List[Dict]) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for pose in poses:
            landmarks = pose['landmarks']
            
            # Draw skeleton connections
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]
            
            # Draw connections
            for start, end in connections:
                if (landmarks[start]['visibility'] > 0.5 and 
                    landmarks[end]['visibility'] > 0.5):
                    start_point = (landmarks[start]['x'], landmarks[start]['y'])
                    end_point = (landmarks[end]['x'], landmarks[end]['y'])
                    cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw key points
            for name, landmark in landmarks.items():
                if landmark['visibility'] > 0.5:
                    cv2.circle(annotated_frame, 
                             (landmark['x'], landmark['y']), 
                             5, (255, 0, 0), -1)
            
            # Add behavior annotations
            behavior = pose['traffic_behavior']
            y_offset = 30
            for i, behavior_text in enumerate(behavior['behaviors']):
                cv2.putText(annotated_frame, behavior_text, 
                           (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 255), 2)
            
            # Risk level indicator
            risk_color = {'low': (0, 255, 0), 'medium': (0, 255, 255), 'high': (0, 0, 255)}
            cv2.putText(annotated_frame, f"Risk: {behavior['risk_level']}", 
                       (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       risk_color.get(behavior['risk_level'], (255, 255, 255)), 2)
        
        return annotated_frame
    
    def get_pose_statistics(self, poses: List[Dict]) -> Dict:
        if not poses:
            return {'total_people': 0}
        
        stats = {
            'total_people': len(poses),
            'risk_levels': {'low': 0, 'medium': 0, 'high': 0},
            'behaviors': {},
            'average_confidence': 0.0
        }
        
        total_confidence = 0
        for pose in poses:
            # Risk level
            risk = pose['traffic_behavior']['risk_level']
            stats['risk_levels'][risk] += 1
            
            # Behaviors
            for behavior in pose['traffic_behavior']['behaviors']:
                stats['behaviors'][behavior] = stats['behaviors'].get(behavior, 0) + 1
            
            # Confidence
            total_confidence += pose['pose_confidence']
        
        stats['average_confidence'] = total_confidence / len(poses)
        return stats