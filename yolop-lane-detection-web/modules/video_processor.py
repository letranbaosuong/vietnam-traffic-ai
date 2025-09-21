"""
Video Processing Module
Handles video upload, processing, and output generation
"""

import cv2
import numpy as np
import os
from datetime import datetime
import json

class VideoProcessor:
    def __init__(self):
        """Initialize video processor"""
        self.supported_formats = ['mp4', 'avi', 'mov', 'mkv']
        self.output_fps = 30
        self.output_codec = 'mp4v'

    def validate_video(self, video_path):
        """
        Validate uploaded video
        Returns: dict with video info or error
        """
        if not os.path.exists(video_path):
            return {'valid': False, 'error': 'File không tồn tại'}

        # Check extension
        ext = video_path.rsplit('.', 1)[-1].lower()
        if ext not in self.supported_formats:
            return {'valid': False, 'error': f'Định dạng {ext} không được hỗ trợ'}

        # Open video to check
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'valid': False, 'error': 'Không thể mở file video'}

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'valid': True,
            'info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'format': ext
            }
        }

    def create_output_video(self, input_path, output_path, fps=None):
        """
        Create output video writer
        """
        cap = cv2.VideoCapture(input_path)

        # Get input video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        cap.release()

        # Use input fps if not specified
        if fps is None:
            fps = input_fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        return writer

    def draw_statistics_overlay(self, frame, stats):
        """
        Draw statistics overlay on frame
        """
        h, w = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()

        # Background for stats
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        # Draw statistics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        line_height = 20

        texts = [
            f"Frame: {stats.get('current_frame', 0)}/{stats.get('total_frames', 0)}",
            f"Vehicles: {stats.get('vehicle_count', 0)}",
            f"Lane Departures: {stats.get('lane_departures', 0)}",
            f"Warnings: {stats.get('warnings', 0)}"
        ]

        y = 30
        for text in texts:
            cv2.putText(frame, text, (20, y), font, font_scale, color, thickness)
            y += line_height

        return frame

    def generate_report(self, processing_stats):
        """
        Generate processing report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_frames': processing_stats.get('total_frames', 0),
                'processed_frames': processing_stats.get('processed_frames', 0),
                'processing_time': processing_stats.get('processing_time', 0),
                'fps': processing_stats.get('average_fps', 0)
            },
            'detections': {
                'total_vehicles': len(processing_stats.get('vehicles_detected', [])),
                'vehicle_types': processing_stats.get('vehicles_detected', []),
                'lane_departures': processing_stats.get('lane_departures', 0),
                'dangerous_situations': len(processing_stats.get('dangerous_situations', []))
            },
            'safety_analysis': {
                'risk_level': self.calculate_risk_level(processing_stats),
                'recommendations': self.generate_recommendations(processing_stats)
            },
            'events': processing_stats.get('dangerous_situations', [])
        }

        return report

    def calculate_risk_level(self, stats):
        """
        Calculate overall risk level
        """
        departures = stats.get('lane_departures', 0)
        frames = stats.get('processed_frames', 1)

        departure_rate = departures / frames if frames > 0 else 0

        if departure_rate > 0.1:
            return 'high'
        elif departure_rate > 0.05:
            return 'medium'
        else:
            return 'low'

    def generate_recommendations(self, stats):
        """
        Generate safety recommendations
        """
        recommendations = []

        departures = stats.get('lane_departures', 0)
        frames = stats.get('processed_frames', 1)
        departure_rate = departures / frames if frames > 0 else 0

        if departure_rate > 0.1:
            recommendations.append(
                'Tần suất lệch làn cao. Cần kiểm tra tình trạng lái xe và điều kiện đường.'
            )

        if departure_rate > 0.05:
            recommendations.append(
                'Có dấu hiệu mất tập trung. Nên nghỉ ngơi định kỳ khi lái xe đường dài.'
            )

        dangerous = stats.get('dangerous_situations', [])
        if len(dangerous) > 5:
            recommendations.append(
                'Nhiều tình huống nguy hiểm được phát hiện. Cần cải thiện kỹ năng lái xe.'
            )

        if not recommendations:
            recommendations.append('Lái xe an toàn. Tiếp tục duy trì thói quen lái xe tốt.')

        return recommendations

    def create_thumbnail(self, video_path, output_path, time_seconds=1):
        """
        Create thumbnail from video
        """
        cap = cv2.VideoCapture(video_path)

        # Seek to specified time
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * time_seconds)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            # Resize thumbnail
            thumbnail = cv2.resize(frame, (320, 180))
            cv2.imwrite(output_path, thumbnail)

        cap.release()
        return ret

    def extract_key_frames(self, video_path, num_frames=5):
        """
        Extract key frames from video for preview
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices to extract
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames