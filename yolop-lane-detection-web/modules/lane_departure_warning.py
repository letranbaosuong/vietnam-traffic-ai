"""
Lane Departure Warning System
Vietnamese alerts for lane safety
"""

import numpy as np
import cv2

class LaneDepartureWarning:
    def __init__(self):
        """Initialize lane departure warning system"""
        # Vehicle position (center of bottom frame)
        self.vehicle_position = 0.5  # Normalized position (0=left, 1=right)

        # Lane departure thresholds
        self.departure_threshold = 0.15  # 15% deviation from center
        self.critical_threshold = 0.25   # 25% for critical warning

        # History for stability
        self.position_history = []
        self.history_size = 10

        # Warning state
        self.is_departed = False
        self.departure_direction = None
        self.consecutive_departures = 0

    def check_departure(self, lanes):
        """
        Check if vehicle is departing from lane
        Args:
            lanes: List of lane lines detected
        Returns:
            dict: departure status and warning message
        """
        if len(lanes) < 2:
            return {
                'departed': False,
                'message': '',
                'severity': 'none'
            }

        # Get lane boundaries
        left_lane = self.get_lane_x_at_vehicle(lanes[0])
        right_lane = self.get_lane_x_at_vehicle(lanes[1])

        if left_lane is None or right_lane is None:
            return {
                'departed': False,
                'message': '',
                'severity': 'none'
            }

        # Calculate vehicle position relative to lanes
        lane_center = (left_lane + right_lane) / 2
        lane_width = right_lane - left_lane

        if lane_width <= 0:
            return {
                'departed': False,
                'message': '',
                'severity': 'none'
            }

        # Normalized position within lane (0=left edge, 1=right edge)
        vehicle_relative_pos = (self.vehicle_position - left_lane) / lane_width

        # Update position history
        self.position_history.append(vehicle_relative_pos)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)

        # Calculate average position for stability
        if len(self.position_history) >= 3:
            avg_position = np.mean(self.position_history[-3:])
        else:
            avg_position = vehicle_relative_pos

        # Check departure
        departed = False
        message = ''
        severity = 'none'

        # Left departure
        if avg_position < self.departure_threshold:
            departed = True
            self.departure_direction = 'left'

            if avg_position < 0:
                severity = 'critical'
                message = '⚠️ NGUY HIỂM: Xe đang lấn làn TRÁI!'
            elif avg_position < self.departure_threshold / 2:
                severity = 'warning'
                message = '⚠️ CẢNH BÁO: Xe lệch sang TRÁI quá nhiều!'
            else:
                severity = 'mild'
                message = 'Chú ý: Xe đang lệch sang trái'

        # Right departure
        elif avg_position > (1 - self.departure_threshold):
            departed = True
            self.departure_direction = 'right'

            if avg_position > 1:
                severity = 'critical'
                message = '⚠️ NGUY HIỂM: Xe đang lấn làn PHẢI!'
            elif avg_position > (1 - self.departure_threshold / 2):
                severity = 'warning'
                message = '⚠️ CẢNH BÁO: Xe lệch sang PHẢI quá nhiều!'
            else:
                severity = 'mild'
                message = 'Chú ý: Xe đang lệch sang phải'

        # Update consecutive counter
        if departed:
            self.consecutive_departures += 1
        else:
            self.consecutive_departures = 0
            self.departure_direction = None

        # Escalate warning if persistent
        if self.consecutive_departures > 5:
            severity = 'critical'
            if self.departure_direction == 'left':
                message = '🚨 RẤT NGUY HIỂM: Xe liên tục lấn làn TRÁI!'
            else:
                message = '🚨 RẤT NGUY HIỂM: Xe liên tục lấn làn PHẢI!'

        self.is_departed = departed

        return {
            'departed': departed,
            'message': message,
            'severity': severity,
            'direction': self.departure_direction,
            'position': avg_position,
            'consecutive': self.consecutive_departures
        }

    def get_lane_x_at_vehicle(self, lane):
        """
        Get X coordinate of lane at vehicle position (bottom of frame)
        """
        if len(lane) < 2:
            return None

        # Get bottom point of lane (closest to vehicle)
        # Assuming lane points are sorted by Y coordinate
        bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)

        if len(bottom_points) > 0:
            # Normalize X coordinate (0-1)
            return bottom_points[0][0] / 640  # Assuming 640px width

        return None

    def calculate_lane_curvature(self, lane):
        """Calculate lane curvature for advanced warnings"""
        if len(lane) < 3:
            return 0

        # Fit polynomial to lane
        points = np.array(lane)
        x = points[:, 0]
        y = points[:, 1]

        # Fit second-order polynomial
        try:
            coeffs = np.polyfit(y, x, 2)
            # Curvature is related to second derivative
            curvature = abs(coeffs[0])
            return curvature
        except:
            return 0

    def get_safety_recommendations(self):
        """Get safety recommendations based on driving pattern"""
        recommendations = []

        if self.consecutive_departures > 10:
            recommendations.append({
                'type': 'critical',
                'message': 'Nghỉ ngơi ngay! Có dấu hiệu mất tập trung nghiêm trọng.'
            })
        elif self.consecutive_departures > 5:
            recommendations.append({
                'type': 'warning',
                'message': 'Nên dừng xe nghỉ ngơi. Có dấu hiệu buồn ngủ hoặc mất tập trung.'
            })

        if len(self.position_history) > 5:
            variance = np.var(self.position_history)
            if variance > 0.1:
                recommendations.append({
                    'type': 'info',
                    'message': 'Lái xe không ổn định. Giữ tay lái chắc chắn hơn.'
                })

        return recommendations

    def reset(self):
        """Reset warning system"""
        self.position_history = []
        self.is_departed = False
        self.departure_direction = None
        self.consecutive_departures = 0