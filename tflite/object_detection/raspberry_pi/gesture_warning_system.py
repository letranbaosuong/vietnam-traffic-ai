"""
Gesture Warning System for Driver Safety
Simple and maintainable warning management
Vietnamese language support
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional
from datetime import datetime


class GestureWarningSystem:
    """
    Manages warnings for dangerous driver gestures
    Provides visual and audio alerts
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize warning system

        Args:
            config: Optional configuration dict
        """
        config = config or {}

        # Warning levels
        self.WARNING_LEVELS = {
            'LOW': {'color': (0, 255, 255), 'priority': 1},     # Yellow
            'MEDIUM': {'color': (0, 165, 255), 'priority': 2},  # Orange
            'HIGH': {'color': (0, 0, 255), 'priority': 3},      # Red
            'CRITICAL': {'color': (255, 0, 255), 'priority': 4}  # Purple
        }

        # Warning categories and their levels
        self.WARNING_CATEGORIES = {
            'phone_usage': 'HIGH',
            'distraction': 'MEDIUM',
            'hands_off_wheel': 'MEDIUM',
            'drowsiness': 'CRITICAL',
            'unknown': 'LOW'
        }

        # Alert messages (Vietnamese)
        self.ALERT_MESSAGES = {
            'phone_usage': '⚠️ NGUY HIỂM: Đang sử dụng điện thoại!',
            'distraction': '⚠️ CHÚ Ý: Mất tập trung!',
            'hands_off_wheel': '⚠️ CẢNH BÁO: Tay rời vô lăng!',
            'drowsiness': '🚨 RẤT NGUY HIỂM: Buồn ngủ!',
            'general': '⚠️ CẢNH BÁO!'
        }

        # State tracking
        self.active_warnings = []
        self.warning_history = []
        self.last_warning_time = {}

        # Configuration
        self.warning_cooldown = config.get('warning_cooldown', 2.0)  # seconds
        self.max_history = config.get('max_history', 100)

        # Statistics
        self.stats = {
            'total_warnings': 0,
            'warnings_by_type': {},
            'critical_warnings': 0
        }

    def add_warning(self, warning_text: str, category: str = 'unknown') -> bool:
        """
        Add a warning to the system

        Args:
            warning_text: Warning message text
            category: Warning category (phone_usage, distraction, etc.)

        Returns:
            True if warning was added, False if in cooldown
        """
        current_time = time.time()

        # Check cooldown
        if category in self.last_warning_time:
            time_since_last = current_time - self.last_warning_time[category]
            if time_since_last < self.warning_cooldown:
                return False

        # Get warning level
        level = self.WARNING_CATEGORIES.get(category, 'LOW')

        # Create warning record
        warning = {
            'text': warning_text,
            'category': category,
            'level': level,
            'timestamp': current_time,
            'datetime': datetime.now().isoformat()
        }

        # Add to active warnings
        self.active_warnings.append(warning)

        # Add to history
        self.warning_history.append(warning)
        if len(self.warning_history) > self.max_history:
            self.warning_history.pop(0)

        # Update last warning time
        self.last_warning_time[category] = current_time

        # Update statistics
        self.stats['total_warnings'] += 1
        self.stats['warnings_by_type'][category] = \
            self.stats['warnings_by_type'].get(category, 0) + 1

        if level == 'CRITICAL':
            self.stats['critical_warnings'] += 1

        return True

    def clear_old_warnings(self, max_age: float = 3.0):
        """
        Clear warnings older than max_age seconds

        Args:
            max_age: Maximum age of warnings in seconds
        """
        current_time = time.time()
        self.active_warnings = [
            w for w in self.active_warnings
            if (current_time - w['timestamp']) < max_age
        ]

    def get_active_warnings(self) -> List[Dict]:
        """Get list of active warnings"""
        return self.active_warnings

    def get_highest_priority_warning(self) -> Optional[Dict]:
        """Get the highest priority active warning"""
        if not self.active_warnings:
            return None

        return max(
            self.active_warnings,
            key=lambda w: self.WARNING_LEVELS[w['level']]['priority']
        )

    def draw_warnings(self, frame: np.ndarray,
                     warnings: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw warnings on frame

        Args:
            frame: Input frame
            warnings: Optional list of warning text to display
                     If None, uses active_warnings

        Returns:
            Frame with warnings drawn
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # Use provided warnings or active warnings
        if warnings is None:
            if not self.active_warnings:
                return output
            warning_texts = [w['text'] for w in self.active_warnings]
            levels = [w['level'] for w in self.active_warnings]
        else:
            warning_texts = warnings
            levels = ['HIGH'] * len(warnings)  # Default to HIGH

        if not warning_texts:
            return output

        # Draw alert banner
        alert_height = min(120, h // 3)

        # Get highest priority color
        highest_priority = max(
            levels,
            key=lambda lvl: self.WARNING_LEVELS[lvl]['priority']
        )
        alert_color = self.WARNING_LEVELS[highest_priority]['color']

        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, alert_height), alert_color, -1)
        cv2.addWeighted(overlay[:alert_height], 0.35,
                       output[:alert_height], 0.65, 0,
                       output[:alert_height])

        # Draw border
        cv2.rectangle(output, (0, 0), (w - 1, alert_height), alert_color, 3)

        # Draw warning text
        y_offset = 35
        for i, warning_text in enumerate(warning_texts[:3]):  # Max 3 warnings
            # Draw shadow for better readability
            cv2.putText(output, warning_text, (12, y_offset + 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
            # Draw main text
            cv2.putText(output, warning_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35

        # Draw warning indicator dots
        dot_size = 8
        for i in range(min(len(warning_texts), 5)):
            x_pos = w - 20 - (i * 15)
            cv2.circle(output, (x_pos, 15), dot_size, (255, 255, 255), -1)

        return output

    def draw_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw status bar with warning count

        Args:
            frame: Input frame

        Returns:
            Frame with status bar
        """
        output = frame.copy()
        h, w = frame.shape[:2]

        # Status bar at bottom
        bar_height = 30
        bar_y = h - bar_height

        # Create bar background
        overlay = output.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (50, 50, 50), -1)
        cv2.addWeighted(overlay[bar_y:], 0.6, output[bar_y:], 0.4, 0, output[bar_y:])

        # Warning count
        warning_count = len(self.active_warnings)
        count_color = (0, 255, 0) if warning_count == 0 else (0, 0, 255)

        status_text = f"Warnings: {warning_count}"
        cv2.putText(output, status_text, (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2)

        # Total warnings
        total_text = f"Total: {self.stats['total_warnings']}"
        cv2.putText(output, total_text, (200, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return output

    def get_statistics(self) -> Dict:
        """Get warning statistics"""
        return {
            'total_warnings': self.stats['total_warnings'],
            'warnings_by_type': self.stats['warnings_by_type'].copy(),
            'critical_warnings': self.stats['critical_warnings'],
            'active_warnings': len(self.active_warnings),
            'warning_history_size': len(self.warning_history)
        }

    def get_warning_report(self) -> str:
        """
        Generate text report of warnings

        Returns:
            Formatted warning report string
        """
        report = "=== DRIVER SAFETY WARNING REPORT ===\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += f"Total Warnings: {self.stats['total_warnings']}\n"
        report += f"Critical Warnings: {self.stats['critical_warnings']}\n"
        report += f"Active Warnings: {len(self.active_warnings)}\n\n"

        report += "Warnings by Type:\n"
        for wtype, count in sorted(self.stats['warnings_by_type'].items(),
                                   key=lambda x: x[1], reverse=True):
            report += f"  {wtype}: {count}\n"

        report += "\nRecent Warnings:\n"
        for warning in self.warning_history[-5:]:
            report += f"  [{warning['level']}] {warning['text']}\n"

        return report

    def reset(self):
        """Reset all warnings and statistics"""
        self.active_warnings = []
        self.warning_history = []
        self.last_warning_time = {}
        self.stats = {
            'total_warnings': 0,
            'warnings_by_type': {},
            'critical_warnings': 0
        }
