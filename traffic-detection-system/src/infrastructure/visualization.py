"""Visualization utilities for drawing detection results."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

from ..domain.entities import Detection, GestureDetection, PerformanceMetrics, GestureType
from ..domain.interfaces import IVisualization


logger = logging.getLogger(__name__)


class DetectionVisualizer(IVisualization):
    """Visualization for detection results."""

    # Color scheme (BGR format for OpenCV)
    COLORS = {
        "car": (255, 178, 50),      # Orange
        "truck": (255, 100, 50),     # Dark orange
        "bus": (255, 50, 150),       # Pink
        "motorcycle": (50, 255, 178), # Cyan
        "bicycle": (50, 178, 255),   # Light blue
        "person": (50, 255, 50),     # Green
        "pedestrian": (100, 255, 100), # Light green
        "traffic_light": (50, 50, 255), # Red
        "stop_sign": (0, 0, 255),    # Pure red
        "traffic_sign": (255, 255, 50), # Yellow
        "hand_gesture": (255, 50, 255), # Magenta
        "default": (128, 128, 128)   # Gray
    }

    GESTURE_COLORS = {
        GestureType.STOP: (0, 0, 255),      # Red
        GestureType.GO: (0, 255, 0),        # Green
        GestureType.TURN_LEFT: (255, 178, 0), # Orange
        GestureType.TURN_RIGHT: (0, 178, 255), # Blue
        GestureType.SLOW_DOWN: (255, 255, 0), # Yellow
        GestureType.UNKNOWN: (128, 128, 128)  # Gray
    }

    def __init__(
        self,
        font_scale: float = 0.5,
        thickness: int = 2,
        show_confidence: bool = True,
        show_labels: bool = True
    ):
        """
        Initialize visualizer.

        Args:
            font_scale: Font scale for text
            thickness: Line thickness
            show_confidence: Whether to show confidence scores
            show_labels: Whether to show class labels
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.show_confidence = show_confidence
        self.show_labels = show_labels
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Frame with visualizations
        """
        output = frame.copy()

        for detection in detections:
            if detection.bounding_box is None:
                continue

            # Get color for class
            color = self.COLORS.get(detection.class_name, self.COLORS["default"])

            # Draw bounding box
            x1 = int(detection.bounding_box.x1)
            y1 = int(detection.bounding_box.y1)
            x2 = int(detection.bounding_box.x2)
            y2 = int(detection.bounding_box.y2)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            # Prepare label
            if self.show_labels:
                label_parts = [detection.class_name]
                if self.show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                label = " ".join(label_parts)

                # Calculate label size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, self.thickness
                )

                # Draw label background
                label_y1 = max(y1 - label_height - 10, 0)
                label_y2 = y1
                cv2.rectangle(
                    output,
                    (x1, label_y1),
                    (x1 + label_width + 5, label_y2),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 5),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    self.thickness - 1,
                    cv2.LINE_AA
                )

        return output

    def draw_gestures(
        self,
        frame: np.ndarray,
        gestures: List[GestureDetection]
    ) -> np.ndarray:
        """
        Draw gesture detections on frame.

        Args:
            frame: Input frame
            gestures: List of gesture detections

        Returns:
            Frame with visualizations
        """
        output = frame.copy()

        for gesture in gestures:
            # Get color based on gesture type
            color = self.GESTURE_COLORS.get(
                gesture.gesture_type,
                self.GESTURE_COLORS[GestureType.UNKNOWN]
            )

            # Draw bounding box if available
            if gesture.bounding_box:
                x1 = int(gesture.bounding_box.x1)
                y1 = int(gesture.bounding_box.y1)
                x2 = int(gesture.bounding_box.x2)
                y2 = int(gesture.bounding_box.y2)

                cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

                # Draw gesture label
                if self.show_labels:
                    label = f"{gesture.gesture_type.value}"
                    if gesture.hand_side:
                        label += f" ({gesture.hand_side})"

                    # Calculate label position
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, self.font, self.font_scale, self.thickness
                    )

                    # Draw label background
                    label_y1 = max(y1 - label_height - 10, 0)
                    label_y2 = y1
                    cv2.rectangle(
                        output,
                        (x1, label_y1),
                        (x1 + label_width + 5, label_y2),
                        color,
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        output,
                        label,
                        (x1 + 2, y1 - 5),
                        self.font,
                        self.font_scale,
                        (255, 255, 255),
                        self.thickness - 1,
                        cv2.LINE_AA
                    )

            # Draw hand landmarks if available
            if gesture.landmarks:
                # Draw skeleton connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]

                # Draw connections
                for connection in connections:
                    if connection[0] < len(gesture.landmarks) and connection[1] < len(gesture.landmarks):
                        pt1 = gesture.landmarks[connection[0]]
                        pt2 = gesture.landmarks[connection[1]]
                        cv2.line(
                            output,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            color,
                            self.thickness - 1
                        )

                # Draw landmarks
                for landmark in gesture.landmarks:
                    cv2.circle(
                        output,
                        (int(landmark[0]), int(landmark[1])),
                        3,
                        color,
                        -1
                    )

        return output

    def draw_metrics(
        self,
        frame: np.ndarray,
        metrics: PerformanceMetrics
    ) -> np.ndarray:
        """
        Draw performance metrics on frame.

        Args:
            frame: Input frame
            metrics: Performance metrics

        Returns:
            Frame with metrics overlay
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Create semi-transparent overlay for metrics
        overlay = output.copy()

        # Define metrics text
        metrics_text = [
            f"FPS: {metrics.avg_fps:.1f}",
            f"Processing: {metrics.avg_processing_time_ms:.1f}ms",
            f"Detections: {metrics.total_detections}",
            f"Memory: {metrics.memory_usage_mb:.0f}MB",
            f"CPU: {metrics.cpu_usage_percent:.1f}%"
        ]

        if metrics.gpu_usage_percent is not None:
            metrics_text.append(f"GPU: {metrics.gpu_usage_percent:.1f}%")

        # Calculate background size
        max_text_width = 0
        total_height = 10
        for text in metrics_text:
            (text_width, text_height), _ = cv2.getTextSize(
                text, self.font, self.font_scale, 1
            )
            max_text_width = max(max_text_width, text_width)
            total_height += text_height + 5

        # Draw background rectangle
        cv2.rectangle(
            overlay,
            (10, 10),
            (max_text_width + 20, total_height + 10),
            (0, 0, 0),
            -1
        )

        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw metrics text
        y_position = 30
        for text in metrics_text:
            # Determine color based on metric
            if "FPS" in text:
                fps_value = metrics.avg_fps
                if fps_value >= 25:
                    color = (0, 255, 0)  # Green
                elif fps_value >= 15:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White

            cv2.putText(
                output,
                text,
                (15, y_position),
                self.font,
                self.font_scale,
                color,
                1,
                cv2.LINE_AA
            )
            y_position += 20

        return output

    def draw_traffic_info(
        self,
        frame: np.ndarray,
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw traffic-specific information overlay.

        Args:
            frame: Input frame
            info: Traffic information dictionary

        Returns:
            Frame with traffic info overlay
        """
        output = frame.copy()
        height, width = output.shape[:2]

        # Draw traffic statistics in bottom-left corner
        y_position = height - 100

        if "vehicle_count" in info:
            text = f"Vehicles: {info['vehicle_count']}"
            cv2.putText(
                output,
                text,
                (15, y_position),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_position += 25

        if "pedestrian_count" in info:
            text = f"Pedestrians: {info['pedestrian_count']}"
            cv2.putText(
                output,
                text,
                (15, y_position),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_position += 25

        if "active_gesture" in info:
            text = f"Gesture: {info['active_gesture']}"
            cv2.putText(
                output,
                text,
                (15, y_position),
                self.font,
                self.font_scale * 1.2,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        return output