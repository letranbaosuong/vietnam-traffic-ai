"""Unit tests for domain entities."""

import pytest
from datetime import datetime

from src.domain.entities import (
    BoundingBox,
    Detection,
    GestureDetection,
    Frame,
    DetectionResult,
    PerformanceMetrics,
    ObjectClass,
    GestureType,
    DetectionSource
)


class TestBoundingBox:
    """Test BoundingBox entity."""

    def test_bounding_box_creation(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 150

    def test_bounding_box_properties(self):
        """Test bounding box calculated properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=170)
        assert bbox.width == 100
        assert bbox.height == 150
        assert bbox.center == (60, 95)
        assert bbox.area == 15000

    def test_bounding_box_to_dict(self):
        """Test converting bounding box to dictionary."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=150)
        result = bbox.to_dict()
        assert result["x1"] == 10
        assert result["y1"] == 20
        assert result["x2"] == 100
        assert result["y2"] == 150
        assert result["width"] == 90
        assert result["height"] == 130


class TestDetection:
    """Test Detection entity."""

    def test_detection_creation(self, sample_detection):
        """Test creating a detection."""
        assert sample_detection.class_name == ObjectClass.CAR.value
        assert sample_detection.confidence == 0.85
        assert sample_detection.bounding_box is not None
        assert sample_detection.id is not None

    def test_detection_is_valid(self):
        """Test detection validation."""
        # Valid detection
        detection = Detection(
            class_name="car",
            confidence=0.9,
            bounding_box=BoundingBox(x1=0, y1=0, x2=100, y2=100)
        )
        assert detection.is_valid

        # Invalid - no bounding box
        detection = Detection(class_name="car", confidence=0.9)
        assert not detection.is_valid

        # Invalid - confidence out of range
        detection = Detection(
            class_name="car",
            confidence=1.5,
            bounding_box=BoundingBox(x1=0, y1=0, x2=100, y2=100)
        )
        assert not detection.is_valid

    def test_detection_to_dict(self, sample_detection):
        """Test converting detection to dictionary."""
        result = sample_detection.to_dict()
        assert result["class_name"] == ObjectClass.CAR.value
        assert result["confidence"] == 0.85
        assert result["bounding_box"] is not None
        assert "timestamp" in result
        assert "id" in result


class TestGestureDetection:
    """Test GestureDetection entity."""

    def test_gesture_detection_creation(self, sample_gesture_detection):
        """Test creating a gesture detection."""
        assert sample_gesture_detection.gesture_type == GestureType.STOP
        assert sample_gesture_detection.hand_side == "right"
        assert len(sample_gesture_detection.landmarks) == 3

    def test_gesture_detection_to_dict(self, sample_gesture_detection):
        """Test converting gesture detection to dictionary."""
        result = sample_gesture_detection.to_dict()
        assert result["gesture_type"] == GestureType.STOP.value
        assert result["hand_side"] == "right"
        assert result["landmarks"] == [(175, 175), (200, 200), (225, 225)]
        assert "confidence" in result
        assert "bounding_box" in result


class TestFrame:
    """Test Frame entity."""

    def test_frame_creation(self, sample_frame):
        """Test creating a frame."""
        assert sample_frame.width == 640
        assert sample_frame.height == 480
        assert sample_frame.frame_number == 1
        assert sample_frame.source == DetectionSource.CAMERA
        assert sample_frame.data is not None

    def test_frame_properties(self, sample_frame):
        """Test frame properties."""
        assert sample_frame.shape == (480, 640, 3)
        assert sample_frame.is_valid

    def test_frame_invalid(self):
        """Test invalid frame."""
        frame = Frame()
        assert not frame.is_valid

        frame = Frame(data=None, width=640, height=480)
        assert not frame.is_valid


class TestDetectionResult:
    """Test DetectionResult entity."""

    def test_detection_result_creation(self, sample_detections, sample_gesture_detection):
        """Test creating detection result."""
        result = DetectionResult(
            frame_id="frame_123",
            frame_number=5,
            object_detections=sample_detections,
            gesture_detections=[sample_gesture_detection],
            processing_time_ms=25.5
        )

        assert result.frame_id == "frame_123"
        assert result.frame_number == 5
        assert len(result.object_detections) == 3
        assert len(result.gesture_detections) == 1
        assert result.processing_time_ms == 25.5

    def test_detection_result_properties(self, sample_detections):
        """Test detection result properties."""
        result = DetectionResult(
            frame_id="test",
            frame_number=1,
            object_detections=sample_detections,
            processing_time_ms=40.0
        )

        assert result.total_detections == 3
        assert result.fps == pytest.approx(25.0)

    def test_detection_result_to_dict(self, sample_detections):
        """Test converting detection result to dictionary."""
        result = DetectionResult(
            frame_id="test",
            frame_number=1,
            object_detections=sample_detections,
            processing_time_ms=20.0
        )

        data = result.to_dict()
        assert data["frame_id"] == "test"
        assert data["frame_number"] == 1
        assert len(data["object_detections"]) == 3
        assert data["fps"] == pytest.approx(50.0)
        assert data["total_detections"] == 3


class TestPerformanceMetrics:
    """Test PerformanceMetrics entity."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            avg_fps=30.0,
            min_fps=25.0,
            max_fps=35.0,
            avg_processing_time_ms=33.3,
            total_frames_processed=1000,
            total_detections=5000,
            memory_usage_mb=512.5,
            cpu_usage_percent=45.2,
            gpu_usage_percent=60.0
        )

        assert metrics.avg_fps == 30.0
        assert metrics.min_fps == 25.0
        assert metrics.max_fps == 35.0
        assert metrics.total_frames_processed == 1000
        assert metrics.total_detections == 5000

    def test_performance_metrics_to_dict(self):
        """Test converting performance metrics to dictionary."""
        metrics = PerformanceMetrics(
            avg_fps=30.0,
            total_frames_processed=100
        )

        data = metrics.to_dict()
        assert data["avg_fps"] == 30.0
        assert data["total_frames_processed"] == 100
        assert "timestamp" in data
        assert "memory_usage_mb" in data


class TestEnums:
    """Test enumeration types."""

    def test_object_class_enum(self):
        """Test ObjectClass enum."""
        assert ObjectClass.CAR.value == "car"
        assert ObjectClass.PERSON.value == "person"
        assert ObjectClass.TRAFFIC_LIGHT.value == "traffic_light"

    def test_gesture_type_enum(self):
        """Test GestureType enum."""
        assert GestureType.STOP.value == "stop"
        assert GestureType.GO.value == "go"
        assert GestureType.TURN_LEFT.value == "turn_left"

    def test_detection_source_enum(self):
        """Test DetectionSource enum."""
        assert DetectionSource.CAMERA.value == "camera"
        assert DetectionSource.VIDEO_FILE.value == "video_file"
        assert DetectionSource.STREAM.value == "stream"