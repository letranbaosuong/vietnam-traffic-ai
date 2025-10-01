"""Core domain entities for traffic detection system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import uuid


class ObjectClass(Enum):
    """Enumeration of detectable object classes."""

    # Vehicles
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"

    # People
    PERSON = "person"
    PEDESTRIAN = "pedestrian"

    # Traffic elements
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"
    TRAFFIC_SIGN = "traffic_sign"

    # Unknown
    UNKNOWN = "unknown"


class GestureType(Enum):
    """Enumeration of traffic gesture types."""

    STOP = "stop"
    GO = "go"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SLOW_DOWN = "slow_down"
    SPEED_UP = "speed_up"
    UNKNOWN = "unknown"


class DetectionSource(Enum):
    """Source of detection input."""

    CAMERA = "camera"
    VIDEO_FILE = "video_file"
    IMAGE_FILE = "image_file"
    STREAM = "stream"


@dataclass
class BoundingBox:
    """Represents a bounding box for detected objects."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return abs(self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Calculate area of bounding box."""
        return self.width * self.height

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height
        }


@dataclass
class Detection:
    """Represents a single detection result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    class_name: str = ""
    confidence: float = 0.0
    bounding_box: Optional[BoundingBox] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if detection is valid."""
        return (
            self.confidence > 0
            and self.confidence <= 1.0
            and self.bounding_box is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class GestureDetection(Detection):
    """Represents a gesture detection result."""

    gesture_type: GestureType = GestureType.UNKNOWN
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    hand_side: Optional[str] = None  # "left" or "right"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "gesture_type": self.gesture_type.value,
            "landmarks": self.landmarks,
            "hand_side": self.hand_side
        })
        return result


@dataclass
class Frame:
    """Represents a single frame of video/image data."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Any = None  # numpy array
    width: int = 0
    height: int = 0
    channels: int = 3
    frame_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    source: DetectionSource = DetectionSource.CAMERA
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get frame shape."""
        return (self.height, self.width, self.channels)

    @property
    def is_valid(self) -> bool:
        """Check if frame is valid."""
        return (
            self.data is not None
            and self.width > 0
            and self.height > 0
        )


@dataclass
class DetectionResult:
    """Aggregated detection results for a frame."""

    frame_id: str
    frame_number: int
    object_detections: List[Detection] = field(default_factory=list)
    gesture_detections: List[GestureDetection] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_detections(self) -> int:
        """Get total number of detections."""
        return len(self.object_detections) + len(self.gesture_detections)

    @property
    def fps(self) -> float:
        """Calculate FPS based on processing time."""
        if self.processing_time_ms > 0:
            return 1000.0 / self.processing_time_ms
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "frame_id": self.frame_id,
            "frame_number": self.frame_number,
            "object_detections": [d.to_dict() for d in self.object_detections],
            "gesture_detections": [g.to_dict() for g in self.gesture_detections],
            "processing_time_ms": self.processing_time_ms,
            "fps": self.fps,
            "timestamp": self.timestamp.isoformat(),
            "total_detections": self.total_detections,
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """System performance metrics."""

    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    avg_processing_time_ms: float = 0.0
    total_frames_processed: int = 0
    total_detections: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "avg_fps": self.avg_fps,
            "min_fps": self.min_fps,
            "max_fps": self.max_fps,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_usage_percent": self.gpu_usage_percent,
            "timestamp": self.timestamp.isoformat()
        }