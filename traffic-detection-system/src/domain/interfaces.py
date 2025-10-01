"""Domain interfaces and abstract base classes."""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator, Dict, Any
import numpy as np

from .entities import (
    Detection,
    GestureDetection,
    Frame,
    DetectionResult,
    PerformanceMetrics,
    DetectionSource
)


class IObjectDetector(ABC):
    """Interface for object detection."""

    @abstractmethod
    async def detect(self, frame: Frame) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input frame for detection

        Returns:
            List of detected objects
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detector and load models."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """Get list of supported object classes."""
        pass


class IGestureDetector(ABC):
    """Interface for gesture detection."""

    @abstractmethod
    async def detect(self, frame: Frame) -> List[GestureDetection]:
        """
        Detect gestures in a frame.

        Args:
            frame: Input frame for detection

        Returns:
            List of detected gestures
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the gesture detector."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class IFrameSource(ABC):
    """Interface for frame sources (camera, video, etc.)."""

    @abstractmethod
    async def start(self) -> None:
        """Start the frame source."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the frame source."""
        pass

    @abstractmethod
    async def get_frame(self) -> Optional[Frame]:
        """
        Get a single frame.

        Returns:
            Frame object or None if no frame available
        """
        pass

    @abstractmethod
    def get_frames(self) -> AsyncIterator[Frame]:
        """
        Get frames as an async iterator.

        Yields:
            Frame objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if frame source is available."""
        pass

    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """Get frame source properties (resolution, fps, etc.)."""
        pass


class IDetectionService(ABC):
    """Interface for the main detection service."""

    @abstractmethod
    async def process_frame(
        self,
        frame: Frame,
        detect_objects: bool = True,
        detect_gestures: bool = True
    ) -> DetectionResult:
        """
        Process a single frame for detections.

        Args:
            frame: Input frame
            detect_objects: Whether to detect objects
            detect_gestures: Whether to detect gestures

        Returns:
            Detection results
        """
        pass

    @abstractmethod
    async def process_stream(
        self,
        source: IFrameSource,
        detect_objects: bool = True,
        detect_gestures: bool = True
    ) -> AsyncIterator[DetectionResult]:
        """
        Process a stream of frames.

        Args:
            source: Frame source
            detect_objects: Whether to detect objects
            detect_gestures: Whether to detect gestures

        Yields:
            Detection results for each frame
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the detection service."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class IPerformanceMonitor(ABC):
    """Interface for performance monitoring."""

    @abstractmethod
    def record_frame_processing(self, processing_time_ms: float) -> None:
        """Record frame processing time."""
        pass

    @abstractmethod
    def record_detection(self, count: int) -> None:
        """Record detection count."""
        pass

    @abstractmethod
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset performance metrics."""
        pass


class IModelCache(ABC):
    """Interface for model caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached model.

        Args:
            key: Cache key

        Returns:
            Cached model or None
        """
        pass

    @abstractmethod
    async def set(self, key: str, model: Any, ttl: Optional[int] = None) -> None:
        """
        Cache a model.

        Args:
            key: Cache key
            model: Model to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cached model."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached models."""
        pass


class IVisualization(ABC):
    """Interface for result visualization."""

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass