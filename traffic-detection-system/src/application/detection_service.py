"""Main detection service that orchestrates object and gesture detection."""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional, List, Dict, Any
from datetime import datetime

from ..domain.entities import (
    Frame,
    DetectionResult,
    Detection,
    GestureDetection
)
from ..domain.interfaces import (
    IDetectionService,
    IObjectDetector,
    IGestureDetector,
    IFrameSource,
    IPerformanceMonitor
)


logger = logging.getLogger(__name__)


class DetectionService(IDetectionService):
    """Main service for coordinating detection operations."""

    def __init__(
        self,
        object_detector: Optional[IObjectDetector] = None,
        gesture_detector: Optional[IGestureDetector] = None,
        performance_monitor: Optional[IPerformanceMonitor] = None,
        frame_skip: int = 0,
        parallel_processing: bool = True
    ):
        """
        Initialize detection service.

        Args:
            object_detector: Object detection implementation
            gesture_detector: Gesture detection implementation
            performance_monitor: Performance monitoring implementation
            frame_skip: Number of frames to skip (for performance)
            parallel_processing: Run detectors in parallel
        """
        self.object_detector = object_detector
        self.gesture_detector = gesture_detector
        self.performance_monitor = performance_monitor
        self.frame_skip = frame_skip
        self.parallel_processing = parallel_processing
        self.frame_counter = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the detection service and its components."""
        if self._initialized:
            return

        try:
            # Initialize detectors in parallel
            init_tasks = []

            if self.object_detector:
                init_tasks.append(self.object_detector.initialize())

            if self.gesture_detector:
                init_tasks.append(self.gesture_detector.initialize())

            if init_tasks:
                await asyncio.gather(*init_tasks)

            self._initialized = True
            logger.info("Detection service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize detection service: {e}")
            raise

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
        if not self._initialized:
            await self.initialize()

        # Check if we should skip this frame
        if self.frame_skip > 0:
            self.frame_counter += 1
            if self.frame_counter % (self.frame_skip + 1) != 0:
                # Return empty result for skipped frames
                return DetectionResult(
                    frame_id=frame.id,
                    frame_number=frame.frame_number,
                    metadata={"skipped": True}
                )

        start_time = time.time()
        object_detections: List[Detection] = []
        gesture_detections: List[GestureDetection] = []

        try:
            if self.parallel_processing and detect_objects and detect_gestures:
                # Run both detectors in parallel
                tasks = []

                if detect_objects and self.object_detector:
                    tasks.append(self.object_detector.detect(frame))

                if detect_gestures and self.gesture_detector:
                    tasks.append(self.gesture_detector.detect(frame))

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    idx = 0
                    if detect_objects and self.object_detector:
                        if not isinstance(results[idx], Exception):
                            object_detections = results[idx]
                        else:
                            logger.error(f"Object detection error: {results[idx]}")
                        idx += 1

                    if detect_gestures and self.gesture_detector:
                        if not isinstance(results[idx], Exception):
                            gesture_detections = results[idx]
                        else:
                            logger.error(f"Gesture detection error: {results[idx]}")

            else:
                # Run detectors sequentially
                if detect_objects and self.object_detector:
                    try:
                        object_detections = await self.object_detector.detect(frame)
                    except Exception as e:
                        logger.error(f"Object detection error: {e}")

                if detect_gestures and self.gesture_detector:
                    try:
                        gesture_detections = await self.gesture_detector.detect(frame)
                    except Exception as e:
                        logger.error(f"Gesture detection error: {e}")

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Record performance metrics
        if self.performance_monitor:
            self.performance_monitor.record_frame_processing(processing_time_ms)
            self.performance_monitor.record_detection(
                len(object_detections) + len(gesture_detections)
            )

        # Create and return result
        result = DetectionResult(
            frame_id=frame.id,
            frame_number=frame.frame_number,
            object_detections=object_detections,
            gesture_detections=gesture_detections,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(),
            metadata={
                "frame_skip": self.frame_skip,
                "parallel_processing": self.parallel_processing
            }
        )

        return result

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
        if not self._initialized:
            await self.initialize()

        try:
            # Start the frame source
            await source.start()

            # Process frames
            async for frame in source.get_frames():
                result = await self.process_frame(
                    frame,
                    detect_objects=detect_objects,
                    detect_gestures=detect_gestures
                )
                yield result

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            raise

        finally:
            # Stop the frame source
            await source.stop()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        cleanup_tasks = []

        if self.object_detector:
            cleanup_tasks.append(self.object_detector.cleanup())

        if self.gesture_detector:
            cleanup_tasks.append(self.gesture_detector.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._initialized = False
        logger.info("Detection service cleaned up")

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update service settings.

        Args:
            settings: Dictionary of settings to update
        """
        if "frame_skip" in settings:
            self.frame_skip = settings["frame_skip"]
            logger.info(f"Updated frame_skip to {self.frame_skip}")

        if "parallel_processing" in settings:
            self.parallel_processing = settings["parallel_processing"]
            logger.info(f"Updated parallel_processing to {self.parallel_processing}")

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "initialized": self._initialized,
            "object_detector": self.object_detector is not None,
            "gesture_detector": self.gesture_detector is not None,
            "performance_monitor": self.performance_monitor is not None,
            "frame_skip": self.frame_skip,
            "parallel_processing": self.parallel_processing,
            "frames_processed": self.frame_counter
        }