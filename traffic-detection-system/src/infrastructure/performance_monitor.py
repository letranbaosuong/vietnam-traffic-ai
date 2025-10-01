"""Performance monitoring implementation."""

import time
import psutil
import logging
from typing import List, Optional
from collections import deque
from datetime import datetime
import numpy as np

from ..domain.entities import PerformanceMetrics
from ..domain.interfaces import IPerformanceMonitor


logger = logging.getLogger(__name__)


class PerformanceMonitor(IPerformanceMonitor):
    """Implementation of performance monitoring."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of the sliding window for metrics
        """
        self.window_size = window_size
        self.processing_times: deque = deque(maxlen=window_size)
        self.detection_counts: deque = deque(maxlen=window_size)
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        self.process = psutil.Process()

    def record_frame_processing(self, processing_time_ms: float) -> None:
        """
        Record frame processing time.

        Args:
            processing_time_ms: Processing time in milliseconds
        """
        self.processing_times.append(processing_time_ms)
        self.total_frames += 1

    def record_detection(self, count: int) -> None:
        """
        Record detection count.

        Args:
            count: Number of detections
        """
        self.detection_counts.append(count)
        self.total_detections += count

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.

        Returns:
            Performance metrics
        """
        # Calculate FPS metrics
        if self.processing_times:
            processing_times_array = np.array(self.processing_times)
            fps_array = 1000.0 / processing_times_array
            avg_fps = float(np.mean(fps_array))
            min_fps = float(np.min(fps_array))
            max_fps = float(np.max(fps_array))
            avg_processing_time = float(np.mean(processing_times_array))
        else:
            avg_fps = min_fps = max_fps = avg_processing_time = 0.0

        # Get system metrics
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        cpu_usage_percent = self.process.cpu_percent()

        # Try to get GPU usage (if available)
        gpu_usage_percent = self._get_gpu_usage()

        return PerformanceMetrics(
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            avg_processing_time_ms=avg_processing_time,
            total_frames_processed=self.total_frames,
            total_detections=self.total_detections,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            gpu_usage_percent=gpu_usage_percent,
            timestamp=datetime.now()
        )

    def reset(self) -> None:
        """Reset performance metrics."""
        self.processing_times.clear()
        self.detection_counts.clear()
        self.total_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        logger.info("Performance metrics reset")

    def _get_gpu_usage(self) -> Optional[float]:
        """
        Get GPU usage if available.

        Returns:
            GPU usage percentage or None
        """
        try:
            import torch
            if torch.cuda.is_available():
                # Get CUDA memory usage
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                if memory_reserved > 0:
                    return (memory_allocated / memory_reserved) * 100
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not get GPU usage: {e}")

        return None

    def get_summary(self) -> str:
        """
        Get a summary of performance metrics.

        Returns:
            Summary string
        """
        metrics = self.get_metrics()
        runtime = time.time() - self.start_time

        summary = [
            "=== Performance Summary ===",
            f"Runtime: {runtime:.2f} seconds",
            f"Total frames: {metrics.total_frames_processed}",
            f"Total detections: {metrics.total_detections}",
            f"Average FPS: {metrics.avg_fps:.2f}",
            f"FPS range: {metrics.min_fps:.2f} - {metrics.max_fps:.2f}",
            f"Avg processing time: {metrics.avg_processing_time_ms:.2f} ms",
            f"Memory usage: {metrics.memory_usage_mb:.2f} MB",
            f"CPU usage: {metrics.cpu_usage_percent:.1f}%",
        ]

        if metrics.gpu_usage_percent is not None:
            summary.append(f"GPU usage: {metrics.gpu_usage_percent:.1f}%")

        return "\n".join(summary)