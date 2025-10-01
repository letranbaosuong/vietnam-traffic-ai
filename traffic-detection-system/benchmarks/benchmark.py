#!/usr/bin/env python3
"""Performance benchmarking tool for the traffic detection system."""

import asyncio
import time
import argparse
import json
import platform
import psutil
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.yolo_detector import YOLODetector
from src.application.gesture_detector import TrafficGestureDetector
from src.application.detection_service import DetectionService
from src.infrastructure.performance_monitor import PerformanceMonitor
from src.infrastructure.camera_source import CameraSource
from src.infrastructure.video_source import VideoFileSource
from src.domain.entities import Frame, DetectionSource


class BenchmarkRunner:
    """Performance benchmark runner."""

    def __init__(self, output_dir: Path = None):
        """Initialize benchmark runner."""
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    async def benchmark_yolo_detector(
        self,
        num_frames: int = 100,
        image_sizes: List[tuple] = [(640, 480), (1280, 720), (1920, 1080)]
    ) -> Dict[str, Any]:
        """
        Benchmark YOLO detector performance.

        Args:
            num_frames: Number of frames to process
            image_sizes: List of image sizes to test

        Returns:
            Benchmark results
        """
        print("\n=== YOLO Detector Benchmark ===")
        results = {}

        for model_name in ["yolov8n.pt", "yolov8s.pt"]:
            print(f"\nTesting model: {model_name}")
            detector = YOLODetector(model_name=model_name)
            await detector.initialize()

            model_results = {}
            for width, height in image_sizes:
                print(f"  Resolution: {width}x{height}")

                # Create test frames
                frames = []
                for i in range(num_frames):
                    # Generate random image with some objects
                    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                    # Add some rectangles to simulate objects
                    for _ in range(5):
                        x1 = np.random.randint(0, width - 100)
                        y1 = np.random.randint(0, height - 100)
                        cv2.rectangle(image, (x1, y1), (x1 + 100, y1 + 100), (255, 255, 255), -1)

                    frames.append(Frame(
                        data=image,
                        width=width,
                        height=height,
                        frame_number=i
                    ))

                # Benchmark
                start_time = time.time()
                total_detections = 0

                for frame in frames:
                    detections = await detector.detect(frame)
                    total_detections += len(detections)

                elapsed_time = time.time() - start_time
                fps = num_frames / elapsed_time

                model_results[f"{width}x{height}"] = {
                    "fps": fps,
                    "avg_time_ms": (elapsed_time / num_frames) * 1000,
                    "total_detections": total_detections
                }

                print(f"    FPS: {fps:.2f}, Avg time: {(elapsed_time / num_frames) * 1000:.2f}ms")

            await detector.cleanup()
            results[model_name] = model_results

        return results

    async def benchmark_gesture_detector(
        self,
        num_frames: int = 100,
        image_sizes: List[tuple] = [(640, 480), (1280, 720)]
    ) -> Dict[str, Any]:
        """
        Benchmark gesture detector performance.

        Args:
            num_frames: Number of frames to process
            image_sizes: List of image sizes to test

        Returns:
            Benchmark results
        """
        print("\n=== Gesture Detector Benchmark ===")
        results = {}

        detector = TrafficGestureDetector()
        await detector.initialize()

        for width, height in image_sizes:
            print(f"  Resolution: {width}x{height}")

            # Create test frames
            frames = []
            for i in range(num_frames):
                # Generate image with hand-like shapes
                image = np.zeros((height, width, 3), dtype=np.uint8)
                # Add white circles to simulate hand landmarks
                center_x = width // 2
                center_y = height // 2
                for j in range(21):  # 21 hand landmarks
                    x = center_x + np.random.randint(-50, 50)
                    y = center_y + np.random.randint(-50, 50)
                    cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

                frames.append(Frame(
                    data=image,
                    width=width,
                    height=height,
                    frame_number=i
                ))

            # Benchmark
            start_time = time.time()
            total_detections = 0

            for frame in frames:
                detections = await detector.detect(frame)
                total_detections += len(detections)

            elapsed_time = time.time() - start_time
            fps = num_frames / elapsed_time

            results[f"{width}x{height}"] = {
                "fps": fps,
                "avg_time_ms": (elapsed_time / num_frames) * 1000,
                "total_detections": total_detections
            }

            print(f"    FPS: {fps:.2f}, Avg time: {(elapsed_time / num_frames) * 1000:.2f}ms")

        await detector.cleanup()
        return results

    async def benchmark_full_pipeline(
        self,
        num_frames: int = 100,
        resolution: tuple = (640, 480),
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Benchmark full detection pipeline.

        Args:
            num_frames: Number of frames to process
            resolution: Image resolution
            parallel: Whether to use parallel processing

        Returns:
            Benchmark results
        """
        print(f"\n=== Full Pipeline Benchmark (parallel={parallel}) ===")

        # Initialize components
        object_detector = YOLODetector(model_name="yolov8n.pt")
        gesture_detector = TrafficGestureDetector()
        performance_monitor = PerformanceMonitor()

        detection_service = DetectionService(
            object_detector=object_detector,
            gesture_detector=gesture_detector,
            performance_monitor=performance_monitor,
            parallel_processing=parallel
        )

        await detection_service.initialize()

        width, height = resolution
        print(f"  Resolution: {width}x{height}")

        # Create test frames
        frames = []
        for i in range(num_frames):
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add some features
            for _ in range(3):
                x1 = np.random.randint(0, width - 100)
                y1 = np.random.randint(0, height - 100)
                cv2.rectangle(image, (x1, y1), (x1 + 100, y1 + 100), (255, 255, 255), -1)

            frames.append(Frame(
                data=image,
                width=width,
                height=height,
                frame_number=i,
                source=DetectionSource.CAMERA
            ))

        # Benchmark
        start_time = time.time()
        total_object_detections = 0
        total_gesture_detections = 0

        for frame in frames:
            result = await detection_service.process_frame(frame)
            total_object_detections += len(result.object_detections)
            total_gesture_detections += len(result.gesture_detections)

        elapsed_time = time.time() - start_time
        fps = num_frames / elapsed_time

        # Get performance metrics
        metrics = performance_monitor.get_metrics()

        await detection_service.cleanup()

        return {
            "parallel": parallel,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "avg_time_ms": (elapsed_time / num_frames) * 1000,
            "total_object_detections": total_object_detections,
            "total_gesture_detections": total_gesture_detections,
            "metrics": metrics.to_dict()
        }

    async def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        print("\n=== Memory Usage Benchmark ===")

        process = psutil.Process()
        results = {}

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        results["baseline_mb"] = baseline_memory

        # After YOLO initialization
        yolo_detector = YOLODetector()
        await yolo_detector.initialize()
        yolo_memory = process.memory_info().rss / 1024 / 1024
        results["yolo_loaded_mb"] = yolo_memory
        results["yolo_delta_mb"] = yolo_memory - baseline_memory

        # After gesture detector initialization
        gesture_detector = TrafficGestureDetector()
        await gesture_detector.initialize()
        full_memory = process.memory_info().rss / 1024 / 1024
        results["full_loaded_mb"] = full_memory
        results["gesture_delta_mb"] = full_memory - yolo_memory

        # Cleanup
        await yolo_detector.cleanup()
        await gesture_detector.cleanup()

        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  YOLO loaded: {yolo_memory:.2f} MB (+{results['yolo_delta_mb']:.2f} MB)")
        print(f"  Full loaded: {full_memory:.2f} MB (+{results['gesture_delta_mb']:.2f} MB)")

        return results

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Starting Traffic Detection System Benchmark")
        print("=" * 50)

        results = {
            "system_info": self.get_system_info(),
            "benchmarks": {}
        }

        # Run benchmarks
        results["benchmarks"]["yolo"] = await self.benchmark_yolo_detector(
            num_frames=50,
            image_sizes=[(640, 480), (1280, 720)]
        )

        results["benchmarks"]["gesture"] = await self.benchmark_gesture_detector(
            num_frames=50,
            image_sizes=[(640, 480)]
        )

        results["benchmarks"]["pipeline_parallel"] = await self.benchmark_full_pipeline(
            num_frames=50,
            parallel=True
        )

        results["benchmarks"]["pipeline_sequential"] = await self.benchmark_full_pipeline(
            num_frames=50,
            parallel=False
        )

        results["benchmarks"]["memory"] = await self.benchmark_memory_usage()

        # Save results
        output_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return results

    def generate_report(self, results: Dict[str, Any]):
        """Generate benchmark report with visualizations."""
        print("\nGenerating report...")

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Traffic Detection System Performance Benchmark", fontsize=16)

        # YOLO FPS comparison
        ax = axes[0, 0]
        if "yolo" in results["benchmarks"]:
            models = []
            fps_640 = []
            fps_1280 = []

            for model_name, model_results in results["benchmarks"]["yolo"].items():
                models.append(model_name.replace(".pt", ""))
                fps_640.append(model_results.get("640x480", {}).get("fps", 0))
                fps_1280.append(model_results.get("1280x720", {}).get("fps", 0))

            x = np.arange(len(models))
            width = 0.35

            ax.bar(x - width/2, fps_640, width, label="640x480")
            ax.bar(x + width/2, fps_1280, width, label="1280x720")
            ax.set_xlabel("Model")
            ax.set_ylabel("FPS")
            ax.set_title("YOLO Model Performance")
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()

        # Pipeline comparison
        ax = axes[0, 1]
        if "pipeline_parallel" in results["benchmarks"] and "pipeline_sequential" in results["benchmarks"]:
            parallel_fps = results["benchmarks"]["pipeline_parallel"]["fps"]
            sequential_fps = results["benchmarks"]["pipeline_sequential"]["fps"]

            ax.bar(["Parallel", "Sequential"], [parallel_fps, sequential_fps])
            ax.set_ylabel("FPS")
            ax.set_title("Pipeline Processing Comparison")

        # Memory usage
        ax = axes[1, 0]
        if "memory" in results["benchmarks"]:
            memory_data = results["benchmarks"]["memory"]
            components = ["Baseline", "YOLO", "Full"]
            memory_values = [
                memory_data["baseline_mb"],
                memory_data["yolo_loaded_mb"],
                memory_data["full_loaded_mb"]
            ]

            ax.bar(components, memory_values)
            ax.set_ylabel("Memory (MB)")
            ax.set_title("Memory Usage")

        # System info
        ax = axes[1, 1]
        ax.axis("off")
        system_info = results["system_info"]
        info_text = f"""System Information:
Platform: {system_info['platform'][:30]}
CPU Cores: {system_info['cpu_count']}
Memory: {system_info['memory_gb']:.2f} GB
Python: {system_info['python_version']}"""
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="center")

        plt.tight_layout()
        report_path = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(report_path, dpi=150)
        print(f"Report saved to: {report_path}")
        plt.show()


async def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="Traffic Detection System Benchmark")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")

    args = parser.parse_args()

    runner = BenchmarkRunner(Path(args.output))

    if args.quick:
        # Quick benchmark
        print("Running quick benchmark...")
        result = await runner.benchmark_full_pipeline(num_frames=args.frames)
        print("\nQuick Benchmark Results:")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Avg processing time: {result['avg_time_ms']:.2f}ms")
    else:
        # Full benchmark
        results = await runner.run_full_benchmark()
        runner.generate_report(results)


if __name__ == "__main__":
    asyncio.run(main())