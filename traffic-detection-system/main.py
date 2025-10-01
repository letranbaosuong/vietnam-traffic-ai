#!/usr/bin/env python3
"""Main entry point for the Traffic Detection System."""

import asyncio
import logging
import signal
import sys
from pathlib import Path
import uvicorn
import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings, get_settings
from src.presentation.api import app


# Setup logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *(
                [logging.FileHandler(settings.log_file)]
                if settings.log_file
                else []
            )
        ]
    )


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle shutdown signals."""
    logging.info("Received shutdown signal, closing...")
    sys.exit(0)


@click.group()
def cli():
    """Traffic Detection System CLI."""
    pass


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to"
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to bind to"
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development"
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes"
)
@click.option(
    "--env",
    default="development",
    type=click.Choice(["development", "production"]),
    help="Environment to run in"
)
def serve(host: str, port: int, reload: bool, workers: int, env: str):
    """Start the API server."""
    # Update settings based on environment
    global settings
    settings = get_settings(env)

    setup_logging(settings.log_level)

    logging.info(f"Starting Traffic Detection System API Server")
    logging.info(f"Platform: {settings.platform_name}")
    logging.info(f"Is Raspberry Pi: {settings.is_raspberry_pi}")
    logging.info(f"Environment: {env}")
    logging.info(f"Host: {host}:{port}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run server
    uvicorn.run(
        "src.presentation.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=settings.log_level.lower()
    )


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output video path"
)
@click.option(
    "--show",
    is_flag=True,
    help="Show live preview"
)
@click.option(
    "--no-objects",
    is_flag=True,
    help="Disable object detection"
)
@click.option(
    "--no-gestures",
    is_flag=True,
    help="Disable gesture detection"
)
def process_video(video_path: str, output: str, show: bool, no_objects: bool, no_gestures: bool):
    """Process a video file."""
    import cv2
    from src.application.yolo_detector import YOLODetector
    from src.application.gesture_detector import TrafficGestureDetector
    from src.application.detection_service import DetectionService
    from src.infrastructure.video_source import VideoFileSource
    from src.infrastructure.performance_monitor import PerformanceMonitor
    from src.infrastructure.visualization import DetectionVisualizer

    setup_logging(settings.log_level)

    async def run():
        # Initialize components
        object_detector = None if no_objects else YOLODetector(
            model_name=settings.yolo_model,
            confidence_threshold=settings.yolo_confidence
        )

        gesture_detector = None if no_gestures else TrafficGestureDetector(
            min_detection_confidence=settings.gesture_confidence
        )

        performance_monitor = PerformanceMonitor()

        detection_service = DetectionService(
            object_detector=object_detector,
            gesture_detector=gesture_detector,
            performance_monitor=performance_monitor
        )

        visualizer = DetectionVisualizer()

        # Initialize service
        await detection_service.initialize()

        # Setup video source
        video_source = VideoFileSource(Path(video_path))
        await video_source.start()

        # Setup output writer if needed
        output_writer = None
        if output:
            props = video_source.get_properties()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(
                output,
                fourcc,
                video_source.original_fps,
                (video_source.width, video_source.height)
            )

        logging.info(f"Processing video: {video_path}")

        # Process video
        try:
            frame_count = 0
            async for result in detection_service.process_stream(
                video_source,
                detect_objects=not no_objects,
                detect_gestures=not no_gestures
            ):
                frame_count += 1

                # Get the actual frame for visualization
                frame = await video_source.get_frame()
                if frame and frame.data is not None:
                    # Visualize
                    output_frame = frame.data.copy()
                    output_frame = visualizer.draw_detections(output_frame, result.object_detections)
                    output_frame = visualizer.draw_gestures(output_frame, result.gesture_detections)

                    # Add metrics overlay
                    metrics = performance_monitor.get_metrics()
                    output_frame = visualizer.draw_metrics(output_frame, metrics)

                    # Show preview if requested
                    if show:
                        cv2.imshow("Traffic Detection", output_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    # Write to output if specified
                    if output_writer:
                        output_writer.write(output_frame)

                # Log progress
                if frame_count % 30 == 0:
                    logging.info(f"Processed {frame_count} frames, "
                               f"FPS: {result.fps:.2f}, "
                               f"Detections: {result.total_detections}")

        finally:
            # Cleanup
            if output_writer:
                output_writer.release()
            if show:
                cv2.destroyAllWindows()
            await video_source.stop()
            await detection_service.cleanup()

            # Print summary
            print("\n" + performance_monitor.get_summary())

    asyncio.run(run())


@cli.command()
@click.option(
    "--camera",
    "-c",
    default=0,
    type=int,
    help="Camera index"
)
@click.option(
    "--pi-camera",
    is_flag=True,
    help="Use Raspberry Pi Camera"
)
@click.option(
    "--no-objects",
    is_flag=True,
    help="Disable object detection"
)
@click.option(
    "--no-gestures",
    is_flag=True,
    help="Disable gesture detection"
)
def live_demo(camera: int, pi_camera: bool, no_objects: bool, no_gestures: bool):
    """Run live camera demo."""
    import cv2
    from src.application.yolo_detector import YOLODetector
    from src.application.gesture_detector import TrafficGestureDetector
    from src.application.detection_service import DetectionService
    from src.infrastructure.camera_source import CameraSource, PiCameraSource
    from src.infrastructure.performance_monitor import PerformanceMonitor
    from src.infrastructure.visualization import DetectionVisualizer

    setup_logging(settings.log_level)

    async def run():
        # Initialize components
        object_detector = None if no_objects else YOLODetector(
            model_name=settings.yolo_model,
            confidence_threshold=settings.yolo_confidence
        )

        gesture_detector = None if no_gestures else TrafficGestureDetector(
            min_detection_confidence=settings.gesture_confidence
        )

        performance_monitor = PerformanceMonitor()

        detection_service = DetectionService(
            object_detector=object_detector,
            gesture_detector=gesture_detector,
            performance_monitor=performance_monitor
        )

        visualizer = DetectionVisualizer()

        # Initialize service
        await detection_service.initialize()

        # Setup camera source
        if pi_camera and settings.is_raspberry_pi:
            camera_source = PiCameraSource(
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps
            )
        else:
            camera_source = CameraSource(
                camera_index=camera,
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps
            )

        await camera_source.start()

        logging.info("Starting live demo. Press 'q' to quit.")

        try:
            frame_count = 0
            async for result in detection_service.process_stream(
                camera_source,
                detect_objects=not no_objects,
                detect_gestures=not no_gestures
            ):
                frame_count += 1

                # Get the actual frame for visualization
                frame = await camera_source.get_frame()
                if frame and frame.data is not None:
                    # Visualize
                    output_frame = frame.data.copy()
                    output_frame = visualizer.draw_detections(output_frame, result.object_detections)
                    output_frame = visualizer.draw_gestures(output_frame, result.gesture_detections)

                    # Add metrics overlay
                    metrics = performance_monitor.get_metrics()
                    output_frame = visualizer.draw_metrics(output_frame, metrics)

                    # Show frame
                    cv2.imshow("Traffic Detection - Live", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Log performance periodically
                if frame_count % 60 == 0:
                    logging.info(f"FPS: {result.fps:.2f}, Detections: {result.total_detections}")

        finally:
            cv2.destroyAllWindows()
            await camera_source.stop()
            await detection_service.cleanup()

            # Print summary
            print("\n" + performance_monitor.get_summary())

    asyncio.run(run())


@cli.command()
def download_models():
    """Download required models."""
    from ultralytics import YOLO

    setup_logging()

    logging.info("Downloading models...")

    # Create models directory
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # Download YOLO model
    model_path = settings.models_dir / settings.yolo_model
    if not model_path.exists():
        logging.info(f"Downloading {settings.yolo_model}...")
        model = YOLO(settings.yolo_model)
        model.export(format="onnx", imgsz=640)  # Also export ONNX version
        logging.info(f"Model downloaded to {model_path}")
    else:
        logging.info(f"Model already exists: {model_path}")

    logging.info("Models ready!")


if __name__ == "__main__":
    cli()