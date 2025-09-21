#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.detector import TrafficObjectDetector
from src.video_processor import VideoProcessor
from src.rpi_optimizer import RPiOptimizer


def main():
    parser = argparse.ArgumentParser(description='Traffic Object Detection for Raspberry Pi')
    parser.add_argument('--mode', choices=['video', 'camera', 'benchmark', 'image'],
                       default='video', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input video/image path')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Config file path')
    parser.add_argument('--display', action='store_true',
                       help='Display output (not recommended on RPi)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--benchmark-frames', type=int, default=100,
                       help='Number of frames for benchmark')

    args = parser.parse_args()

    optimizer = RPiOptimizer(args.config)
    optimizer.optimize_system()
    optimizer.optimize_opencv()

    print("\n=== System Information ===")
    sys_info = optimizer.get_system_info()
    for key, value in sys_info.items():
        print(f"{key}: {value}")

    print("\n=== Initializing Detector ===")
    detector = TrafficObjectDetector(args.config)

    if args.mode == 'video':
        print("\n=== Processing Video ===")
        processor = VideoProcessor(args.config)

        input_path = args.input or processor.config['video']['input_path']
        output_path = args.output or processor.config['video']['output_path']

        stats = processor.process_video(
            detector,
            input_path=input_path,
            output_path=output_path,
            display=args.display
        )

        print("\n=== Processing Statistics ===")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print("\nDetection counts by class:")
        for class_name, count in sorted(stats['class_counts'].items()):
            print(f"  {class_name}: {count}")

    elif args.mode == 'camera':
        print("\n=== Starting Camera Detection ===")
        processor = VideoProcessor(args.config)
        processor.process_camera(
            detector,
            camera_id=args.camera_id,
            display=True,
            save_output=args.output is not None
        )

    elif args.mode == 'benchmark':
        print("\n=== Running Benchmark ===")
        processor = VideoProcessor(args.config)

        if not args.input and not Path(processor.config['video']['input_path']).exists():
            print("Error: Please provide input video for benchmark")
            sys.exit(1)

        results = processor.benchmark(detector, num_frames=args.benchmark_frames)

        if results:
            tips = optimizer.get_optimization_tips()
            if tips:
                print("\n=== Optimization Tips ===")
                for tip in tips:
                    print(tip)

    elif args.mode == 'image':
        import cv2

        if not args.input:
            print("Error: Please provide input image path")
            sys.exit(1)

        print(f"\n=== Processing Image: {args.input} ===")
        image = cv2.imread(args.input)

        if image is None:
            print(f"Error: Cannot read image from {args.input}")
            sys.exit(1)

        detections = detector.detect(image)
        result_image = detector.draw_detections(image, detections)

        output_path = args.output or 'data/outputs/result.jpg'
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result_image)

        print(f"Found {len(detections)} objects")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")

        print(f"Result saved to: {output_path}")

        if args.display:
            cv2.imshow('Detection Result', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    resources = optimizer.monitor_resources()
    print(f"\n=== Resource Usage ===")
    print(f"CPU: {resources['cpu_percent']:.1f}%")
    print(f"Memory: {resources['memory_percent']:.1f}% ({resources['memory_used_gb']:.2f} GB)")


if __name__ == '__main__':
    main()