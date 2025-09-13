#!/usr/bin/env python3
"""
Test Vietnamese Traffic Sign Models
So sánh hiệu suất các model phát hiện biển báo Việt Nam
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import psutil

class VietnameseModelTester:
    def __init__(self):
        """Initialize model tester"""
        self.models_dir = Path("../models")
        self.test_videos = [
            "video/sample.mp4",
            "video/car-driver.mp4",
            "video/mobycle-driver.mp4"
        ]

    def test_model_on_video(self, model_path, video_path, max_frames=100):
        """Test a model on a specific video"""
        print(f"\n🎬 Testing {model_path.name} on {Path(video_path).name}")
        print("-" * 60)

        try:
            # Load model
            model = YOLO(str(model_path))

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Could not open video: {video_path}")
                return None

            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"📺 Video: {width}x{height}, {fps} FPS")

            # Performance tracking
            frame_times = []
            detection_counts = []
            cpu_usage = []
            memory_usage = []

            frame_count = 0
            total_detections = 0

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Measure inference time
                start_time = time.time()

                # Run detection
                results = model(frame, conf=0.5, verbose=False)

                inference_time = time.time() - start_time

                # Count detections
                detections_in_frame = 0
                if results and len(results) > 0:
                    for result in results:
                        if result.boxes is not None:
                            detections_in_frame = len(result.boxes)
                            total_detections += detections_in_frame

                # Collect metrics
                frame_times.append(inference_time)
                detection_counts.append(detections_in_frame)
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)

                frame_count += 1

                # Progress update
                if frame_count % 20 == 0:
                    avg_fps = frame_count / sum(frame_times)
                    print(f"  📊 Frame {frame_count}/{max_frames} - FPS: {avg_fps:.1f}")

            cap.release()

            # Calculate metrics
            if frame_times:
                avg_fps = len(frame_times) / sum(frame_times)
                avg_inference_time = np.mean(frame_times)
                avg_cpu = np.mean(cpu_usage)
                avg_memory = np.mean(memory_usage)
                avg_detections = np.mean(detection_counts)

                results = {
                    'model': model_path.name,
                    'video': Path(video_path).name,
                    'frames_tested': frame_count,
                    'avg_fps': round(avg_fps, 2),
                    'avg_inference_time': round(avg_inference_time * 1000, 1),  # ms
                    'avg_cpu': round(avg_cpu, 1),
                    'avg_memory': round(avg_memory, 1),
                    'total_detections': total_detections,
                    'avg_detections_per_frame': round(avg_detections, 1),
                    'pi4_estimated_fps': round(avg_fps * 0.6, 2)  # Pi 4 estimate
                }

                # Print results
                print(f"📊 Results:")
                print(f"  ⚡ Average FPS: {results['avg_fps']}")
                print(f"  🕒 Inference Time: {results['avg_inference_time']}ms")
                print(f"  💻 CPU Usage: {results['avg_cpu']}%")
                print(f"  💾 Memory: {results['avg_memory']}%")
                print(f"  🎯 Total Detections: {results['total_detections']}")
                print(f"  📈 Avg Detections/Frame: {results['avg_detections_per_frame']}")
                print(f"  🥧 Pi 4 Est. FPS: {results['pi4_estimated_fps']}")

                return results

        except Exception as e:
            print(f"❌ Error testing {model_path.name}: {e}")
            return None

    def run_comprehensive_test(self):
        """Run comprehensive test on all models and videos"""
        print("🚦 Vietnamese Traffic Sign Models - Comprehensive Test")
        print("=" * 70)

        # Find available models
        model_files = list(self.models_dir.glob("*.pt"))
        if not model_files:
            print("❌ No .pt model files found in models directory")
            return

        print(f"📦 Found {len(model_files)} models:")
        for model in model_files:
            print(f"  - {model.name}")

        # Find available videos
        available_videos = []
        for video_path in self.test_videos:
            if Path(video_path).exists():
                available_videos.append(video_path)
                print(f"✅ Video found: {video_path}")
            else:
                print(f"⚠️ Video not found: {video_path}")

        if not available_videos:
            print("❌ No test videos found")
            return

        # Test all combinations
        all_results = []

        for model_path in model_files:
            for video_path in available_videos:
                result = self.test_model_on_video(model_path, video_path, max_frames=50)
                if result:
                    all_results.append(result)

        # Generate comparison report
        self.generate_comparison_report(all_results)

    def generate_comparison_report(self, results):
        """Generate detailed comparison report"""
        if not results:
            print("❌ No results to compare")
            return

        print("\n" + "=" * 70)
        print("📊 VIETNAMESE TRAFFIC SIGN MODELS COMPARISON REPORT")
        print("=" * 70)

        # Group by model
        models = {}
        for result in results:
            model_name = result['model']
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(result)

        # Summary table
        print("\n🎯 PERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"{'Model':<15} {'Video':<20} {'FPS':<8} {'Det/Frame':<10} {'Pi4 FPS':<10}")
        print("-" * 70)

        for model_name, model_results in models.items():
            for result in model_results:
                print(f"{result['model']:<15} {result['video']:<20} "
                      f"{result['avg_fps']:<8} {result['avg_detections_per_frame']:<10} "
                      f"{result['pi4_estimated_fps']:<10}")

        # Best performers
        print("\n🏆 BEST PERFORMERS")
        print("-" * 30)

        # Fastest model
        fastest = max(results, key=lambda x: x['avg_fps'])
        print(f"⚡ Fastest: {fastest['model']} ({fastest['avg_fps']} FPS on {fastest['video']})")

        # Most detections
        most_detections = max(results, key=lambda x: x['total_detections'])
        print(f"🎯 Most Detections: {most_detections['model']} ({most_detections['total_detections']} total on {most_detections['video']})")

        # Best for Pi 4
        best_pi4 = max(results, key=lambda x: x['pi4_estimated_fps'])
        print(f"🥧 Best for Pi 4: {best_pi4['model']} (~{best_pi4['pi4_estimated_fps']} FPS)")

        # Recommendations
        print("\n💡 RECOMMENDATIONS FOR RASPBERRY PI 4")
        print("-" * 40)

        avg_fps_by_model = {}
        for model_name, model_results in models.items():
            avg_fps = np.mean([r['avg_fps'] for r in model_results])
            avg_fps_by_model[model_name] = avg_fps

        sorted_models = sorted(avg_fps_by_model.items(), key=lambda x: x[1], reverse=True)

        for i, (model, fps) in enumerate(sorted_models, 1):
            pi4_fps = fps * 0.6
            if pi4_fps >= 8:
                recommendation = "✅ Excellent for real-time"
            elif pi4_fps >= 5:
                recommendation = "👍 Good for real-time"
            elif pi4_fps >= 3:
                recommendation = "⚠️ Acceptable for monitoring"
            else:
                recommendation = "❌ Too slow for real-time"

            print(f"{i}. {model}: ~{pi4_fps:.1f} FPS - {recommendation}")

        # Technical notes
        print("\n📝 TECHNICAL NOTES")
        print("-" * 20)
        print("- Tests performed on macOS with limited frames (50 per video)")
        print("- Pi 4 estimates based on 60% performance scaling")
        print("- Detection counts may vary based on video content")
        print("- Real Pi 4 performance depends on cooling and configuration")

        # Save results to file
        report_path = Path("outputs/vietnamese_models_comparison.txt")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Vietnamese Traffic Sign Models Comparison Report\n")
            f.write("=" * 50 + "\n\n")

            for result in results:
                f.write(f"Model: {result['model']}\n")
                f.write(f"Video: {result['video']}\n")
                f.write(f"FPS: {result['avg_fps']}\n")
                f.write(f"Detections: {result['total_detections']}\n")
                f.write(f"Pi 4 Est: {result['pi4_estimated_fps']} FPS\n")
                f.write("-" * 30 + "\n")

        print(f"\n💾 Detailed report saved to: {report_path}")

def main():
    """Main testing function"""
    tester = VietnameseModelTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()