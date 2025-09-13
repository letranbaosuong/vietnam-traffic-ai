#!/usr/bin/env python3
"""
Evaluate Vietnamese Traffic Sign Models
So sánh hiệu suất các model phát hiện biển báo
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import psutil

class ModelEvaluator:
    def __init__(self):
        self.models_dir = Path(".")
        self.test_video = "../../video/sample.mp4"

    def evaluate_model(self, model_path, frames_to_test=100):
        """Evaluate a single model"""
        print(f"\n🧪 Testing {model_path.name}")
        print("-" * 40)

        try:
            # Load model
            model = YOLO(str(model_path))

            # Open video
            cap = cv2.VideoCapture(self.test_video)
            if not cap.isOpened():
                print("❌ Could not open test video")
                return None

            # Performance metrics
            frame_times = []
            cpu_usage = []
            memory_usage = []

            for i in range(frames_to_test):
                ret, frame = cap.read()
                if not ret:
                    break

                # Measure inference time
                start_time = time.time()
                results = model(frame, conf=0.5, verbose=False)
                inference_time = time.time() - start_time

                frame_times.append(inference_time)
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)

                if i % 20 == 0:
                    print(f"  Frame {i}/{frames_to_test}")

            cap.release()

            # Calculate metrics
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            avg_memory = sum(memory_usage) / len(memory_usage)

            results = {
                'model': model_path.name,
                'avg_fps': round(avg_fps, 2),
                'avg_cpu': round(avg_cpu, 1),
                'avg_memory': round(avg_memory, 1),
                'pi4_estimated_fps': round(avg_fps * 0.6, 2)  # Estimate for Pi 4
            }

            print(f"📊 Results:")
            print(f"  Average FPS: {results['avg_fps']}")
            print(f"  CPU Usage: {results['avg_cpu']}%")
            print(f"  Memory: {results['avg_memory']}%")
            print(f"  Pi 4 Est. FPS: {results['pi4_estimated_fps']}")

            return results

        except Exception as e:
            print(f"❌ Error testing {model_path.name}: {e}")
            return None

    def run_comparison(self):
        """Run comparison of all available models"""
        print("🚦 Vietnamese Traffic Sign Model Comparison")
        print("=" * 50)

        models = list(self.models_dir.glob("*.pt"))
        if not models:
            print("❌ No .pt model files found")
            return

        results = []
        for model_path in models:
            result = self.evaluate_model(model_path)
            if result:
                results.append(result)

        # Print comparison table
        if results:
            print("\n📊 Comparison Summary:")
            print("-" * 60)
            print(f"{'Model':<15} {'FPS':<8} {'CPU%':<8} {'Memory%':<10} {'Pi4 FPS':<10}")
            print("-" * 60)

            for result in sorted(results, key=lambda x: x['avg_fps'], reverse=True):
                print(f"{result['model']:<15} {result['avg_fps']:<8} "
                      f"{result['avg_cpu']:<8} {result['avg_memory']:<10} "
                      f"{result['pi4_estimated_fps']:<10}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_comparison()
