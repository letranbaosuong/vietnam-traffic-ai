#!/usr/bin/env python3
"""
Benchmark performance của các models trên Raspberry Pi 4
"""

import time
import cv2
import psutil
import json
from pathlib import Path
from traffic_detector import TrafficDetector

class PerformanceBenchmark:
    def __init__(self):
        self.results = {
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }

    def _get_system_info(self):
        """Thu thập thông tin hệ thống"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "python_version": psutil.sys.version,
            "opencv_version": cv2.__version__
        }

    def _get_pi_temperature(self):
        """Lấy nhiệt độ Pi"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except:
            return 0.0

    def benchmark_model(self, model_type, test_video, duration=60):
        """
        Benchmark 1 model trong thời gian nhất định
        Args:
            model_type: "yolov8n", "yolov5n", "mobilenet"
            test_video: đường dẫn video test
            duration: thời gian test (giây)
        """
        print(f"\n🔄 Benchmarking {model_type}...")

        # Khởi tạo detector
        detector = TrafficDetector(model_type=model_type)

        # Mở video
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {test_video}")
            return None

        # Metrics tracking
        metrics = {
            "model": model_type,
            "total_frames": 0,
            "total_time": 0,
            "fps_samples": [],
            "cpu_samples": [],
            "memory_samples": [],
            "temp_samples": [],
            "inference_times": []
        }

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Measure inference time
            inference_start = time.time()
            result_frame = detector.detect_frame(frame)
            inference_time = time.time() - inference_start

            # Collect metrics
            metrics["inference_times"].append(inference_time)
            metrics["cpu_samples"].append(psutil.cpu_percent())
            metrics["memory_samples"].append(psutil.virtual_memory().percent)
            metrics["temp_samples"].append(self._get_pi_temperature())

            # Calculate instantaneous FPS
            if inference_time > 0:
                fps = 1.0 / inference_time
                metrics["fps_samples"].append(fps)

            frame_count += 1

            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = 30 / (time.time() - inference_start) if frame_count > 0 else 0
                print(f"  Frame {frame_count}, FPS: {current_fps:.1f}, "
                      f"CPU: {psutil.cpu_percent():.1f}%, "
                      f"Temp: {self._get_pi_temperature():.1f}°C")

        cap.release()

        # Calculate final metrics
        total_time = time.time() - start_time
        metrics.update({
            "total_frames": frame_count,
            "total_time": total_time,
            "average_fps": frame_count / total_time,
            "average_inference_time": sum(metrics["inference_times"]) / len(metrics["inference_times"]),
            "max_fps": max(metrics["fps_samples"]) if metrics["fps_samples"] else 0,
            "min_fps": min(metrics["fps_samples"]) if metrics["fps_samples"] else 0,
            "average_cpu": sum(metrics["cpu_samples"]) / len(metrics["cpu_samples"]),
            "max_cpu": max(metrics["cpu_samples"]),
            "average_memory": sum(metrics["memory_samples"]) / len(metrics["memory_samples"]),
            "max_memory": max(metrics["memory_samples"]),
            "average_temp": sum(metrics["temp_samples"]) / len(metrics["temp_samples"]),
            "max_temp": max(metrics["temp_samples"])
        })

        print(f"✅ {model_type} benchmark complete:")
        print(f"   Average FPS: {metrics['average_fps']:.2f}")
        print(f"   CPU Usage: {metrics['average_cpu']:.1f}%")
        print(f"   Memory: {metrics['average_memory']:.1f}%")
        print(f"   Temperature: {metrics['average_temp']:.1f}°C")

        return metrics

    def run_all_benchmarks(self, test_video, duration=60):
        """Chạy benchmark cho tất cả models"""
        models = ["yolov8n", "yolov5n"]  # , "mobilenet"]

        print(f"🚀 Starting performance benchmarks")
        print(f"Test video: {test_video}")
        print(f"Duration: {duration}s each")
        print(f"System: {self.results['system_info']}")

        for model in models:
            try:
                result = self.benchmark_model(model, test_video, duration)
                if result:
                    self.results["benchmarks"][model] = result

                # Cool down period
                print("❄️ Cooling down...")
                time.sleep(30)

            except Exception as e:
                print(f"❌ Error benchmarking {model}: {e}")

        return self.results

    def save_results(self, output_file="benchmark_results.json"):
        """Lưu kết quả benchmark"""
        output_path = Path("../outputs") / output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Results saved to: {output_path}")

    def print_comparison(self):
        """In bảng so sánh các models"""
        if not self.results["benchmarks"]:
            print("❌ No benchmark results available")
            return

        print("\n📊 Performance Comparison")
        print("=" * 80)
        print(f"{'Model':<12} {'Avg FPS':<10} {'CPU %':<8} {'Memory %':<10} {'Temp °C':<10}")
        print("-" * 80)

        for model, metrics in self.results["benchmarks"].items():
            print(f"{model:<12} {metrics['average_fps']:<10.2f} "
                  f"{metrics['average_cpu']:<8.1f} {metrics['average_memory']:<10.1f} "
                  f"{metrics['average_temp']:<10.1f}")

        # Recommendations
        print("\n💡 Recommendations:")

        # Find best FPS
        best_fps_model = max(self.results["benchmarks"].items(),
                           key=lambda x: x[1]['average_fps'])
        print(f"   🏃 Fastest: {best_fps_model[0]} ({best_fps_model[1]['average_fps']:.1f} FPS)")

        # Find most efficient
        most_efficient = min(self.results["benchmarks"].items(),
                           key=lambda x: x[1]['average_cpu'])
        print(f"   ⚡ Most efficient: {most_efficient[0]} ({most_efficient[1]['average_cpu']:.1f}% CPU)")

        # Temperature warning
        for model, metrics in self.results["benchmarks"].items():
            if metrics['max_temp'] > 70:
                print(f"   🌡️ Warning: {model} runs hot ({metrics['max_temp']:.1f}°C)")

def main():
    """Main benchmark function"""
    benchmark = PerformanceBenchmark()

    # Test video path
    test_video = "../videos/traffic_sample.mp4"

    # Tạo test video nếu không có
    if not Path(test_video).exists():
        print("⚠️ Test video not found. Using webcam for 30s...")
        test_video = 0  # Webcam

    # Run benchmarks
    results = benchmark.run_all_benchmarks(test_video, duration=60)

    # Save and display results
    benchmark.save_results()
    benchmark.print_comparison()

if __name__ == "__main__":
    main()