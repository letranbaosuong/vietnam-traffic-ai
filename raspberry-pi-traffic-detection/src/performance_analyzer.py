#!/usr/bin/env python3
"""
Performance Analysis và Report Generator
Phân tích hiệu suất các video test khác nhau
"""

import json
import time
import cv2
import psutil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from integrated_detector import IntegratedTrafficSystem

class PerformanceAnalyzer:
    def __init__(self):
        """Initialize performance analyzer"""
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self):
        """Thu thập thông tin hệ thống"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "platform": "macOS" if "darwin" in __import__("platform").system().lower() else "Other"
        }

    def analyze_video(self, video_path, video_name, max_frames=300):
        """
        Phân tích hiệu suất một video cụ thể
        """
        print(f"\n🎬 Analyzing: {video_name}")
        print("=" * 50)

        # Initialize system
        system = IntegratedTrafficSystem(model_type="yolov8n", confidence=0.5)

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        video_info = {
            "resolution": f"{width}x{height}",
            "original_fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(duration, 2)
        }

        # Performance tracking
        metrics = {
            "frame_times": [],
            "vehicle_counts": [],
            "processing_fps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "inference_times": [],
            "lane_detections": 0,
            "sign_detections": 0
        }

        frame_count = 0
        total_processing_time = 0
        start_time = time.time()

        print(f"📺 Processing {min(max_frames, total_frames)} frames...")

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame_start = time.time()
            results = system.detect_all(frame)
            frame_time = time.time() - frame_start

            # Collect metrics
            metrics["frame_times"].append(frame_time)
            metrics["processing_fps"].append(1.0 / frame_time if frame_time > 0 else 0)
            metrics["cpu_usage"].append(psutil.cpu_percent())
            metrics["memory_usage"].append(psutil.virtual_memory().percent)

            # Count detections
            vehicle_count = 0
            if results.get('vehicles'):
                # Count vehicles from YOLO output (rough estimation)
                vehicle_count = len([x for x in str(results['vehicles']) if 'car' in x or 'truck' in x or 'bus' in x])

            metrics["vehicle_counts"].append(vehicle_count)

            # Lane and sign detection flags
            if results.get('lanes') and results['lanes']['data']['left_lane']:
                metrics["lane_detections"] += 1

            if results.get('signs') and results['signs']['detections']:
                metrics["sign_detections"] += len(results['signs']['detections'])

            frame_count += 1
            total_processing_time += frame_time

            # Progress update
            if frame_count % 30 == 0:
                avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
                print(f"  📊 Frame {frame_count}/{min(max_frames, total_frames)} - Avg FPS: {avg_fps:.2f}")

        cap.release()

        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

        analysis_result = {
            "video_info": video_info,
            "performance": {
                "frames_processed": frame_count,
                "total_processing_time": round(total_processing_time, 2),
                "real_time_duration": round(total_time, 2),
                "average_fps": round(avg_fps, 2),
                "max_fps": round(max(metrics["processing_fps"]) if metrics["processing_fps"] else 0, 2),
                "min_fps": round(min(metrics["processing_fps"]) if metrics["processing_fps"] else 0, 2),
                "frame_time_avg": round(np.mean(metrics["frame_times"]), 3),
                "frame_time_std": round(np.std(metrics["frame_times"]), 3),
                "cpu_usage_avg": round(np.mean(metrics["cpu_usage"]), 1),
                "memory_usage_avg": round(np.mean(metrics["memory_usage"]), 1)
            },
            "detections": {
                "vehicle_count_avg": round(np.mean(metrics["vehicle_counts"]), 1),
                "vehicle_count_max": max(metrics["vehicle_counts"]) if metrics["vehicle_counts"] else 0,
                "lane_detection_rate": round(metrics["lane_detections"] / frame_count * 100, 1),
                "sign_detections_total": metrics["sign_detections"]
            },
            "raw_metrics": metrics
        }

        print(f"✅ Analysis complete: {frame_count} frames, {avg_fps:.2f} avg FPS")

        return analysis_result

    def run_full_analysis(self):
        """Chạy phân tích đầy đủ cho tất cả videos"""
        videos = [
            ("video/sample.mp4", "Traffic Sample"),
            ("video/car-driver.mp4", "Car Driver"),
            ("video/mobycle-driver.mp4", "Motorcycle Driver")
        ]

        print("🚀 Starting Full Performance Analysis")
        print(f"💻 System: {self.system_info}")

        for video_path, video_name in videos:
            if Path(video_path).exists():
                self.results[video_name] = self.analyze_video(video_path, video_name)
            else:
                print(f"⚠️ Video not found: {video_path}")

        return self.results

    def generate_report(self, output_dir="outputs"):
        """Tạo báo cáo hiệu suất chi tiết"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # JSON Report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.system_info,
            "results": self.results
        }

        json_file = output_path / "performance_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Markdown Report
        md_file = output_path / "performance_report.md"
        self._generate_markdown_report(md_file, report_data)

        # Performance Charts
        self._generate_charts(output_path)

        print(f"\n📊 Reports generated:")
        print(f"  📄 JSON: {json_file}")
        print(f"  📝 Markdown: {md_file}")
        print(f"  📈 Charts: {output_path}/charts/")

        return str(json_file), str(md_file)

    def _generate_markdown_report(self, file_path, data):
        """Tạo báo cáo Markdown"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Traffic Detection Performance Report\n\n")
            f.write(f"**Generated:** {data['timestamp']}  \n")
            f.write(f"**System:** {data['system_info']['platform']}  \n")
            f.write(f"**CPU Cores:** {data['system_info']['cpu_count']}  \n")
            f.write(f"**Memory:** {data['system_info']['memory_total_gb']}GB  \n\n")

            f.write("## 🎯 Executive Summary\n\n")

            if data['results']:
                # Summary table
                f.write("| Video | Resolution | Avg FPS | CPU % | Memory % | Vehicles/Frame |\n")
                f.write("|-------|------------|---------|-------|----------|----------------|\n")

                for video_name, result in data['results'].items():
                    perf = result['performance']
                    det = result['detections']
                    vid_info = result['video_info']

                    f.write(f"| {video_name} | {vid_info['resolution']} | "
                           f"{perf['average_fps']:.1f} | {perf['cpu_usage_avg']:.1f}% | "
                           f"{perf['memory_usage_avg']:.1f}% | {det['vehicle_count_avg']:.1f} |\n")

            f.write("\n## 📈 Detailed Analysis\n\n")

            for video_name, result in data['results'].items():
                f.write(f"### 🎬 {video_name}\n\n")

                perf = result['performance']
                det = result['detections']
                vid_info = result['video_info']

                # Video Info
                f.write("**📺 Video Information:**\n")
                f.write(f"- Resolution: {vid_info['resolution']}\n")
                f.write(f"- Original FPS: {vid_info['original_fps']}\n")
                f.write(f"- Duration: {vid_info['duration_sec']}s\n")
                f.write(f"- Total Frames: {vid_info['total_frames']}\n\n")

                # Performance Metrics
                f.write("**⚡ Performance Metrics:**\n")
                f.write(f"- Average FPS: **{perf['average_fps']} FPS**\n")
                f.write(f"- FPS Range: {perf['min_fps']} - {perf['max_fps']} FPS\n")
                f.write(f"- Frame Time: {perf['frame_time_avg']}ms ± {perf['frame_time_std']}ms\n")
                f.write(f"- CPU Usage: {perf['cpu_usage_avg']}%\n")
                f.write(f"- Memory Usage: {perf['memory_usage_avg']}%\n")
                f.write(f"- Processing Efficiency: {perf['total_processing_time']/perf['frames_processed']*1000:.1f}ms/frame\n\n")

                # Detection Results
                f.write("**🔍 Detection Results:**\n")
                f.write(f"- Average Vehicles per Frame: {det['vehicle_count_avg']}\n")
                f.write(f"- Max Vehicles Detected: {det['vehicle_count_max']}\n")
                f.write(f"- Lane Detection Rate: {det['lane_detection_rate']}%\n")
                f.write(f"- Traffic Signs Detected: {det['sign_detections_total']}\n\n")

                # Raspberry Pi 4 Estimates
                pi4_fps = perf['average_fps'] * 0.6  # Estimated 60% of macOS performance
                f.write("**🥧 Raspberry Pi 4 Estimates:**\n")
                f.write(f"- Estimated FPS: ~{pi4_fps:.1f} FPS\n")
                f.write(f"- Recommended for: {'Real-time' if pi4_fps >= 10 else 'Batch processing'}\n\n")

            f.write("## 🎯 Recommendations\n\n")

            # Find best performing video
            if data['results']:
                best_fps = max(data['results'].items(), key=lambda x: x[1]['performance']['average_fps'])
                f.write(f"**Best Performance:** {best_fps[0]} ({best_fps[1]['performance']['average_fps']} FPS)\n\n")

                f.write("**Optimization Suggestions:**\n")
                f.write("- For Pi 4 deployment, consider:\n")
                f.write("  - Reducing input resolution to 640x480\n")
                f.write("  - Disabling sign detection for higher FPS\n")
                f.write("  - Using frame skipping (process every 2nd frame)\n")
                f.write("  - Implementing multi-threading optimizations\n\n")

            f.write("## 📝 Technical Notes\n\n")
            f.write("- All tests performed with YOLOv8n model\n")
            f.write("- Integrated detection includes: Vehicle + Lane + Sign detection\n")
            f.write("- Pi 4 estimates based on CPU performance scaling\n")
            f.write("- Actual Pi 4 performance may vary based on cooling and configuration\n")

    def _generate_charts(self, output_dir):
        """Tạo biểu đồ hiệu suất"""
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        if not self.results:
            return

        # FPS Comparison Chart
        videos = list(self.results.keys())
        fps_values = [self.results[v]['performance']['average_fps'] for v in videos]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(videos, fps_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Average FPS Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('FPS')
        plt.ylim(0, max(fps_values) * 1.2)

        # Add value labels on bars
        for bar, fps in zip(bars, fps_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{fps:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(charts_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Resource Usage Chart
        cpu_values = [self.results[v]['performance']['cpu_usage_avg'] for v in videos]
        memory_values = [self.results[v]['performance']['memory_usage_avg'] for v in videos]

        x = np.arange(len(videos))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, cpu_values, width, label='CPU Usage (%)', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, memory_values, width, label='Memory Usage (%)', color='#4ECDC4')

        ax.set_title('System Resource Usage', fontsize=16, fontweight='bold')
        ax.set_ylabel('Usage Percentage (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(videos)
        ax.legend()

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(charts_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Detection Performance Chart
        vehicle_counts = [self.results[v]['detections']['vehicle_count_avg'] for v in videos]
        lane_rates = [self.results[v]['detections']['lane_detection_rate'] for v in videos]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Vehicle detection
        bars1 = ax1.bar(videos, vehicle_counts, color='#45B7D1')
        ax1.set_title('Average Vehicles per Frame')
        ax1.set_ylabel('Vehicle Count')

        for bar, count in zip(bars1, vehicle_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{count:.1f}', ha='center', va='bottom')

        # Lane detection rate
        bars2 = ax2.bar(videos, lane_rates, color='#96CEB4')
        ax2.set_title('Lane Detection Success Rate')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)

        for bar, rate in zip(bars2, lane_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(charts_dir / 'detection_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Charts saved to: {charts_dir}")

def main():
    """Main analysis function"""
    analyzer = PerformanceAnalyzer()

    # Run full analysis
    results = analyzer.run_full_analysis()

    # Generate reports
    json_file, md_file = analyzer.generate_report()

    print(f"\n🎉 Performance analysis complete!")
    print(f"📊 View reports:")
    print(f"  - {md_file}")
    print(f"  - {json_file}")

if __name__ == "__main__":
    main()