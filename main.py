#!/usr/bin/env python3
"""
Vietnam Traffic AI System - Main Application
Hệ thống AI phân tích giao thông Việt Nam tích hợp nhiều module
"""

import sys
import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modules.object_detection import ObjectDetector
from modules.traffic_sign_detection import TrafficSignDetector
from modules.pose_analysis import PoseAnalyzer
from utils.video_processor import VideoProcessor

class VietnamTrafficAI:
    """
    Main class integrating all AI modules for Vietnam traffic analysis
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        print("Khởi tạo Vietnam Traffic AI System...")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize modules
        print("Đang tải các module AI...")
        self.object_detector = ObjectDetector(config_path)
        self.sign_detector = TrafficSignDetector(config_path)
        self.pose_analyzer = PoseAnalyzer(config_path)
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            output_fps=self.config['video']['output_fps']
        )
        
        print("✅ Khởi tạo hoàn thành!")
    
    def process_frame(self, frame: np.ndarray, enable_modules: dict = None) -> np.ndarray:
        """
        Process a single frame with all enabled modules
        
        Args:
            frame: Input frame
            enable_modules: Dict specifying which modules to enable
                          {'objects': True, 'signs': True, 'pose': True}
        
        Returns:
            Processed frame with annotations
        """
        if enable_modules is None:
            enable_modules = {'objects': True, 'signs': True, 'pose': True}
        
        processed_frame = frame.copy()
        stats = {'frame_stats': {}}
        
        # Object Detection
        if enable_modules.get('objects', True):
            detections = self.object_detector.detect(frame)
            processed_frame = self.object_detector.draw_detections(processed_frame, detections)
            stats['frame_stats']['objects'] = self.object_detector.get_traffic_statistics(detections)
        
        # Traffic Sign Detection  
        if enable_modules.get('signs', True):
            signs = self.sign_detector.detect_traffic_signs(frame)
            processed_frame = self.sign_detector.draw_signs(processed_frame, signs)
            stats['frame_stats']['signs'] = self.sign_detector.get_sign_statistics(signs)
        
        # Pose Analysis
        if enable_modules.get('pose', True):
            poses = self.pose_analyzer.analyze_pose(frame)
            processed_frame = self.pose_analyzer.draw_pose(processed_frame, poses)
            stats['frame_stats']['poses'] = self.pose_analyzer.get_pose_statistics(poses)
        
        # Add summary statistics to frame
        self._add_stats_overlay(processed_frame, stats['frame_stats'])
        
        return processed_frame
    
    def _add_stats_overlay(self, frame: np.ndarray, stats: dict) -> None:
        """Add statistics overlay to frame"""
        y_offset = 60
        line_height = 25
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 40), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Vietnam Traffic AI Stats", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        # Object stats
        if 'objects' in stats:
            obj_stats = stats['objects']
            cv2.putText(frame, f"Objects: {obj_stats.get('total_objects', 0)}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Vehicles: {obj_stats.get('vehicles', 0)} | People: {obj_stats.get('people', 0)}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Motorcycles: {obj_stats.get('motorcycles', 0)} | Cars: {obj_stats.get('cars', 0)}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += line_height
        
        # Sign stats
        if 'signs' in stats:
            sign_stats = stats['signs']
            cv2.putText(frame, f"Traffic Signs: {sign_stats.get('total_signs', 0)}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += line_height
        
        # Pose stats
        if 'poses' in stats:
            pose_stats = stats['poses']
            cv2.putText(frame, f"People Detected: {pose_stats.get('total_people', 0)}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            y_offset += line_height
    
    def run_webcam(self, enable_modules: dict = None):
        """Run real-time analysis on webcam feed"""
        print("Bắt đầu phân tích webcam...")
        print("Nhấn 'q' để thoát, 's' để lưu frame hiện tại")
        
        def processor_func(frame):
            return self.process_frame(frame, enable_modules)
        
        self.video_processor.process_webcam(
            processor_func=processor_func,
            window_name="Vietnam Traffic AI - Webcam",
            save_output=False
        )
    
    def process_video_file(self, input_path: str, output_path: str, enable_modules: dict = None):
        """Process video file"""
        print(f"Xử lý video: {input_path}")
        
        def processor_func(frame):
            return self.process_frame(frame, enable_modules)
        
        stats = self.video_processor.process_video(
            input_path=input_path,
            output_path=output_path,
            processor_func=processor_func,
            show_progress=True
        )
        
        print(f"✅ Hoàn thành! Video đã lưu: {output_path}")
        print(f"Thống kê: {stats['processed_frames']} frames, "
              f"{stats['processing_time']:.2f}s, "
              f"{stats['fps']:.2f} FPS")
    
    def batch_process_images(self, input_dir: str, output_dir: str, enable_modules: dict = None):
        """Batch process images in directory"""
        print(f"Xử lý batch images từ: {input_dir}")
        
        def processor_func(frame):
            return self.process_frame(frame, enable_modules)
        
        self.video_processor.batch_process_images(
            input_dir=input_dir,
            output_dir=output_dir,
            processor_func=processor_func
        )
    
    def analyze_video_statistics(self, video_path: str, output_file: str = None):
        """Analyze video and generate comprehensive statistics"""
        print(f"Phân tích thống kê video: {video_path}")
        
        def analyzer_func(frame):
            stats = {}
            
            # Get detections
            detections = self.object_detector.detect(frame)
            signs = self.sign_detector.detect_traffic_signs(frame)
            poses = self.pose_analyzer.analyze_pose(frame)
            
            # Extract statistics
            stats.update(self.object_detector.get_traffic_statistics(detections))
            stats.update(self.sign_detector.get_sign_statistics(signs))
            stats.update(self.pose_analyzer.get_pose_statistics(poses))
            
            return stats
        
        video_stats = self.video_processor.extract_statistics(video_path, analyzer_func)
        
        # Save statistics if output file specified
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(video_stats, f, indent=2, ensure_ascii=False)
            print(f"Thống kê đã lưu: {output_file}")
        
        return video_stats
    
    def demo_all_features(self):
        """Run a comprehensive demo of all features"""
        print("🚀 Chạy demo đầy đủ Vietnam Traffic AI System...")
        
        # Check if webcam is available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("📹 Webcam có sẵn - bắt đầu demo real-time")
            
            # Run webcam demo for 30 seconds
            def demo_processor(frame):
                # Add demo info
                demo_frame = self.process_frame(frame)
                cv2.putText(demo_frame, "DEMO MODE - Vietnam Traffic AI", 
                           (10, demo_frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(demo_frame, "Press 'q' to exit demo", 
                           (10, demo_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                return demo_frame
            
            self.video_processor.process_webcam(
                processor_func=demo_processor,
                window_name="Vietnam Traffic AI - DEMO",
                save_output=False
            )
        else:
            print("❌ Không tìm thấy webcam")
            print("💡 Hãy thử chạy với video file: python main.py --mode video --input your_video.mp4")

def main():
    parser = argparse.ArgumentParser(description="Vietnam Traffic AI System")
    
    parser.add_argument('--mode', choices=['webcam', 'video', 'images', 'analyze', 'demo'], 
                       default='demo', help='Chế độ chạy')
    parser.add_argument('--input', type=str, help='Đường dẫn input (video/image directory)')
    parser.add_argument('--output', type=str, help='Đường dẫn output')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='File cấu hình')
    
    # Module enable/disable options
    parser.add_argument('--no-objects', action='store_true', 
                       help='Tắt object detection')
    parser.add_argument('--no-signs', action='store_true', 
                       help='Tắt traffic sign detection')
    parser.add_argument('--no-pose', action='store_true', 
                       help='Tắt pose analysis')
    
    args = parser.parse_args()
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Không tìm thấy file config: {args.config}")
        return
    
    try:
        # Initialize system
        traffic_ai = VietnamTrafficAI(args.config)
        
        # Configure enabled modules
        enable_modules = {
            'objects': not args.no_objects,
            'signs': not args.no_signs,
            'pose': not args.no_pose
        }
        
        # Run based on mode
        if args.mode == 'webcam':
            traffic_ai.run_webcam(enable_modules)
        
        elif args.mode == 'video':
            if not args.input:
                print("❌ Cần cung cấp --input cho mode video")
                return
            
            output = args.output or f"processed_{Path(args.input).name}"
            traffic_ai.process_video_file(args.input, output, enable_modules)
        
        elif args.mode == 'images':
            if not args.input:
                print("❌ Cần cung cấp --input cho mode images")
                return
            
            output = args.output or "processed_images"
            traffic_ai.batch_process_images(args.input, output, enable_modules)
        
        elif args.mode == 'analyze':
            if not args.input:
                print("❌ Cần cung cấp --input cho mode analyze")
                return
            
            output = args.output or f"stats_{Path(args.input).stem}.json"
            traffic_ai.analyze_video_statistics(args.input, output)
        
        elif args.mode == 'demo':
            traffic_ai.demo_all_features()
    
    except KeyboardInterrupt:
        print("\\n🛑 Dừng bởi người dùng")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()