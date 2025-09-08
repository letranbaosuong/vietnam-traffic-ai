#!/usr/bin/env python3
"""
Script để thu thập dữ liệu video từ webcam hoặc video file
cho việc huấn luyện mô hình AI giao thông Việt Nam
"""

import cv2
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class DataCollector:
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of data
        self.video_dir = self.output_dir / "videos"
        self.images_dir = self.output_dir / "images" 
        self.annotations_dir = Path("data/annotations")
        
        for directory in [self.video_dir, self.images_dir, self.annotations_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def collect_from_webcam(self, duration: int = 60, fps: int = 30):
        """Thu thập video từ webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_traffic_{timestamp}.mp4"
        output_path = self.video_dir / filename
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (1280, 720))
        
        print(f"Bắt đầu thu thập video trong {duration} giây...")
        print(f"Lưu tại: {output_path}")
        print("Nhấn 'q' để dừng sớm, 's' để chụp ảnh")
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Display frame
            display_frame = frame.copy()
            elapsed = int(time.time() - start_time)
            remaining = max(0, duration - elapsed)
            
            cv2.putText(display_frame, f"Recording: {elapsed}s / {duration}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frames: {frame_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit, 's' to save frame", 
                       (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Data Collection - Vietnam Traffic AI', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or elapsed >= duration:
                break
            elif key == ord('s'):
                # Save current frame as image
                img_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_filename = f"frame_{img_timestamp}.jpg"
                img_path = self.images_dir / img_filename
                cv2.imwrite(str(img_path), frame)
                print(f"Đã lưu frame: {img_filename}")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Hoàn thành! Đã thu thập {frame_count} frames")
        print(f"Video lưu tại: {output_path}")
    
    def extract_frames_from_video(self, video_path: str, interval: int = 30):
        """Trích xuất frames từ video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return
        
        video_name = Path(video_path).stem
        frame_dir = self.images_dir / f"frames_{video_name}"
        frame_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        print(f"Trích xuất frames từ {video_path} (mỗi {interval} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                frame_path = frame_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"Đã lưu {saved_count} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Hoàn thành! Đã trích xuất {saved_count} frames từ {frame_count} frames tổng")
    
    def create_annotation_template(self, image_dir: str):
        """Tạo template cho việc gán nhãn"""
        image_dir = Path(image_dir)
        
        # Vietnam traffic classes
        classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic_light', 'stop_sign', 'parking_sign', 'yield_sign',
            'speed_limit_sign', 'no_entry_sign', 'one_way_sign',
            'pedestrian_crossing_sign', 'school_zone_sign', 'construction_sign'
        ]
        
        # Create classes.txt file
        classes_file = self.annotations_dir / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(classes):
                f.write(f"{i}: {class_name}\n")
        
        print(f"Đã tạo file classes: {classes_file}")
        
        # Create annotation guide
        guide_file = self.annotations_dir / "annotation_guide.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write("""# Hướng dẫn gán nhãn dữ liệu giao thông Việt Nam

## Các lớp đối tượng:

### Phương tiện giao thông:
- person: Người đi bộ, người điều khiển giao thông
- bicycle: Xe đạp
- car: Ô tô con, taxi
- motorcycle: Xe máy, xe scooter (rất quan trọng cho VN!)
- bus: Xe buýt
- truck: Xe tải, xe container

### Biển báo giao thông:
- traffic_light: Đèn giao thông
- stop_sign: Biển báo dừng
- parking_sign: Biển báo đỗ xe
- yield_sign: Biển báo nhường đường
- speed_limit_sign: Biển báo giới hạn tốc độ
- no_entry_sign: Biển báo cấm đi vào
- one_way_sign: Biển báo một chiều
- pedestrian_crossing_sign: Biển báo cho người đi bộ qua đường
- school_zone_sign: Biển báo khu vực trường học
- construction_sign: Biển báo thi công

## Lưu ý đặc biệt cho giao thông Việt Nam:
1. **Xe máy** là phương tiện chính - cần gán nhãn cẩn thận
2. **Mật độ cao** - nhiều đối tượng chồng lấn
3. **Biển báo song ngữ** - Tiếng Việt và English
4. **Giao thông hỗn hợp** - nhiều loại phương tiện cùng lúc

## Format annotation:
- YOLO format: class_id center_x center_y width height (normalized)
- Sử dụng tools như LabelImg hoặc Roboflow
""")
        
        print(f"Đã tạo hướng dẫn gán nhãn: {guide_file}")
    
    def create_dataset_split(self, images_dir: str, train_ratio: float = 0.7, 
                           val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Chia dataset thành train/val/test"""
        import random
        
        images_dir = Path(images_dir)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        if not image_files:
            print(f"Không tìm thấy ảnh trong {images_dir}")
            return
        
        # Shuffle
        random.shuffle(image_files)
        
        # Calculate splits
        total = len(image_files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Create split files
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_file = self.annotations_dir / f"{split_name}.txt"
            with open(split_file, 'w') as f:
                for file_path in files:
                    f.write(f"{file_path}\\n")
            
            print(f"{split_name}: {len(files)} files -> {split_file}")
        
        print(f"Dataset split hoàn thành: {len(train_files)}/{len(val_files)}/{len(test_files)}")

def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu cho AI giao thông Việt Nam")
    parser.add_argument('--mode', choices=['webcam', 'extract', 'annotate', 'split'], 
                       required=True, help='Chế độ thu thập dữ liệu')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Thời gian thu thập từ webcam (giây)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frame rate cho video')
    parser.add_argument('--video', type=str, 
                       help='Đường dẫn video để trích xuất frames')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Khoảng cách giữa các frames được trích xuất')
    parser.add_argument('--images_dir', type=str, default='data/raw/images',
                       help='Thư mục chứa ảnh')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Thư mục output')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.output_dir)
    
    if args.mode == 'webcam':
        collector.collect_from_webcam(args.duration, args.fps)
    
    elif args.mode == 'extract':
        if not args.video:
            print("Cần cung cấp --video cho chế độ extract")
            return
        collector.extract_frames_from_video(args.video, args.interval)
    
    elif args.mode == 'annotate':
        collector.create_annotation_template(args.images_dir)
    
    elif args.mode == 'split':
        collector.create_dataset_split(args.images_dir)

if __name__ == "__main__":
    main()