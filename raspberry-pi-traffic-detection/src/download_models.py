#!/usr/bin/env python3
"""
Download và setup models cho Raspberry Pi 4
"""

import os
import urllib.request
from pathlib import Path
from ultralytics import YOLO

def download_yolo_models():
    """Download YOLO models"""
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)

    print("🔄 Downloading YOLOv8n...")
    yolo8n = YOLO('yolov8n.pt')
    print("✅ YOLOv8n downloaded")

    print("🔄 Downloading YOLOv5n...")
    yolo5n = YOLO('yolov5n.pt')
    print("✅ YOLOv5n downloaded")

def download_mobilenet_ssd():
    """Download MobileNet SSD"""
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)

    base_url = "http://download.tensorflow.org/models/object_detection/"

    files = {
        "ssd_mobilenet_v2_coco_2018_03_29.tar.gz":
            "ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
        "ssd_mobilenet_v2_coco.pbtxt":
            "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    }

    print("🔄 Downloading MobileNet SSD...")
    # Implementation here
    print("✅ MobileNet SSD downloaded")

def download_sample_videos():
    """Download sample traffic videos"""
    videos_dir = Path("../videos")
    videos_dir.mkdir(exist_ok=True)

    # Sample traffic video URLs (replace with actual ones)
    sample_videos = [
        "https://sample-videos.com/zip/10/mp4/720/mp4-720p.mp4"
    ]

    print("🔄 Downloading sample videos...")
    for i, url in enumerate(sample_videos):
        try:
            filename = f"traffic_sample_{i+1}.mp4"
            filepath = videos_dir / filename
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ Downloaded: {filename}")
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Raspberry Pi Traffic Detection")

    download_yolo_models()
    download_mobilenet_ssd()
    download_sample_videos()

    print("✅ Setup complete!")