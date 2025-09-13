#!/usr/bin/env python3
"""
Auto Download Free Test Videos
Tự động tải video test miễn phí từ các nguồn có sẵn
"""

import requests
import cv2
from pathlib import Path
import time

class AutoVideoDownloader:
    def __init__(self):
        """Initialize auto downloader"""
        self.video_dir = Path("../video")
        self.video_dir.mkdir(exist_ok=True)

        # Sample free video URLs (these would need to be actual direct links)
        self.sample_videos = {
            "urban_traffic": {
                "filename": "urban_traffic_test.mp4",
                "description": "Urban traffic for testing",
                "expected_size": "5-10MB"
            },
            "highway_driving": {
                "filename": "highway_driving_test.mp4",
                "description": "Highway driving footage",
                "expected_size": "8-15MB"
            }
        }

    def create_sample_video(self, filename, duration=10):
        """Create a sample video for testing if no downloads available"""
        video_path = self.video_dir / filename

        if video_path.exists():
            print(f"✅ Sample video already exists: {filename}")
            return str(video_path)

        print(f"🎬 Creating sample test video: {filename}")

        # Create a simple test video with moving objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))

        for frame_num in range(duration * 30):  # 30 FPS for duration seconds
            # Create frame with moving rectangles (simulating vehicles)
            frame = cv2.imread("README.md") if Path("README.md").exists() else None

            if frame is None:
                # Create synthetic frame
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = (50, 100, 50)  # Dark green background

                # Add moving "vehicles"
                for i in range(3):
                    x = (frame_num * 5 + i * 200) % 1280
                    y = 300 + i * 100
                    cv2.rectangle(frame, (x, y), (x + 80, y + 40), (0, 255, 255), -1)
                    cv2.putText(frame, f"Vehicle {i+1}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add frame number
                cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"✅ Created sample video: {video_path}")
        return str(video_path)

    def download_from_url(self, url, filename):
        """Download video from direct URL"""
        video_path = self.video_dir / filename

        if video_path.exists():
            print(f"✅ Video already exists: {filename}")
            return str(video_path)

        try:
            print(f"📥 Downloading {filename} from {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✅ Downloaded: {filename}")
            return str(video_path)

        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return None

    def get_free_demo_videos(self):
        """Get free demo videos from public sources"""
        demo_urls = {
            # These would be actual URLs to free demo videos
            "demo_traffic.mp4": "https://sample-videos.com/zip/10/mp4/720/mp4-25s-1.2MB.mp4",  # Example
        }

        downloaded_videos = []

        for filename, url in demo_urls.items():
            try:
                video_path = self.download_from_url(url, filename)
                if video_path and self.verify_video(video_path):
                    downloaded_videos.append(video_path)
            except:
                print(f"⚠️ Could not download {filename}, will create sample instead")

        return downloaded_videos

    def verify_video(self, video_path):
        """Verify video is valid and get info"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            print(f"📺 {Path(video_path).name}: {width}x{height}, {fps}FPS, {duration:.1f}s")

            cap.release()
            return True

        except:
            return False

    def run_auto_download(self):
        """Run automatic video download process"""
        print("🎬 Auto Download Test Videos")
        print("=" * 40)

        downloaded_videos = []

        # Try to get free demo videos
        print("\n📥 Trying to download free demo videos...")
        demo_videos = self.get_free_demo_videos()
        downloaded_videos.extend(demo_videos)

        # Create sample videos if no downloads available
        if len(downloaded_videos) == 0:
            print("\n🎨 Creating sample test videos...")

            sample_videos = [
                "sample_urban_traffic.mp4",
                "sample_highway_drive.mp4"
            ]

            for video_name in sample_videos:
                try:
                    # Import numpy here to avoid dependency issues
                    import numpy as np
                    video_path = self.create_sample_video(video_name, duration=15)
                    if video_path:
                        downloaded_videos.append(video_path)
                except ImportError:
                    print("⚠️ numpy not available, creating simple video")
                    # Create very simple video without numpy
                    self.create_simple_video(video_name)

        # Verify all videos
        print(f"\n✅ Available test videos: {len(downloaded_videos)}")
        valid_videos = []
        for video_path in downloaded_videos:
            if self.verify_video(video_path):
                valid_videos.append(video_path)

        if valid_videos:
            print(f"\n🧪 Ready to test with {len(valid_videos)} videos!")
            print("Run these commands to test:")
            print("  python src/integrated_detector.py")
            print("  python src/test_vietnamese_models.py")
        else:
            print("\n⚠️ No valid videos available.")
            print("Please manually download videos from:")
            print("  - https://pixabay.com/videos/search/traffic%20vietnam/")
            print("  - https://www.pexels.com/search/videos/driving/")

        return valid_videos

    def create_simple_video(self, filename):
        """Create simple video without numpy dependency"""
        video_path = self.video_dir / filename

        if video_path.exists():
            return str(video_path)

        print(f"🎬 Creating simple test video: {filename}")

        # Create minimal test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        for i in range(300):  # 10 seconds at 30 FPS
            # Create simple colored frame
            frame = [[[(50 + i) % 255, 100, 150] for _ in range(640)] for _ in range(480)]
            frame = cv2.UMat(frame).get()
            out.write(frame)

        out.release()
        return str(video_path)

def main():
    """Main function"""
    downloader = AutoVideoDownloader()
    videos = downloader.run_auto_download()

    if videos:
        print(f"\n🎯 SUCCESS: {len(videos)} test videos ready!")
    else:
        print("\n📝 Please check manual download instructions in video/download_instructions.md")

if __name__ == "__main__":
    main()