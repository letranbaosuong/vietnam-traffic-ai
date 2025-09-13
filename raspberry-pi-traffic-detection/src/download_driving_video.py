#!/usr/bin/env python3
"""
Download Free Driving Videos for Testing
Tải video lái xe miễn phí để test hệ thống
"""

import os
import urllib.request
from pathlib import Path
import cv2

class DrivingVideoDownloader:
    def __init__(self):
        """Initialize video downloader"""
        self.video_dir = Path("../video")
        self.video_dir.mkdir(exist_ok=True)

        # Free video sources (sample URLs)
        self.video_sources = {
            "vietnam_traffic_1": {
                "name": "Vietnam Traffic 1",
                "description": "Vietnam street traffic footage",
                "format": "mp4",
                "source": "pixabay"
            },
            "dashcam_driving": {
                "name": "Dashcam Driving",
                "description": "First-person driving view",
                "format": "mp4",
                "source": "pexels"
            },
            "vietnam_motorbike": {
                "name": "Vietnam Motorbike",
                "description": "Motorbike perspective Vietnam",
                "format": "mp4",
                "source": "vecteezy"
            }
        }

    def list_free_sources(self):
        """List all free video download sources"""
        print("🎬 Free Driving Video Sources for Testing")
        print("=" * 50)

        print("\n📺 **Top Platforms:**")
        print("1. **Pixabay**: https://pixabay.com/videos/search/traffic%20vietnam/")
        print("   - 2,314+ free Vietnam traffic videos")
        print("   - 4K & HD quality, royalty-free")
        print("   - No attribution required")

        print("\n2. **Pexels**: https://www.pexels.com/search/videos/dash%20cam/")
        print("   - 1,454+ dashcam stock videos")
        print("   - Free to use, high-quality HD")
        print("   - Great for first-person driving view")

        print("\n3. **Vecteezy**: https://www.vecteezy.com/free-videos/vietnam-traffic")
        print("   - 729+ Vietnam traffic stock videos")
        print("   - 4,939+ general Vietnam videos")
        print("   - Royalty-free download")

        print("\n4. **Videvo**: https://www.videvo.net/stock-video-footage/dashcam/")
        print("   - 428+ free dashcam videos")
        print("   - 4K and HD quality")
        print("   - Personal/commercial use")

    def suggest_search_terms(self):
        """Suggest search terms for finding good test videos"""
        print("\n🔍 **Recommended Search Terms:**")
        print("- 'vietnam traffic dashcam'")
        print("- 'hanoi street driving'")
        print("- 'saigon traffic motorcycle'")
        print("- 'vietnam road first person'")
        print("- 'asian traffic driving'")
        print("- 'urban traffic vietnam'")
        print("- 'motorbike pov vietnam'")

    def download_sample_video(self, url, filename):
        """Download a video from URL"""
        video_path = self.video_dir / filename

        if video_path.exists():
            print(f"✅ Video already exists: {filename}")
            return str(video_path)

        try:
            print(f"📥 Downloading {filename}...")
            urllib.request.urlretrieve(url, video_path)
            print(f"✅ Downloaded: {filename}")
            return str(video_path)

        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return None

    def verify_video(self, video_path):
        """Verify downloaded video is valid"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return False

            # Get video info
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            print(f"📺 Video Info: {width}x{height}, {fps} FPS, {duration:.1f}s")

            cap.release()
            return True

        except Exception as e:
            print(f"❌ Error verifying video: {e}")
            return False

    def create_test_instructions(self):
        """Create instructions for manual download"""
        instructions_file = self.video_dir / "download_instructions.md"

        content = """# Download Instructions - Free Driving Videos

## 🎯 Best Videos for Traffic Detection Testing

### 1. Vietnam Traffic Videos (Pixabay)
**URL**: https://pixabay.com/videos/search/traffic%20vietnam/

**Recommended videos:**
- Search: "vietnam traffic street"
- Look for: HD quality (720p+), 10-30 seconds
- Download as: `vietnam_street_traffic.mp4`

### 2. Dashcam Videos (Pexels)
**URL**: https://www.pexels.com/search/videos/dash%20cam/

**Recommended videos:**
- Search: "driving dashboard camera"
- Look for: First-person view, clear road
- Download as: `dashcam_driving.mp4`

### 3. Vietnam Motorbike (Vecteezy)
**URL**: https://www.vecteezy.com/free-videos/vietnam

**Recommended videos:**
- Search: "vietnam motorbike traffic"
- Look for: Motorbike POV, busy traffic
- Download as: `vietnam_motorbike_pov.mp4`

## 📋 Download Steps:

1. Visit the URLs above
2. Search using recommended terms
3. Select HD quality videos (720p or higher)
4. Download and save to `video/` directory
5. Rename files as suggested above

## 🧪 Testing:

After download, test with:
```bash
python src/integrated_detector.py
python src/test_vietnamese_models.py
```

## 📝 Notes:

- Choose videos with clear traffic signs visible
- Prefer 10-60 second clips for testing
- Ensure good lighting conditions
- Vietnam/Asian traffic preferred for local relevance
"""

        with open(instructions_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"📝 Instructions saved: {instructions_file}")

    def run_download_guide(self):
        """Run complete download guidance"""
        print("🎬 Free Driving Video Download Guide")
        print("=" * 50)

        # List sources
        self.list_free_sources()

        # Suggest search terms
        self.suggest_search_terms()

        # Create instructions
        self.create_test_instructions()

        print("\n" + "=" * 50)
        print("📋 NEXT STEPS:")
        print("1. ✅ Review free video sources above")
        print("2. 📥 Manually download 2-3 test videos")
        print("3. 📁 Save to video/ directory")
        print("4. 🧪 Run: python src/integrated_detector.py")
        print("5. 📊 Run: python src/test_vietnamese_models.py")

        print("\n💡 TIPS:")
        print("- Choose HD quality (720p+)")
        print("- Pick videos with visible traffic signs")
        print("- Vietnam/Asian traffic preferred")
        print("- 10-60 second clips work best")

def main():
    """Main function"""
    downloader = DrivingVideoDownloader()
    downloader.run_download_guide()

if __name__ == "__main__":
    main()