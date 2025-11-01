#!/bin/bash

echo "================================================================"
echo "📥 DRIVER FRONTAL VIEW VIDEO DOWNLOADER"
echo "================================================================"
echo ""

# Create downloads directory
mkdir -p downloads
cd downloads

echo "🔍 Found 3 Free Datasets with Frontal View Videos:"
echo ""
echo "1. D3S Dataset (Google Drive)"
echo "   - Videos of drivers: eye close, yawning, neutral states"
echo "   - Frontal camera view"
echo "   - Link: https://drive.google.com/file/d/1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view"
echo ""
echo "2. YawDD Dataset (Direct Download)"
echo "   - 107 participants (male/female)"
echo "   - Dashboard camera, frontal view"
echo "   - Yawning, talking, normal expressions"
echo "   - Link: http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar"
echo ""
echo "3. VBDDD Dataset (Baidu Pan - Chinese)"
echo "   - 558 video samples (3s-50s each)"
echo "   - 640x480 resolution, 30 FPS"
echo "   - Link: https://pan.baidu.com/s/1qxRKT_ydBDVpCE5-OSgP2Q?pwd=4kna"
echo ""
echo "================================================================"
echo ""

# Option to download YawDD (direct link)
echo "💡 Recommendation: Try YawDD dataset (has direct download link)"
echo ""
read -p "Download YawDD dataset now? (~500MB) [y/N]: " choice

if [[ "$choice" =~ ^[Yy]$ ]]; then
    echo ""
    echo "📥 Downloading YawDD dataset..."
    echo "This may take several minutes depending on your connection..."

    # Download YawDD
    curl -L -o YawDD.rar "http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar"

    if [ -f "YawDD.rar" ]; then
        echo "✅ Download complete!"
        echo ""
        echo "📦 To extract (requires unrar):"
        echo "   brew install unrar  # macOS"
        echo "   unrar x YawDD.rar"
        echo ""
        echo "Then copy sample videos to test_videos/"
    else
        echo "❌ Download failed. Please try manual download:"
        echo "   http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar"
    fi
else
    echo ""
    echo "================================================================"
    echo "📋 MANUAL DOWNLOAD INSTRUCTIONS:"
    echo "================================================================"
    echo ""
    echo "Option 1 - D3S Dataset (Google Drive):"
    echo "  1. Open: https://drive.google.com/file/d/1r27hqFlvznT8f7FyV7ipUtfOJ2nio_LA/view"
    echo "  2. Click 'Download' button"
    echo "  3. Extract and copy videos to test_videos/"
    echo ""
    echo "Option 2 - YawDD Dataset (Direct):"
    echo "  1. Download: http://www.discover.uottawa.ca/images/files/external/YawDD_Dataset/YawDD.rar"
    echo "  2. Extract .rar file"
    echo "  3. Copy sample videos to test_videos/"
    echo ""
    echo "Option 3 - Use Webcam (Fastest!):"
    echo "  python3 record_test_video.py 30"
    echo ""
    echo "================================================================"
fi

cd ..

echo ""
echo "✅ Done!"
echo ""
