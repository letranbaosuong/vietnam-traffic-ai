#!/bin/bash

# Demo script to test lane detection on sample videos
echo "=========================================="
echo "Lane Detection Demo Script"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Check if test videos exist
if [ ! -d "test_videos" ]; then
    echo "ERROR: test_videos folder not found!"
    exit 1
fi

# Test 1: solidWhiteRight.mp4 (smaller, standard lanes)
echo "Test 1: Standard white lanes (solidWhiteRight.mp4)"
echo "------------------------------------------"
if [ -f "test_videos/solidWhiteRight.mp4" ]; then
    echo "Running lane detection..."
    python3 test_lane_detection.py \
        --mode video \
        --source test_videos/solidWhiteRight.mp4 \
        --save
    echo ""
    echo "Output saved to: lane_detection_output.mp4"
    echo ""
else
    echo "WARNING: solidWhiteRight.mp4 not found, skipping..."
fi

read -p "Press Enter to continue to next test..."

# Test 2: detect_video_danang.mp4 (Vietnam traffic)
echo ""
echo "Test 2: Vietnam traffic - Danang (detect_video_danang.mp4)"
echo "------------------------------------------"
if [ -f "test_videos/detect_video_danang.mp4" ]; then
    echo "Running lane detection on Vietnam traffic..."
    python3 test_lane_detection.py \
        --mode video \
        --source test_videos/detect_video_danang.mp4 \
        --save
    echo ""
    echo "Output saved to: lane_detection_output.mp4"
    echo ""
else
    echo "WARNING: detect_video_danang.mp4 not found, skipping..."
fi

echo ""
echo "=========================================="
echo "Demo completed!"
echo "=========================================="
echo ""
echo "Tips:"
echo "  - Press 'q' or ESC to stop processing"
echo "  - Output videos are saved as 'lane_detection_output.mp4'"
echo "  - To test with camera: python3 test_lane_detection.py --mode camera"
echo ""
