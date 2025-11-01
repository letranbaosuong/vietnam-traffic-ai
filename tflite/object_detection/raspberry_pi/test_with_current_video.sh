#!/bin/bash

echo "=================================================="
echo "Testing Driver Gesture Detection System"
echo "=================================================="
echo ""

# Test với video hiện có
echo "1. Testing với car-driver.mp4 (góc phía sau)..."
source venv/bin/activate

# Run simulation demo
python3 demo_gesture_simulation.py test_videos/car-driver.mp4 --save

echo ""
echo "Output: driver_gesture_demo_output.mp4"
echo ""
echo "Để xem kết quả:"
echo "  open driver_gesture_demo_output.mp4"
echo ""
echo "=================================================="
