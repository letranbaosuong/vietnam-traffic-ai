#!/usr/bin/env python3
"""Test pose estimation on traffic image to detect people's poses."""
import sys
import cv2
import numpy as np

sys.path.insert(0, '/Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/pose_estimation/raspberry_pi')

from ml import Movenet
from data import Person, BodyPart
import utils as pose_utils

print('=' * 70)
print('POSE ESTIMATION TEST - Traffic Image')
print('=' * 70)

# Load traffic image (copy from object detection test)
print('\n[1/5] Loading traffic image...')
traffic_image_path = '/Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/object_detection/raspberry_pi/traffic_test.jpg'
image = cv2.imread(traffic_image_path)
if image is None:
    print(f'ERROR: Cannot load image from {traffic_image_path}')
    sys.exit(1)

orig_image = image.copy()
img_height, img_width = image.shape[:2]
print(f'✓ Image loaded: {img_width}x{img_height}')

# Initialize MoveNet Lightning (faster, good for multi-person scenarios)
print('\n[2/5] Loading MoveNet Lightning model...')
try:
    pose_detector = Movenet('movenet_lightning')
    print('✓ Model loaded successfully!')
except Exception as e:
    print(f'ERROR loading model: {e}')
    sys.exit(1)

# Run pose detection
print('\n[3/5] Running pose estimation...')
try:
    person = pose_detector.detect(image)
    print(f'✓ Pose detection completed!')
    print(f'  - Detected {len(person.keypoints)} keypoints')
    print(f'  - Overall confidence: {person.score:.2%}')
except Exception as e:
    print(f'ERROR during detection: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Analyze detected keypoints
print('\n[4/5] Analyzing detected pose...')
keypoint_threshold = 0.2
detected_parts = []
low_confidence_parts = []

# Key body parts for traffic safety analysis
critical_parts = {
    BodyPart.NOSE: 'Head',
    BodyPart.LEFT_SHOULDER: 'Left Shoulder',
    BodyPart.RIGHT_SHOULDER: 'Right Shoulder',
    BodyPart.LEFT_HIP: 'Left Hip',
    BodyPart.RIGHT_HIP: 'Right Hip',
    BodyPart.LEFT_KNEE: 'Left Knee',
    BodyPart.RIGHT_KNEE: 'Right Knee',
}

print('\n' + '=' * 70)
print('KEYPOINT ANALYSIS')
print('=' * 70)

print(f'\n📍 DETECTED KEYPOINTS (threshold: {keypoint_threshold}):')
for kp in person.keypoints:
    if kp.score >= keypoint_threshold:
        detected_parts.append(kp.body_part.name)
        part_name = critical_parts.get(kp.body_part, kp.body_part.name)
        print(f'  ✓ {part_name:20s}: {kp.score:.1%} at ({kp.coordinate.x}, {kp.coordinate.y})')
    else:
        low_confidence_parts.append(kp.body_part.name)

if low_confidence_parts:
    print(f'\n⚠️  LOW CONFIDENCE PARTS (below {keypoint_threshold}):')
    for i, part in enumerate(low_confidence_parts[:5], 1):
        print(f'  {i}. {part}')

# Draw pose on image
print('\n[5/5] Visualizing pose...')
list_persons = [person]
output_image = pose_utils.visualize(
    orig_image,
    list_persons,
    keypoint_threshold=keypoint_threshold
)

# Add analysis overlay
print('\n' + '=' * 70)
print('POSE ESTIMATION SUMMARY')
print('=' * 70)

# Determine if person is in a standing/walking/riding pose
head_detected = any(kp.body_part in [BodyPart.NOSE, BodyPart.LEFT_EYE, BodyPart.RIGHT_EYE]
                    and kp.score >= keypoint_threshold for kp in person.keypoints)
torso_detected = any(kp.body_part in [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER,
                                       BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP]
                     and kp.score >= keypoint_threshold for kp in person.keypoints)
legs_detected = any(kp.body_part in [BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE,
                                      BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE]
                    and kp.score >= keypoint_threshold for kp in person.keypoints)

print(f'\n🚶 BODY STRUCTURE DETECTION:')
print(f'  • Head detected: {"✓ Yes" if head_detected else "✗ No"}')
print(f'  • Torso detected: {"✓ Yes" if torso_detected else "✗ No"}')
print(f'  • Legs detected: {"✓ Yes" if legs_detected else "✗ No"}')

print(f'\n📊 STATISTICS:')
print(f'  • Total keypoints: {len(person.keypoints)}')
print(f'  • Detected (>{keypoint_threshold}): {len(detected_parts)}')
print(f'  • Low confidence: {len(low_confidence_parts)}')
print(f'  • Overall pose score: {person.score:.2%}')

# Determine pose quality
if person.score > 0.5:
    quality = "🟢 EXCELLENT - Full body visible"
elif person.score > 0.3:
    quality = "🟡 GOOD - Partial visibility"
elif person.score > 0.15:
    quality = "🟠 FAIR - Limited visibility"
else:
    quality = "🔴 POOR - Difficult to detect"

print(f'\n🎯 POSE QUALITY: {quality}')

# Application for traffic monitoring
print(f'\n🚦 TRAFFIC MONITORING CONTEXT:')
if head_detected and torso_detected:
    print(f'  • Person clearly visible in traffic scene')
    print(f'  • Suitable for pedestrian tracking')
    if legs_detected:
        print(f'  • Full body pose available for gait analysis')
else:
    print(f'  • Limited visibility - may be inside vehicle or partially occluded')

# Save result
output_path = '/Users/letranbaosuong/Documents/projects/utils/vietnam-traffic-ai/tflite/pose_estimation/raspberry_pi/traffic_pose_result.jpg'
cv2.imwrite(output_path, output_image)
print(f'\n💾 Result saved to: traffic_pose_result.jpg')

print('\n' + '=' * 70)
print('✅ POSE ESTIMATION TEST COMPLETED!')
print('=' * 70)
