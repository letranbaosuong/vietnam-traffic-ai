#!/usr/bin/env python3
import cv2
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter

print('=' * 70)
print('TRAFFIC OBJECT DETECTION TEST - Simple Version')
print('=' * 70)

# Load COCO labels
print('\n[1/5] Loading labels...')
with open('coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f'✓ Loaded {len(labels)} labels')

# Load image
print('\n[2/5] Loading image...')
image = cv2.imread('traffic_test.jpg')
if image is None:
    print('ERROR: Cannot load traffic_test.jpg')
    exit(1)
orig_image = image.copy()
img_height, img_width = image.shape[:2]
print(f'✓ Image loaded: {img_width}x{img_height}')

# Load model
print('\n[3/5] Loading TFLite model...')
interpreter = Interpreter(model_path='efficientdet_lite0.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
print(f'✓ Model loaded! Expected input: {input_shape}')

# Prepare input
print('\n[4/5] Preparing input...')
height, width = input_shape[1], input_shape[2]
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Check if model expects float32 input
if input_details[0]['dtype'] == np.float32:
    input_data = input_data.astype(np.float32) / 255.0
elif input_details[0]['dtype'] == np.uint8:
    input_data = input_data.astype(np.uint8)

print(f'✓ Input prepared: {input_data.shape}, dtype: {input_data.dtype}')

# Run inference
print('\n[5/5] Running inference...')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
print('✓ Inference completed!')

# Get outputs
print('\nProcessing results...')
# EfficientDet outputs: [scores, boxes, num_detections, classes]
all_scores = interpreter.get_tensor(output_details[0]['index'])[0]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]

# Get best class for each detection
scores = np.max(all_scores, axis=1)
classes = np.argmax(all_scores, axis=1)

# Visualize detections
threshold = 0.4
detections = []
traffic_related = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train',
                   'truck', 'traffic light', 'stop sign', 'parking meter']

print('\n' + '=' * 70)
print('DETECTION RESULTS')
print('=' * 70)

for i in range(len(scores)):
    if scores[i] > threshold:
        # Get box coordinates (EfficientDet: [cy, cx, h, w] normalized)
        cy, cx, h, w = boxes[i]

        # Convert to corners
        ymin = cy - h / 2
        xmin = cx - w / 2
        ymax = cy + h / 2
        xmax = cx + w / 2

        # Convert to pixel coordinates
        left = int(xmin * img_width)
        top = int(ymin * img_height)
        right = int(xmax * img_width)
        bottom = int(ymax * img_height)

        # Clamp to bounds
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(0, min(right, img_width))
        bottom = max(0, min(bottom, img_height))

        class_id = int(classes[i])
        label = labels[class_id] if class_id < len(labels) else f"Class {class_id}"

        detections.append({
            'label': label,
            'score': scores[i],
            'box': [left, top, right, bottom],
            'is_traffic': label.lower() in traffic_related
        })

        # Choose color
        color = (0, 0, 255) if label.lower() in traffic_related else (0, 255, 0)

        # Draw box
        cv2.rectangle(orig_image, (left, top), (right, bottom), color, 3)

        # Draw label
        label_text = f"{label}: {scores[i]:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(orig_image, (left, top - text_height - 10),
                     (left + text_width, top), color, -1)
        cv2.putText(orig_image, label_text, (left, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Print results
if detections:
    traffic_objs = [d for d in detections if d['is_traffic']]
    other_objs = [d for d in detections if not d['is_traffic']]

    if traffic_objs:
        print(f'\n🚦 TRAFFIC-RELATED OBJECTS: {len(traffic_objs)}')
        for i, det in enumerate(traffic_objs, 1):
            print(f'  {i}. {det["label"].upper()}: {det["score"]:.1%} confidence')
            print(f'     Box: x={det["box"][0]}, y={det["box"][1]}, '
                  f'w={det["box"][2]-det["box"][0]}, h={det["box"][3]-det["box"][1]}')

    if other_objs:
        print(f'\n📦 OTHER OBJECTS: {len(other_objs)}')
        for i, det in enumerate(other_objs, 1):
            print(f'  {i}. {det["label"]}: {det["score"]:.1%}')

    print(f'\n📊 SUMMARY:')
    print(f'  • Total detections: {len(detections)}')
    print(f'  • Traffic-related: {len(traffic_objs)}')
    print(f'  • Other objects: {len(other_objs)}')
    print(f'  • Threshold: {threshold}')
else:
    print(f'\n⚠️  No objects detected (threshold: {threshold})')

# Save result
output_path = 'traffic_detection_result.jpg'
cv2.imwrite(output_path, orig_image)
print(f'\n💾 Result saved to: {output_path}')

print('\n' + '=' * 70)
print('✅ TEST COMPLETED SUCCESSFULLY!')
print('=' * 70)
