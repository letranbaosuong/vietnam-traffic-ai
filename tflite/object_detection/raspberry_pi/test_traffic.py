import cv2
import numpy as np
import tensorflow as tf

# Load COCO labels
with open('coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print('=' * 60)
print('TRAFFIC OBJECT DETECTION TEST')
print('=' * 60)

print('\n[1/5] Loading image...')
image = cv2.imread('traffic_test.jpg')
if image is None:
    print('ERROR: Cannot load image!')
    exit(1)

orig_image = image.copy()
print(f'✓ Image loaded successfully! Shape: {image.shape}')

print('\n[2/5] Loading TFLite model...')
interpreter = tf.lite.Interpreter(model_path='efficientdet_lite0.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f'✓ Model loaded! Input shape: {input_details[0]["shape"]}')

print('\n[3/5] Preparing input...')
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0
print(f'✓ Input prepared: {input_data.shape}')

print('\n[4/5] Running inference...')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
all_scores = interpreter.get_tensor(output_details[0]['index'])[0]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]

# Get the best class for each box
scores = np.max(all_scores, axis=1)
classes = np.argmax(all_scores, axis=1)
print(f'✓ Inference completed!')

print('\n[5/5] Processing detections...')
# Visualize detections
threshold = 0.4
img_height, img_width, _ = orig_image.shape
detections = []

for i in range(len(scores)):
    if scores[i] > threshold:
        # Get box coordinates
        # EfficientDet format: [cy, cx, h, w] in relative coordinates
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

        # Clamp to image bounds
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(0, min(right, img_width))
        bottom = max(0, min(bottom, img_height))

        class_id = int(classes[i])
        label_name = labels[class_id] if class_id < len(labels) else f"Class {class_id}"

        detections.append({
            'label': label_name,
            'score': scores[i],
            'box': [left, top, right, bottom]
        })

        # Draw bounding box
        color = (0, 255, 0)  # Green for general objects
        # Red for traffic-related objects
        if label_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'stop sign']:
            color = (0, 0, 255)  # Red

        cv2.rectangle(orig_image, (left, top), (right, bottom), color, 3)

        # Draw label
        label_text = f"{label_name}: {scores[i]:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(orig_image, (left, top - label_size[1] - 10),
                     (left + label_size[0], top), color, -1)
        cv2.putText(orig_image, label_text, (left, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

print('\n' + '=' * 60)
print('DETECTION RESULTS')
print('=' * 60)

if len(detections) > 0:
    # Group by category
    traffic_objects = []
    other_objects = []

    for det in detections:
        if det['label'] in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'stop sign', 'person']:
            traffic_objects.append(det)
        else:
            other_objects.append(det)

    if traffic_objects:
        print(f'\n🚦 TRAFFIC-RELATED OBJECTS ({len(traffic_objects)}):')
        for i, det in enumerate(traffic_objects, 1):
            print(f'  {i}. {det["label"].upper()}: {det["score"]:.2%} confidence')
            print(f'     Box: [{det["box"][0]}, {det["box"][1]}, {det["box"][2]}, {det["box"][3]}]')

    if other_objects:
        print(f'\n📦 OTHER OBJECTS ({len(other_objects)}):')
        for i, det in enumerate(other_objects, 1):
            print(f'  {i}. {det["label"]}: {det["score"]:.2%}')

    print(f'\n📊 SUMMARY:')
    print(f'  - Total objects detected: {len(detections)}')
    print(f'  - Traffic-related: {len(traffic_objects)}')
    print(f'  - Other objects: {len(other_objects)}')
else:
    print('\n⚠️  No objects detected above threshold ({threshold})')

# Save result
output_path = 'traffic_detection_result.jpg'
cv2.imwrite(output_path, orig_image)
print(f'\n💾 Result saved to: {output_path}')

print('\n' + '=' * 60)
print('✅ TEST COMPLETED SUCCESSFULLY!')
print('=' * 60)
