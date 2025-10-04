import cv2
import numpy as np
import tensorflow as tf

# Load COCO labels
with open('coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print('Loading image...')
image = cv2.imread('test_data/table.jpg')
orig_image = image.copy()
print(f'Image shape: {image.shape}')

print('Loading model...')
interpreter = tf.lite.Interpreter(model_path='efficientdet_lite0.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'Model loaded! Input shape: {input_details[0]["shape"]}')

# Prepare input
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

print('\nRunning inference...')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
all_scores = interpreter.get_tensor(output_details[0]['index'])[0]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]

# Get the best class for each box
scores = np.max(all_scores, axis=1)
classes = np.argmax(all_scores, axis=1)

print(f'Inference complete!')

# Debug: show a few boxes
print(f'\nDebug - First 3 boxes:')
for i in range(min(3, len(boxes))):
    print(f'  Box {i}: {boxes[i]} (score: {scores[i]:.3f})')

# Visualize detections
threshold = 0.5
img_height, img_width, _ = orig_image.shape
detections = 0

print(f'\nDetections (threshold > {threshold}):')
for i in range(len(scores)):
    if scores[i] > threshold:
        detections += 1

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

        print(f'{detections}. {label_name}: {scores[i]:.3f} at [{left}, {top}, {right}, {bottom}]')

        # Draw bounding box
        cv2.rectangle(orig_image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label
        label_text = f"{label_name}: {scores[i]:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(orig_image, (left, top - label_size[1] - 10),
                     (left + label_size[0], top), (0, 255, 0), -1)
        cv2.putText(orig_image, label_text, (left, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

print(f'\nTotal detections: {detections}')

# Save result
output_path = 'detection_result.jpg'
cv2.imwrite(output_path, orig_image)
print(f'Result saved to: {output_path}')

print('\n✅ Test completed successfully!')
