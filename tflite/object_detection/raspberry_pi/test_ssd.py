import cv2
import numpy as np
import tensorflow as tf

# Load labels
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print('Loading image...')
image = cv2.imread('test_data/table.jpg')
orig_image = image.copy()
img_height, img_width, _ = image.shape
print(f'Image shape: {image.shape}')

print('\nLoading SSD MobileNet model...')
interpreter = tf.lite.Interpreter(model_path='detect.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'Input details: {input_details[0]["shape"]}, dtype: {input_details[0]["dtype"]}')
print(f'Number of outputs: {len(output_details)}')
for i, detail in enumerate(output_details):
    print(f'  Output {i}: shape {detail["shape"]}, dtype: {detail["dtype"]}')

# Prepare input
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

# Check if quantized or float
if input_details[0]['dtype'] == np.uint8:
    input_data = input_data.astype(np.uint8)
    print('Using UINT8 input')
else:
    input_data = input_data.astype(np.float32) / 255.0
    print('Using FLOAT32 input (normalized)')

print('\nRunning inference...')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# SSD models typically output: boxes, classes, scores, num_detections
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0]) if len(output_details) > 3 else len(scores)

print(f'Inference complete! Found {num_detections} detections')
print(f'All scores: {scores}')
print(f'All classes: {classes}')
print(f'First box: {boxes[0]}')

# Visualize detections
threshold = 0.3  # Lower threshold
print(f'\nDetections (confidence > {threshold}):')

detection_count = 0
for i in range(min(num_detections, len(scores))):
    if scores[i] > threshold:
        detection_count += 1

        # Get box coordinates (ymin, xmin, ymax, xmax in normalized coordinates)
        ymin, xmin, ymax, xmax = boxes[i]

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
        label_name = labels[class_id + 1] if class_id + 1 < len(labels) else f"Class {class_id}"  # +1 because COCO SSD has background as 0

        print(f'{detection_count}. {label_name}: {scores[i]:.3f} at [{left}, {top}, {right}, {bottom}]')

        # Draw bounding box
        cv2.rectangle(orig_image, (left, top), (right, bottom), (0, 255, 0), 3)

        # Draw label
        label_text = f"{label_name}: {scores[i]:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(orig_image, (left, top - label_size[1] - 10),
                     (left + label_size[0] + 5, top), (0, 255, 0), -1)
        cv2.putText(orig_image, label_text, (left + 2, top - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

print(f'\nTotal valid detections: {detection_count}')

# Save result
output_path = 'ssd_detection_result.jpg'
cv2.imwrite(output_path, orig_image)
print(f'Result saved to: {output_path}')

print('\n✅ SSD Detection test completed successfully!')
