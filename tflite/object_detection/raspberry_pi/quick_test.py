import cv2
import numpy as np
import tensorflow as tf

print('Loading image...')
image = cv2.imread('test_data/table.jpg')
print(f'Image shape: {image.shape}')

print('Loading model...')
interpreter = tf.lite.Interpreter(model_path='efficientdet_lite0.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'Input shape: {input_details[0]["shape"]}')
print(f'Model loaded successfully!')

# Prepare input
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height))
# Normalize to [0, 1]
input_data = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

print(f'Number of outputs: {len(output_details)}')
for i, detail in enumerate(output_details):
    print(f'Output {i}: {detail["name"]}, shape: {detail["shape"]}, dtype: {detail["dtype"]}')

print('\nRunning inference...')
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output - format is [scores, boxes]
# scores shape: [1, num_boxes, num_classes]
# boxes shape: [1, num_boxes, 4]
all_scores = interpreter.get_tensor(output_details[0]['index'])[0]  # [num_boxes, num_classes]
boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # [num_boxes, 4]

# Get the best class for each box
scores = np.max(all_scores, axis=1)  # Max score across all classes
classes = np.argmax(all_scores, axis=1)  # Class with max score

print(f'Detection complete!')
print(f'Max score: {np.max(scores):.6f}')
print(f'Top 10 scores: {sorted(scores, reverse=True)[:10]}')

threshold = 0.01  # Lower threshold for testing
detections = sum(1 for score in scores if score > threshold)
print(f'\nFound {detections} objects with confidence > {threshold}')

# Show top detections
top_indices = np.argsort(scores)[::-1][:10]  # Top 10
for idx in top_indices:
    if scores[idx] > threshold:
        print(f'  Object: Class {int(classes[idx])}, Score: {scores[idx]:.6f}')

print('SUCCESS!')
