import cv2
import numpy as np
import random
from typing import List, Tuple, Dict
import albumentations as A
from pathlib import Path
import json

class TrafficDataAugmentor:
    """
    Data augmentation dành riêng cho dữ liệu giao thông Việt Nam
    """
    
    def __init__(self):
        # Augmentation pipeline for Vietnam traffic conditions
        self.transform = A.Compose([
            # Weather conditions common in Vietnam
            A.OneOf([
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, 
                           drop_width=1, drop_color=(200, 200, 200), 
                           blur_value=7, brightness_coefficient=0.7, 
                           rain_type=None, p=0.3),
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, alpha_coef=0.08, p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.1),
            ], p=0.4),
            
            # Lighting conditions (common in urban Vietnam)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
            ], p=0.5),
            
            # Traffic density simulation
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),  # Moving vehicles
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
            ], p=0.3),
            
            # Camera variations
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.1),
                A.Perspective(scale=(0.05, 0.1), p=0.1),
            ], p=0.2),
            
            # Color variations (different camera settings)
            A.OneOf([
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.2),
            ], p=0.3),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # Severe weather augmentation for robustness
        self.severe_weather = A.Compose([
            A.OneOf([
                A.RandomRain(slant_lower=-20, slant_upper=20, drop_length=30, 
                           drop_width=2, drop_color=(150, 150, 150), 
                           blur_value=10, brightness_coefficient=0.5, p=0.5),
                A.RandomFog(fog_coef_lower=0.6, fog_coef_upper=1.0, alpha_coef=0.1, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), 
                                         contrast_limit=0.2, p=0.2),  # Dark/night
            ], p=1.0),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_image_with_boxes(self, image: np.ndarray, bboxes: List[List], 
                                class_labels: List[int], severity: str = 'normal') -> Tuple[np.ndarray, List[List], List[int]]:
        """
        Augment image with bounding boxes
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in YOLO format [center_x, center_y, width, height]
            class_labels: List of class labels for each bbox
            severity: 'normal' or 'severe' weather conditions
        """
        
        # Convert to albumentations format if needed
        if len(bboxes) > 0 and len(bboxes[0]) == 4:
            # Already in YOLO format
            pass
        else:
            # Convert from other formats if needed
            bboxes = self._convert_to_yolo_format(bboxes, image.shape)
        
        try:
            if severity == 'severe':
                transformed = self.severe_weather(image=image, bboxes=bboxes, class_labels=class_labels)
            else:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, bboxes, class_labels
    
    def _convert_to_yolo_format(self, bboxes: List, image_shape: Tuple) -> List[List]:
        """Convert bounding boxes to YOLO format"""
        h, w = image_shape[:2]
        yolo_bboxes = []
        
        for bbox in bboxes:
            if len(bbox) == 4:  # Assuming [x1, y1, x2, y2] format
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2 / w
                center_y = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                yolo_bboxes.append([center_x, center_y, width, height])
            else:
                yolo_bboxes.append(bbox)  # Assume already in YOLO format
        
        return yolo_bboxes
    
    def create_vietnam_specific_augmentations(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create Vietnam-specific traffic augmentations
        """
        augmented_images = []
        
        # 1. Motorbike-heavy traffic simulation (blur background, keep foreground sharp)
        motorcycle_traffic = self._simulate_motorcycle_density(image)
        augmented_images.append(motorcycle_traffic)
        
        # 2. Urban pollution/haze effect
        hazy = self._add_urban_haze(image)
        augmented_images.append(hazy)
        
        # 3. Monsoon rain effect
        rainy = self._add_monsoon_rain(image)
        augmented_images.append(rainy)
        
        # 4. Hot weather glare
        glare = self._add_tropical_glare(image)
        augmented_images.append(glare)
        
        return augmented_images
    
    def _simulate_motorcycle_density(self, image: np.ndarray) -> np.ndarray:
        """Simulate high motorcycle density with motion blur"""
        # Add subtle motion blur to simulate movement
        kernel_size = random.randint(3, 7)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Horizontal motion blur (common direction)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        blurred = cv2.filter2D(image, -1, kernel)
        
        # Mix with original
        alpha = random.uniform(0.3, 0.7)
        result = cv2.addWeighted(image, alpha, blurred, 1 - alpha, 0)
        
        return result
    
    def _add_urban_haze(self, image: np.ndarray) -> np.ndarray:
        """Add urban pollution haze effect"""
        haze_color = np.random.randint(200, 255, 3, dtype=np.uint8)
        haze_intensity = random.uniform(0.1, 0.3)
        
        haze_overlay = np.full_like(image, haze_color, dtype=np.uint8)
        hazy_image = cv2.addWeighted(image, 1 - haze_intensity, haze_overlay, haze_intensity, 0)
        
        return hazy_image
    
    def _add_monsoon_rain(self, image: np.ndarray) -> np.ndarray:
        """Add heavy monsoon rain effect"""
        h, w = image.shape[:2]
        
        # Create rain effect
        rain_intensity = random.randint(1000, 3000)
        rain_drops = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(rain_intensity):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(5, 15)
            
            # Draw rain drop
            end_y = min(h - 1, y + length)
            cv2.line(rain_drops, (x, y), (x, end_y), 255, 1)
        
        # Convert to 3 channel
        rain_drops_color = cv2.cvtColor(rain_drops, cv2.COLOR_GRAY2BGR)
        
        # Add rain to image
        rainy_image = cv2.addWeighted(image, 0.8, rain_drops_color, 0.2, 0)
        
        # Reduce overall brightness
        rainy_image = cv2.convertScaleAbs(rainy_image, alpha=0.7, beta=10)
        
        return rainy_image
    
    def _add_tropical_glare(self, image: np.ndarray) -> np.ndarray:
        """Add tropical sun glare effect"""
        h, w = image.shape[:2]
        
        # Create glare center (usually top part of image)
        glare_center_x = random.randint(w // 4, 3 * w // 4)
        glare_center_y = random.randint(0, h // 3)
        
        # Create radial gradient for glare
        y, x = np.ogrid[:h, :w]
        mask = (x - glare_center_x) ** 2 + (y - glare_center_y) ** 2
        mask = np.sqrt(mask)
        
        # Normalize and invert
        max_dist = np.sqrt(h**2 + w**2) / 2
        glare_mask = 1 - (mask / max_dist)
        glare_mask = np.clip(glare_mask, 0, 1)
        
        # Apply glare intensity
        glare_intensity = random.uniform(0.2, 0.5)
        glare_mask = glare_mask * glare_intensity
        
        # Convert to 3 channels
        glare_mask = np.stack([glare_mask] * 3, axis=-1)
        
        # Add bright overlay
        bright_overlay = np.full_like(image, 255, dtype=np.uint8)
        glare_image = image.astype(np.float32)
        
        # Apply glare
        glare_image = glare_image * (1 - glare_mask) + bright_overlay * glare_mask
        glare_image = np.clip(glare_image, 0, 255).astype(np.uint8)
        
        return glare_image
    
    def batch_augment_dataset(self, images_dir: str, annotations_dir: str, 
                            output_dir: str, augmentation_factor: int = 3):
        """
        Batch augment entire dataset
        
        Args:
            images_dir: Directory containing original images
            annotations_dir: Directory containing YOLO format annotations
            output_dir: Directory to save augmented data
            augmentation_factor: Number of augmented versions per original image
        """
        
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        output_path = Path(output_dir)
        
        output_images = output_path / "images"
        output_annotations = output_path / "annotations"
        
        output_images.mkdir(parents=True, exist_ok=True)
        output_annotations.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        
        print(f"Augmenting {len(image_files)} images with factor {augmentation_factor}")
        
        for i, image_file in enumerate(image_files):
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            # Load annotations
            annotation_file = annotations_path / (image_file.stem + ".txt")
            bboxes = []
            class_labels = []
            
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            class_labels.append(class_id)
                            bboxes.append(bbox)
            
            # Generate augmented versions
            for aug_idx in range(augmentation_factor):
                try:
                    # Apply augmentation
                    if aug_idx == augmentation_factor - 1:  # Last one with severe weather
                        aug_image, aug_bboxes, aug_labels = self.augment_image_with_boxes(
                            image, bboxes, class_labels, severity='severe'
                        )
                    else:
                        aug_image, aug_bboxes, aug_labels = self.augment_image_with_boxes(
                            image, bboxes, class_labels, severity='normal'
                        )
                    
                    # Save augmented image
                    aug_image_name = f"{image_file.stem}_aug_{aug_idx}{image_file.suffix}"
                    aug_image_path = output_images / aug_image_name
                    cv2.imwrite(str(aug_image_path), aug_image)
                    
                    # Save augmented annotations
                    if aug_bboxes and aug_labels:
                        aug_annotation_name = f"{image_file.stem}_aug_{aug_idx}.txt"
                        aug_annotation_path = output_annotations / aug_annotation_name
                        
                        with open(aug_annotation_path, 'w') as f:
                            for bbox, label in zip(aug_bboxes, aug_labels):
                                if len(bbox) == 4:  # Valid bbox
                                    f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\\n")
                    
                except Exception as e:
                    print(f"Error augmenting {image_file}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        print(f"Augmentation complete! Results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    augmentor = TrafficDataAugmentor()
    
    # Example: augment a single image
    # image = cv2.imread("sample_image.jpg")
    # augmented_images = augmentor.create_vietnam_specific_augmentations(image)
    
    # Example: batch augment dataset
    # augmentor.batch_augment_dataset("data/raw/images", "data/annotations", "data/processed", 3)