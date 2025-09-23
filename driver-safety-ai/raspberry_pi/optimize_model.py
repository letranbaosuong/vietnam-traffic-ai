#!/usr/bin/env python3
"""
Model Optimization for Raspberry Pi
Tối ưu hóa model cho edge devices
"""

import os
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional


class ModelOptimizer:
    """
    Optimize TensorFlow/Keras models for edge deployment
    """

    def __init__(self, model_path: str):
        """
        Initialize optimizer

        Args:
            model_path: Path to Keras .h5 model or SavedModel directory
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """Load Keras model"""
        if self.model_path.endswith('.h5'):
            return tf.keras.models.load_model(self.model_path)
        else:
            return tf.saved_model.load(self.model_path)

    def quantize_int8(
        self,
        representative_dataset=None,
        output_path: str = "model_int8.tflite"
    ) -> bytes:
        """
        INT8 Quantization - smallest model size

        Args:
            representative_dataset: Calibration dataset generator
            output_path: Output file path

        Returns:
            Quantized model bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]

        # Representative dataset for calibration
        if representative_dataset:
            converter.representative_dataset = representative_dataset
        else:
            # Default representative dataset
            def default_representative_dataset():
                for _ in range(100):
                    data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                    yield [data]
            converter.representative_dataset = default_representative_dataset

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"INT8 model saved: {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        return tflite_model

    def quantize_float16(
        self,
        output_path: str = "model_fp16.tflite"
    ) -> bytes:
        """
        Float16 Quantization - balance between size and accuracy

        Args:
            output_path: Output file path

        Returns:
            Quantized model bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"FP16 model saved: {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        return tflite_model

    def dynamic_range_quantization(
        self,
        output_path: str = "model_dynamic.tflite"
    ) -> bytes:
        """
        Dynamic Range Quantization - no calibration needed

        Args:
            output_path: Output file path

        Returns:
            Quantized model bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Dynamic range model saved: {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        return tflite_model

    def prune_model(
        self,
        target_sparsity: float = 0.5,
        output_path: str = "model_pruned.h5"
    ):
        """
        Prune model to reduce size

        Args:
            target_sparsity: Target sparsity (0.5 = 50% weights removed)
            output_path: Output model path

        Returns:
            Pruned model
        """
        import tensorflow_model_optimization as tfmot

        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        # Apply pruning to model
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            **pruning_params
        )

        # Compile
        model_for_pruning.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save
        model_for_pruning.save(output_path)
        print(f"Pruned model saved: {output_path}")

        return model_for_pruning

    def edge_tpu_quantization(
        self,
        output_path: str = "model_edgetpu.tflite"
    ):
        """
        Quantization for Google Coral Edge TPU

        Args:
            output_path: Output file path
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Edge TPU requires full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # Representative dataset
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 224, 224, 3)
                data = (data * 255).astype(np.uint8)
                yield [data]

        converter.representative_dataset = representative_dataset

        # Convert
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Edge TPU model saved: {output_path}")
        print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        print("\nNote: Run Edge TPU compiler to make it compatible:")
        print(f"edgetpu_compiler {output_path}")

        return tflite_model


def benchmark_tflite_model(
    model_path: str,
    num_iterations: int = 100,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
):
    """
    Benchmark TFLite model performance

    Args:
        model_path: Path to TFLite model
        num_iterations: Number of inference iterations
        input_shape: Input shape

    Returns:
        Benchmark results
    """
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warm up
    for _ in range(10):
        test_input = np.random.rand(1, *input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        test_input = np.random.rand(1, *input_shape).astype(np.float32)

        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)

    # Calculate statistics
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1000 / avg_time

    # Model size
    model_size = os.path.getsize(model_path) / 1024 / 1024  # MB

    results = {
        'model_path': model_path,
        'model_size_mb': model_size,
        'avg_inference_ms': avg_time,
        'std_inference_ms': std_time,
        'min_inference_ms': min_time,
        'max_inference_ms': max_time,
        'fps': fps,
        'iterations': num_iterations
    }

    return results


def compare_models(original_model_path: str, models_dict: dict):
    """
    Compare different quantized models

    Args:
        original_model_path: Path to original Keras model
        models_dict: Dictionary of model_name: model_path
    """
    print("=" * 60)
    print("Model Comparison Report")
    print("=" * 60)

    # Original model size
    if original_model_path.endswith('.h5'):
        original_size = os.path.getsize(original_model_path) / 1024 / 1024
        print(f"Original model size: {original_size:.2f} MB")

    print("\nQuantized Models Performance:")
    print("-" * 60)

    results = []
    for name, path in models_dict.items():
        if os.path.exists(path):
            print(f"\n{name}:")
            result = benchmark_tflite_model(path)
            results.append(result)

            print(f"  Size: {result['model_size_mb']:.2f} MB")
            print(f"  Avg inference: {result['avg_inference_ms']:.2f} ms")
            print(f"  FPS: {result['fps']:.2f}")

            if original_model_path.endswith('.h5'):
                compression_ratio = original_size / result['model_size_mb']
                print(f"  Compression ratio: {compression_ratio:.2f}x")

    print("\n" + "=" * 60)
    print("Recommendation for Raspberry Pi 4:")

    # Find best model based on FPS
    best_model = max(results, key=lambda x: x['fps'])
    print(f"Best performance: {Path(best_model['model_path']).name}")
    print(f"  - {best_model['fps']:.2f} FPS")
    print(f"  - {best_model['model_size_mb']:.2f} MB")

    return results


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize model for Raspberry Pi deployment'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to Keras model (.h5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='optimized_models',
        help='Output directory for optimized models'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all optimization variants'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all generated models'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize optimizer
    print(f"Loading model: {args.model}")
    optimizer = ModelOptimizer(args.model)

    models = {}

    if args.all:
        # Generate all variants
        print("\nGenerating optimized models...")

        # INT8
        int8_path = os.path.join(args.output_dir, 'model_int8.tflite')
        optimizer.quantize_int8(output_path=int8_path)
        models['INT8'] = int8_path

        # Float16
        fp16_path = os.path.join(args.output_dir, 'model_fp16.tflite')
        optimizer.quantize_float16(output_path=fp16_path)
        models['Float16'] = fp16_path

        # Dynamic Range
        dynamic_path = os.path.join(args.output_dir, 'model_dynamic.tflite')
        optimizer.dynamic_range_quantization(output_path=dynamic_path)
        models['Dynamic Range'] = dynamic_path

        # Edge TPU (if needed)
        # edgetpu_path = os.path.join(args.output_dir, 'model_edgetpu.tflite')
        # optimizer.edge_tpu_quantization(output_path=edgetpu_path)
        # models['Edge TPU'] = edgetpu_path

    if args.compare and models:
        print("\nBenchmarking models...")
        compare_models(args.model, models)


if __name__ == "__main__":
    main()