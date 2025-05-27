![image](https://github.com/user-attachments/assets/d416d9fb-214e-436d-968f-0205200f300b)# Object Detection with ResNet+YOLO on Pascal VOC 2007

## Overview

This project implements a custom object detection system combining ResNet-34 as a feature extractor with a YOLO-style detection head. The model is trained and evaluated on the Pascal VOC 2007 dataset.

## Architecture

### Model Components
- **Backbone**: ResNet-34 (pre-trained on ImageNet)
- **Detection Head**: Two-layer convolutional network
- **Output**: 7×7×30 tensor (20 classes + 2×5 bbox predictions per grid cell)

### Why This Architecture?

**ResNet-34 Choice**: Provides excellent feature extraction while being computationally efficient. The residual connections help with gradient flow during training, and pre-trained weights give us a strong starting point.

**YOLO-style Detection**: Single-stage detector that's simpler to implement and understand compared to two-stage approaches like R-CNN. Performs detection in one forward pass by dividing the image into a 7×7 grid.

### Network Architecture
```
Input (448×448×3)
    ↓
ResNet-34 Backbone (remove final FC layers)
    ↓
Adaptive Average Pooling → (7×7×512)
    ↓
Conv2d(512→1024, 3×3) + ReLU
    ↓
Conv2d(1024→30, 1×1)
    ↓
Output (7×7×30)
```

## Implementation Details

### Dataset Processing
- **Input Size**: 448×448 (divisible by 7 for grid alignment)
- **Normalization**: ImageNet statistics for backbone compatibility
- **Target Encoding**: Convert Pascal VOC XML annotations to YOLO grid format

### Loss Function
Custom YOLO loss with three components:
- **Coordinate Loss**: MSE for bounding box coordinates (λ_coord = 5.0)
- **Confidence Loss**: MSE for objectness predictions
- **Classification Loss**: MSE for class probabilities
- **No-object Loss**: Penalty for false positives (λ_noobj = 0.5)

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 4 (GPU memory constraint)
- **Epochs**: 100 with early stopping
- **Augmentations**: Resize, normalize using Albumentations

## Evaluation Metrics

### Comprehensive Evaluation
- **mAP@0.5**: Standard detection metric
- **mAP@0.75**: Stricter localization requirement
- **mAP@0.9**: Very strict localization
- **Per-class AP**: Individual class performance analysis

### Post-processing Pipeline
1. **Decode Predictions**: Convert grid coordinates to absolute positions
2. **Confidence Filtering**: Remove predictions below 0.3 threshold
3. **Non-Maximum Suppression**: Eliminate duplicates (IoU threshold: 0.4)

## Key Implementation Challenges

### Target Encoding Complexity
Converting Pascal VOC bounding boxes to YOLO grid format required careful handling of:
- Multiple objects in single grid cells
- Coordinate system transformations
- Proper assignment of objects to grid cells

### Memory Constraints
Limited GPU memory necessitated:
- Small batch sizes (4 samples)
- Efficient data loading
- Gradient accumulation strategies

### Loss Function Balancing
Multiple loss components required careful weighting to ensure:
- Proper localization accuracy
- Balanced classification performance
- Suppression of false positives




## Results

### Performance Metrics
*[Results will be updated after training completion]*

- mAP@0.5: [0.7629]
- mAP@0.75: [0.3530]
- Overall Precision: [0.624756]
- Overall Recall: [0.32118]

### Generated Visualizations
- Average Precision comparison across IoU thresholds
- Per-class performance analysis
- Precision-Recall curves
- Detection confidence distributions

## Technical Decisions

### Why 7×7 Grid?
- Balances spatial resolution with computational efficiency
- Each cell covers ~64×64 pixels in original image
- Suitable for Pascal VOC object sizes

### Why 2 Bounding Boxes per Cell?
- Handles multiple objects in same grid cell
- Provides redundancy for better detection
- Standard YOLO configuration

### Pre-trained Backbone Benefits
- Faster convergence on limited dataset
- Better feature representations
- Reduced training time and compute requirements

## Future Improvements

### Model Enhancements
- Experiment with larger backbones (ResNet-50, EfficientNet)
- Implement anchor boxes for better localization
- Add Feature Pyramid Network for multi-scale detection

### Training Improvements
- Advanced data augmentation (mixup, mosaic)
- Learning rate scheduling
- Larger batch sizes with better hardware

### Evaluation Extensions
- Speed benchmarking (FPS measurements)
- Memory usage profiling
- Comparison with other detection methods

## Dependencies

```bash
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.0.0
opencv-python>=4.5.0
matplotlib>=3.3.0
tqdm>=4.60.0
numpy>=1.20.0
```

## Usage

### Training
```bash
python train.py 
```

### Evaluation
```bash
python evaluate.py 
```

### Inference
```bash
python inference.py 
```

## Key Learnings

This implementation provided hands-on experience with:
- Single-stage object detection architectures
- Multi-task loss function design
- Comprehensive evaluation methodologies
- End-to-end computer vision pipeline development

The project demonstrates how established architectures (ResNet, YOLO) can be combined effectively while highlighting the engineering challenges in practical computer vision systems.
