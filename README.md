# SUMBA (Smart Universal Manipulation Based on AI)

## Overview
SUMBA is an advanced AI-powered robotic manipulation project that combines computer vision, pose detection, and audio processing to create an intelligent system for object manipulation. The project utilizes state-of-the-art YOLOv8 models for object detection and segmentation, along with sophisticated pose estimation and audio detection capabilities.

## Features
- Real-time object detection and segmentation using YOLOv8
- Pose detection for precise object manipulation
- Audio processing capabilities
- Flexible pipeline for single or multiple object processing
- Integration with robotic manipulation systems
- Real-time camera feed processing
- Configurable detection and segmentation parameters

## Project Structure
```
SUMBA/
├── Audio Detection/      # Audio processing components
├── Pose Detection/      # Pose detection implementation
├── imgs/               # Image resources
├── src/                # Source code
├── demo_sumba.py       # Main demo application
├── requirements.txt    # Project dependencies
└── *.pt               # YOLOv8 model weights
```

## Prerequisites
- Python 3.x
- Camera system compatible with the project
- Required hardware for manipulation
- Sufficient computational resources for running AI models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SUMBA.git
cd SUMBA
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure all YOLOv8 model weights are properly placed in the root directory:
- yolov8n.pt
- yolov8n-seg.pt
- yolov8l.pt
- yolov8l-seg.pt
- yolov8x.pt
- yolov8x-seg.pt

## Usage

### Running the Demo
The project includes a demo script that showcases the main functionality:

```python
python demo_sumba.py
```

The demo provides two main operation modes:
1. Single image processing
2. Multiple image continuous processing

### Configuration
The SUMBA pipeline can be configured with various parameters:

```python
sumba = Sumba(
    detector_id="yolov8",
    detector_th=0.05,
    detector_max_object_size=0.1,
    segmentator_id="yolov8",
    segmentator_min_mask_size=0.3,
    grasping_N=50,
    grasping_tol=5,
    show=True
)
```

#### Parameters:
- `detector_id`: Object detection model identifier
- `detector_th`: Detection threshold
- `detector_max_object_size`: Maximum size of detectable objects
- `segmentator_id`: Segmentation model identifier
- `segmentator_min_mask_size`: Minimum mask size for segmentation
- `grasping_N`: Number of grasping points to consider
- `grasping_tol`: Grasping tolerance
- `show`: Enable/disable visualization

## Components

### 1. Object Detection and Segmentation
- Utilizes YOLOv8 models for accurate object detection
- Supports both standard detection and instance segmentation
- Multiple model sizes available (nano, large, and extra large)

### 2. Pose Detection
- Implements advanced pose detection algorithms
- Enables precise object manipulation
- Supports real-time pose estimation

### 3. Audio Detection
- Processes audio signals for enhanced interaction
- Integrates with the main pipeline for multi-modal operation

### 4. Camera Integration
- Supports real-time camera feed processing
- Handles both color and depth image processing
- Includes frame management and preprocessing

### 5. Manipulation System
- Interfaces with robotic manipulation hardware
- Executes precise movement based on detected positions
- Supports various manipulation strategies

## Contributing
Contributions to SUMBA are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License
[Specify your license here]

## Authors
[List the project authors and contributors]

## Acknowledgments
- YOLOv8 team for their excellent object detection models
- [Add other acknowledgments as needed]

## Contact
[Add contact information or ways to reach the project maintainers]
