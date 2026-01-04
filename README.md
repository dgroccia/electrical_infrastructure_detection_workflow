# Electrical Infrastructure Detection from Aerial Imagery

This study benchmarks deep learning and classical computer vision methods for detecting transmission towers and power lines from aerial imagery.

## Overview

This repo contains code for training object detection models on the [TTPLA dataset](https://github.com/R3ab/ttpla_dataset). We compare:

- **Single-stage detectors**: YOLOv8m, YOLOv9m
- **Two-stage detectors**: Faster R-CNN, Cascade R-CNN
- **Transformer-based**: RT-DETR
- **Classical CV**: LSD, Canny+Hough, Sobel+Hough

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Cable AP@0.5 | Tower AP@0.5 | FPS |
|-------|---------|--------------|--------------|--------------|-----|
| YOLOv9m | 70.9% | 56.3% | 65.6% | 76.3% | 24.7 |
| YOLOv8m | 69.6% | 55.0% | 64.5% | 74.7% | 23.3 |
| RT-DETR | 64.9% | 46.4% | 59.7% | 70.1% | 21.3 |
| Faster R-CNN | 61.1% | 45.1% | 55.5% | 66.7% | 8.3 |
| Cascade R-CNN | 60.3% | 49.3% | 54.8% | 65.9% | 7.2 |

*FPS measured on RTX 5060 Ti*

| Classical CV Method | Precision | Recall | F1 |
|---------------------|-----------|--------|-----|
| LSD | 49.7% | 42.0% | 45.6% |
| Sobel+Hough | 5.0% | 87.8% | 9.4% |
| Canny+Hough | 4.4% | 72.7% | 8.2% |

## Installation
```bash
git clone https://github.com/dgroccia/electrical_infrastructure_detection_workflow.git
cd electrical_infrastructure_detection_workflow

# Create conda environment
conda create -n infra-detect python=3.10
conda activate infra-detect
```

## Dataset

Download the TTPLA dataset from the [official repository](https://github.com/R3ab/ttpla_dataset) and place images in `data/ttpla_raw/`.

Convert annotations:
```bash
python data/scripts/labelme_to_coco.py
python data/scripts/coco_to_yolo.py
```

## Training
```bash
# YOLOv8m
python src/train/train_yolo.py --model yolov8m --epochs 100

# YOLOv9m
python src/train/train_yolo.py --model yolov9m --epochs 100

# RT-DETR
python src/train/train_rtdetr.py --epochs 100

# Faster R-CNN
python src/train/train_detectron.py --config configs/faster_rcnn_r101_fpn.yaml

# Cascade R-CNN
python src/train/train_detectron.py --config configs/cascade_rcnn_r101_fpn.yaml
```


