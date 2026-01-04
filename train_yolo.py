"""
Train YOLOv8 on TTPLA dataset for pylon and cable detection.

Usage:
    python train_yolo.py --model yolov8m --epochs 100 --batch 16 --imgsz 640
    python train_yolo.py --model yolov8l --epochs 100 --batch 8 --imgsz 700
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def train(args):
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_yaml = project_root / 'data' / 'ttpla_yolo' / 'data.yaml'
    output_dir = project_root / 'outputs'
    
    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{args.model}_ep{args.epochs}_bs{args.batch}_img{args.imgsz}_{timestamp}'
    
    print(f'=' * 60)
    print(f'YOLOv8 Training')
    print(f'=' * 60)
    print(f'Model: {args.model}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch}')
    print(f'Image size: {args.imgsz}')
    print(f'Data config: {data_yaml}')
    print(f'Experiment: {exp_name}')
    print(f'=' * 60)
    
    # Load model (pretrained on COCO)
    model = YOLO(f'{args.model}.pt')
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(output_dir / 'yolo_runs'),
        name=exp_name,
        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # Other settings
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every N epochs
        device=0,  # Use GPU 0
        workers=8,
        seed=42,
        verbose=True,
        plots=True,
    )
    
    print(f'\nTraining complete!')
    print(f'Results saved to: {output_dir / "yolo_runs" / exp_name}')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on TTPLA dataset')
    parser.add_argument('--model', type=str, default='yolov8m', 
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model variant')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
