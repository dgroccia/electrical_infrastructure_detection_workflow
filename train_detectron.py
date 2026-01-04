"""
Train Faster R-CNN (ResNet-101 + FPN) on TTPLA dataset using Detectron2.

Usage:
    python train_detectron.py --epochs 100 --batch 4 --lr 0.001
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


class TTPLATrainer(DefaultTrainer):
    """Custom trainer with COCO evaluator."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def register_datasets(coco_dir):
    """Register TTPLA datasets with Detectron2."""
    
    coco_dir = Path(coco_dir)
    
    # Register each split
    for split in ['train', 'val', 'test']:
        dataset_name = f'ttpla_{split}'
        
        # Skip if already registered
        if dataset_name in DatasetCatalog.list():
            continue
            
        json_file = coco_dir / f'{split}.json'
        
        # Get image directory from the first image in the JSON
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Images are in ttpla_raw directory
        image_root = coco_dir.parent / 'ttpla_raw'
        
        register_coco_instances(
            dataset_name,
            {},
            str(json_file),
            str(image_root)
        )
        
        # Set metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=['cable', 'tower'],
            evaluator_type='coco'
        )
        
        print(f'Registered dataset: {dataset_name}')


def setup_cfg(args, output_dir):
    """Set up Detectron2 configuration."""
    
    cfg = get_cfg()
    
    # Use Faster R-CNN with ResNet-101 + FPN backbone
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    ))
    
    # Load pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    
    # Dataset
    cfg.DATASETS.TRAIN = ("ttpla_train",)
    cfg.DATASETS.TEST = ("ttpla_val",)
    
    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Model - 2 classes (cable, tower)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Solver
    # Calculate iterations from epochs
    # Roughly: iterations = (num_images * epochs) / batch_size
    num_train_images = 905
    iterations_per_epoch = num_train_images // args.batch
    max_iter = iterations_per_epoch * args.epochs
    
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))  # LR decay steps
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(1000, max_iter // 10)
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = iterations_per_epoch * 10  # Save every 10 epochs
    
    # Test
    cfg.TEST.EVAL_PERIOD = iterations_per_epoch * 5  # Evaluate every 5 epochs
    
    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output
    cfg.OUTPUT_DIR = str(output_dir)
    
    return cfg


def train(args):
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    coco_dir = project_root / 'data' / 'ttpla_coco'
    output_base = project_root / 'outputs' / 'detectron_runs'
    
    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'frcnn_r101_ep{args.epochs}_bs{args.batch}_lr{args.lr}_{timestamp}'
    output_dir = output_base / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'=' * 60)
    print(f'Faster R-CNN (ResNet-101 + FPN) Training')
    print(f'=' * 60)
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch}')
    print(f'Learning rate: {args.lr}')
    print(f'COCO dir: {coco_dir}')
    print(f'Output dir: {output_dir}')
    print(f'=' * 60)
    
    # Register datasets
    register_datasets(coco_dir)
    
    # Setup config
    cfg = setup_cfg(args, output_dir)
    
    print(f'\nMax iterations: {cfg.SOLVER.MAX_ITER}')
    print(f'LR decay steps: {cfg.SOLVER.STEPS}')
    print(f'Eval period: {cfg.TEST.EVAL_PERIOD} iterations')
    print(f'Checkpoint period: {cfg.SOLVER.CHECKPOINT_PERIOD} iterations')
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(cfg.dump())
    
    # Train
    trainer = TTPLATrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f'\nTraining complete!')
    print(f'Results saved to: {output_dir}')
    
    # Final evaluation on test set
    print(f'\nRunning final evaluation on test set...')
    cfg.DATASETS.TEST = ("ttpla_test",)
    evaluator = COCOEvaluator("ttpla_test", output_dir=str(output_dir / "test_inference"))
    val_loader = build_detection_test_loader(cfg, "ttpla_test")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print(f'\nTest Results:')
    print(results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on TTPLA dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size (images per GPU)')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
