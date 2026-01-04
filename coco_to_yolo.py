"""
Convert COCO format annotations to YOLO format.

YOLO format: class_id x_center y_center width height (all normalized 0-1)

Usage:
    python coco_to_yolo.py --coco_dir /path/to/ttpla_coco --output_dir /path/to/ttpla_yolo --images_dir /path/to/ttpla_raw
"""

import json
import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO format.
    YOLO: [x_center, y_center, width, height] normalized to 0-1
    """
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    # Clip to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height


def convert_split(coco_json_path, images_source_dir, output_dir, split_name):
    """Convert a single COCO split to YOLO format."""
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_out_dir = os.path.join(output_dir, 'images', split_name)
    labels_out_dir = os.path.join(output_dir, 'labels', split_name)
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    # Build lookup dictionaries
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Build category id mapping (COCO uses 1-indexed, YOLO uses 0-indexed)
    category_mapping = {}
    for cat in coco_data['categories']:
        # COCO category IDs to YOLO class indices (0-indexed)
        category_mapping[cat['id']] = cat['id'] - 1
    
    processed = 0
    
    for img_id, img_info in tqdm(images_dict.items(), desc=f'Converting {split_name}'):
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        base_name = os.path.splitext(file_name)[0]
        
        # Copy image to output directory
        src_image_path = os.path.join(images_source_dir, file_name)
        dst_image_path = os.path.join(images_out_dir, file_name)
        
        if os.path.exists(src_image_path) and not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        
        # Create YOLO label file
        label_path = os.path.join(labels_out_dir, f'{base_name}.txt')
        
        annotations = annotations_by_image.get(img_id, [])
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                coco_cat_id = ann['category_id']
                yolo_class_id = category_mapping[coco_cat_id]
                
                bbox = ann['bbox']
                x_center, y_center, width, height = coco_to_yolo_bbox(bbox, img_width, img_height)
                
                # Write YOLO format line
                f.write(f'{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
        
        processed += 1
    
    print(f'{split_name}: {processed} images processed')
    return processed


def create_data_yaml(output_dir, class_names):
    """Create YOLO data.yaml configuration file."""
    
    data_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names),
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f'Created {yaml_path}')
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format')
    parser.add_argument('--coco_dir', type=str, required=True, help='Directory with COCO JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory with source images')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    class_names = ['cable', 'tower']
    
    print(f'Converting COCO to YOLO format')
    print(f'Classes: {class_names}')
    print()
    
    # Convert each split
    for split_name in ['train', 'val', 'test']:
        coco_json_path = os.path.join(args.coco_dir, f'{split_name}.json')
        if os.path.exists(coco_json_path):
            convert_split(coco_json_path, args.images_dir, args.output_dir, split_name)
        else:
            print(f'Warning: {coco_json_path} not found, skipping {split_name}')
    
    # Create data.yaml
    create_data_yaml(args.output_dir, class_names)
    
    print('\nConversion complete!')
    print(f'\nYOLO dataset structure:')
    print(f'{args.output_dir}/')
    print(f'├── images/')
    print(f'│   ├── train/')
    print(f'│   ├── val/')
    print(f'│   └── test/')
    print(f'├── labels/')
    print(f'│   ├── train/')
    print(f'│   ├── val/')
    print(f'│   └── test/')
    print(f'└── data.yaml')


if __name__ == '__main__':
    main()
