"""
Convert TTPLA labelme JSON annotations to COCO format.
Merges tower subtypes into single 'tower' class.

Classes:
    0: cable (includes 'cable', 'Cable')
    1: tower (includes 'tower_lattice', 'tower_tucohy', 'tower_wooden')
    
Ignores: 'void' labels

Usage:
    python labelme_to_coco.py --input_dir /path/to/ttpla_raw --output_dir /path/to/ttpla_coco --split_dir /path/to/splits
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm


# Class mapping configuration
CLASS_MAPPING = {
    'cable': 'cable',
    'Cable': 'cable',
    'tower_lattice': 'tower',
    'tower_tucohy': 'tower', 
    'tower_wooden': 'tower',
}

CATEGORIES = [
    {'id': 1, 'name': 'cable', 'supercategory': 'infrastructure'},
    {'id': 2, 'name': 'tower', 'supercategory': 'infrastructure'},
]

CLASS_NAME_TO_ID = {'cable': 1, 'tower': 2}


def polygon_to_bbox(points):
    """Convert polygon points to bounding box [x, y, width, height]."""
    points = np.array(points)
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [float(x_min), float(y_min), float(width), float(height)]


def polygon_area(points):
    """Calculate polygon area using shoelace formula."""
    points = np.array(points)
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def flatten_points(points):
    """Flatten polygon points for COCO segmentation format."""
    return [coord for point in points for coord in point]


def load_split_files(split_dir):
    """Load train/val/test split file lists."""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = os.path.join(split_dir, f'{split_name}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                # Files are listed as .json, we need the base name
                splits[split_name] = [line.strip().replace('.json', '') for line in f if line.strip()]
    return splits


def process_labelme_file(json_path, image_id, annotation_id_start):
    """Process a single labelme JSON file and return COCO annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    annotation_id = annotation_id_start
    
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        
        # Skip void and unmapped labels
        if label == 'void' or label not in CLASS_MAPPING:
            continue
            
        mapped_label = CLASS_MAPPING[label]
        category_id = CLASS_NAME_TO_ID[mapped_label]
        
        points = shape.get('points', [])
        if len(points) < 3:  # Need at least 3 points for a polygon
            continue
            
        bbox = polygon_to_bbox(points)
        area = polygon_area(points)
        segmentation = [flatten_points(points)]
        
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': segmentation,
            'area': area,
            'bbox': bbox,
            'iscrowd': 0,
        }
        
        annotations.append(annotation)
        annotation_id += 1
    
    return annotations, annotation_id


def get_image_info(image_path, image_id):
    """Get image metadata for COCO format."""
    with Image.open(image_path) as img:
        width, height = img.size
    
    return {
        'id': image_id,
        'file_name': os.path.basename(image_path),
        'width': width,
        'height': height,
    }


def convert_split(input_dir, output_dir, split_name, file_list):
    """Convert a single split (train/val/test) to COCO format."""
    
    coco_output = {
        'info': {
            'description': f'TTPLA Dataset - {split_name} split',
            'version': '1.0',
            'year': 2024,
            'contributor': 'Converted from labelme format',
            'date_created': datetime.now().strftime('%Y-%m-%d'),
        },
        'licenses': [],
        'categories': CATEGORIES,
        'images': [],
        'annotations': [],
    }
    
    image_id = 1
    annotation_id = 1
    
    skipped = 0
    
    for base_name in tqdm(file_list, desc=f'Processing {split_name}'):
        json_path = os.path.join(input_dir, f'{base_name}.json')
        
        # Try different image extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = os.path.join(input_dir, f'{base_name}{ext}')
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not os.path.exists(json_path) or image_path is None:
            skipped += 1
            continue
        
        # Get image info
        image_info = get_image_info(image_path, image_id)
        coco_output['images'].append(image_info)
        
        # Process annotations
        annotations, annotation_id = process_labelme_file(json_path, image_id, annotation_id)
        coco_output['annotations'].extend(annotations)
        
        image_id += 1
    
    # Save COCO JSON
    output_path = os.path.join(output_dir, f'{split_name}.json')
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f'{split_name}: {len(coco_output["images"])} images, {len(coco_output["annotations"])} annotations')
    if skipped > 0:
        print(f'  Skipped {skipped} files (missing image or annotation)')
    
    return coco_output


def main():
    parser = argparse.ArgumentParser(description='Convert TTPLA labelme to COCO format')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with labelme JSONs and images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for COCO JSONs')
    parser.add_argument('--split_dir', type=str, required=True, help='Directory with train.txt, val.txt, test.txt')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load splits
    splits = load_split_files(args.split_dir)
    
    print(f'Loaded splits: train={len(splits.get("train", []))}, val={len(splits.get("val", []))}, test={len(splits.get("test", []))}')
    print(f'Class mapping: {CLASS_MAPPING}')
    print(f'Categories: {[c["name"] for c in CATEGORIES]}')
    print()
    
    # Convert each split
    for split_name, file_list in splits.items():
        convert_split(args.input_dir, args.output_dir, split_name, file_list)
    
    print('\nConversion complete!')
    

if __name__ == '__main__':
    main()
