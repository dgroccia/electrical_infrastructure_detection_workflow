"""
Classical CV pipeline for powerline detection using edge detection and line detection.

Methods:
- Edge detection: Canny, Sobel
- Line detection: Hough Transform, Probabilistic Hough, LSD

Usage:
    python powerline_detection.py --image /path/to/image.jpg --method canny_hough --visualize
    python powerline_detection.py --input_dir /path/to/images --output_dir /path/to/results
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class LineSegment:
    """Represents a detected line segment."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle(self) -> float:
        """Angle in degrees from horizontal."""
        return np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1))
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class PowerlineDetector:
    """Classical CV pipeline for powerline detection."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional configuration override."""
        self.config = {
            # Preprocessing
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8),
            'blur_kernel': 5,
            
            # Canny parameters
            'canny_low': 50,
            'canny_high': 150,
            
            # Sobel parameters
            'sobel_kernel': 3,
            'sobel_threshold': 50,
            
            # Hough Transform parameters
            'hough_rho': 1,
            'hough_theta': np.pi / 180,
            'hough_threshold': 100,
            'hough_min_line_length': 100,
            'hough_max_line_gap': 10,
            
            # LSD parameters (uses OpenCV defaults mostly)
            'lsd_scale': 0.8,
            
            # Post-processing
            'min_line_length': 50,
            'angle_tolerance': 30,  # Filter lines within this angle from horizontal
        }
        
        if config:
            self.config.update(config)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image: grayscale, CLAHE, blur."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(
            clipLimit=self.config['clahe_clip_limit'],
            tileGridSize=self.config['clahe_grid_size']
        )
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            enhanced, 
            (self.config['blur_kernel'], self.config['blur_kernel']), 
            0
        )
        
        return blurred
    
    def detect_edges_canny(self, preprocessed: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detector."""
        edges = cv2.Canny(
            preprocessed,
            self.config['canny_low'],
            self.config['canny_high']
        )
        return edges
    
    def detect_edges_sobel(self, preprocessed: np.ndarray) -> np.ndarray:
        """Detect edges using Sobel operator."""
        # Compute gradients in x and y directions
        sobel_x = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=self.config['sobel_kernel'])
        sobel_y = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=self.config['sobel_kernel'])
        
        # Compute magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = np.uint8(255 * magnitude / magnitude.max())
        
        # Threshold
        _, edges = cv2.threshold(
            magnitude, 
            self.config['sobel_threshold'], 
            255, 
            cv2.THRESH_BINARY
        )
        
        return edges
    
    def detect_lines_hough(self, edges: np.ndarray) -> List[LineSegment]:
        """Detect lines using Standard Hough Transform."""
        lines = cv2.HoughLines(
            edges,
            self.config['hough_rho'],
            self.config['hough_theta'],
            self.config['hough_threshold']
        )
        
        if lines is None:
            return []
        
        # Convert to line segments (extend lines to image boundaries)
        h, w = edges.shape
        segments = []
        
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extend line to image boundaries
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            
            segments.append(LineSegment(x1, y1, x2, y2))
        
        return segments
    
    def detect_lines_hough_probabilistic(self, edges: np.ndarray) -> List[LineSegment]:
        """Detect lines using Probabilistic Hough Transform."""
        lines = cv2.HoughLinesP(
            edges,
            self.config['hough_rho'],
            self.config['hough_theta'],
            self.config['hough_threshold'],
            minLineLength=self.config['hough_min_line_length'],
            maxLineGap=self.config['hough_max_line_gap']
        )
        
        if lines is None:
            return []
        
        segments = [LineSegment(l[0], l[1], l[2], l[3]) for l in lines[:, 0]]
        return segments
    
    def detect_lines_lsd(self, preprocessed: np.ndarray) -> List[LineSegment]:
        """Detect lines using Line Segment Detector (LSD)."""
        # Create LSD detector
        lsd = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD,
            scale=self.config['lsd_scale']
        )
        
        lines, _, _, _ = lsd.detect(preprocessed)
        
        if lines is None:
            return []
        
        segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(LineSegment(int(x1), int(y1), int(x2), int(y2)))
        
        return segments
    
    def filter_lines(self, segments: List[LineSegment]) -> List[LineSegment]:
        """Filter lines based on length and angle (powerlines are roughly horizontal)."""
        filtered = []
        
        for seg in segments:
            # Filter by minimum length
            if seg.length < self.config['min_line_length']:
                continue
            
            # Filter by angle (keep lines within tolerance of horizontal)
            # Horizontal is 0 degrees, we also accept near 180/-180
            angle = abs(seg.angle)
            if angle > 90:
                angle = 180 - angle
            
            if angle <= self.config['angle_tolerance']:
                filtered.append(seg)
        
        return filtered
    
    def merge_collinear_segments(self, segments: List[LineSegment], 
                                  angle_thresh: float = 5.0,
                                  distance_thresh: float = 50.0) -> List[LineSegment]:
        """Merge collinear and nearby line segments."""
        if not segments:
            return []
        
        merged = []
        used = set()
        
        for i, seg1 in enumerate(segments):
            if i in used:
                continue
            
            # Find all segments that could be merged with this one
            group = [seg1]
            used.add(i)
            
            for j, seg2 in enumerate(segments):
                if j in used:
                    continue
                
                # Check angle similarity
                angle_diff = abs(seg1.angle - seg2.angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff > angle_thresh:
                    continue
                
                # Check distance (simplified: distance between midpoints)
                mid1 = ((seg1.x1 + seg1.x2) / 2, (seg1.y1 + seg1.y2) / 2)
                mid2 = ((seg2.x1 + seg2.x2) / 2, (seg2.y1 + seg2.y2) / 2)
                dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                
                if dist < distance_thresh:
                    group.append(seg2)
                    used.add(j)
            
            # Merge group into single segment (bounding box approach)
            all_x = [s.x1 for s in group] + [s.x2 for s in group]
            all_y = [s.y1 for s in group] + [s.y2 for s in group]
            
            # Create merged segment from extreme points
            min_x_idx = np.argmin(all_x)
            max_x_idx = np.argmax(all_x)
            
            merged.append(LineSegment(
                all_x[min_x_idx], all_y[min_x_idx],
                all_x[max_x_idx], all_y[max_x_idx]
            ))
        
        return merged
    
    def detect(self, image: np.ndarray, method: str = 'canny_hough_p') -> List[LineSegment]:
        """
        Run full detection pipeline.
        
        Methods:
            - 'canny_hough': Canny edges + Standard Hough
            - 'canny_hough_p': Canny edges + Probabilistic Hough
            - 'canny_lsd': Canny edges + LSD (note: LSD usually works on grayscale directly)
            - 'sobel_hough_p': Sobel edges + Probabilistic Hough
            - 'lsd': LSD directly on preprocessed image
        """
        preprocessed = self.preprocess(image)
        
        if method == 'canny_hough':
            edges = self.detect_edges_canny(preprocessed)
            segments = self.detect_lines_hough(edges)
        elif method == 'canny_hough_p':
            edges = self.detect_edges_canny(preprocessed)
            segments = self.detect_lines_hough_probabilistic(edges)
        elif method == 'sobel_hough_p':
            edges = self.detect_edges_sobel(preprocessed)
            segments = self.detect_lines_hough_probabilistic(edges)
        elif method == 'lsd':
            segments = self.detect_lines_lsd(preprocessed)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Filter and merge
        filtered = self.filter_lines(segments)
        merged = self.merge_collinear_segments(filtered)
        
        return merged
    
    def visualize(self, image: np.ndarray, segments: List[LineSegment], 
                  title: str = "Detected Lines") -> np.ndarray:
        """Draw detected lines on image."""
        vis = image.copy()
        
        for seg in segments:
            cv2.line(vis, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)
        
        return vis


def visualize_pipeline(image: np.ndarray, detector: PowerlineDetector, 
                       method: str = 'canny_hough_p', save_path: Optional[str] = None):
    """Visualize the full pipeline with intermediate steps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Preprocessed
    preprocessed = detector.preprocess(image)
    axes[0, 1].imshow(preprocessed, cmap='gray')
    axes[0, 1].set_title('Preprocessed (CLAHE + Blur)')
    axes[0, 1].axis('off')
    
    # Edges
    if 'canny' in method:
        edges = detector.detect_edges_canny(preprocessed)
        edge_title = 'Canny Edges'
    else:
        edges = detector.detect_edges_sobel(preprocessed)
        edge_title = 'Sobel Edges'
    
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title(edge_title)
    axes[0, 2].axis('off')
    
    # Raw lines
    if 'hough_p' in method:
        raw_segments = detector.detect_lines_hough_probabilistic(edges)
    elif 'hough' in method:
        raw_segments = detector.detect_lines_hough(edges)
    else:
        raw_segments = detector.detect_lines_lsd(preprocessed)
    
    raw_vis = detector.visualize(image, raw_segments)
    axes[1, 0].imshow(cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Raw Lines ({len(raw_segments)})')
    axes[1, 0].axis('off')
    
    # Filtered lines
    filtered = detector.filter_lines(raw_segments)
    filtered_vis = detector.visualize(image, filtered)
    axes[1, 1].imshow(cv2.cvtColor(filtered_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Filtered Lines ({len(filtered)})')
    axes[1, 1].axis('off')
    
    # Merged lines (final result)
    merged = detector.merge_collinear_segments(filtered)
    final_vis = detector.visualize(image, merged)
    axes[1, 2].imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Final Result ({len(merged)})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    
    plt.show()
    return fig


def compare_methods(image: np.ndarray, save_path: Optional[str] = None):
    """Compare different detection methods side by side."""
    methods = ['canny_hough_p', 'sobel_hough_p', 'lsd']
    detector = PowerlineDetector()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, method in enumerate(methods):
        segments = detector.detect(image, method)
        vis = detector.visualize(image, segments)
        axes[idx + 1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[idx + 1].set_title(f'{method}\n({len(segments)} lines)')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Classical CV powerline detection')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Directory of images to process')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--method', type=str, default='canny_hough_p',
                        choices=['canny_hough', 'canny_hough_p', 'sobel_hough_p', 'lsd'],
                        help='Detection method')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--compare', action='store_true', help='Compare all methods')
    
    args = parser.parse_args()
    
    detector = PowerlineDetector()
    
    if args.image:
        # Single image mode
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        if args.compare:
            compare_methods(image)
        elif args.visualize:
            visualize_pipeline(image, detector, args.method)
        else:
            segments = detector.detect(image, args.method)
            print(f"Detected {len(segments)} line segments")
            for i, seg in enumerate(segments):
                print(f"  {i+1}: ({seg.x1}, {seg.y1}) -> ({seg.x2}, {seg.y2}), "
                      f"length={seg.length:.1f}, angle={seg.angle:.1f}°")
    
    elif args.input_dir:
        # Batch mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
        
        results = {}
        for img_path in tqdm(image_files, desc='Processing images'):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            segments = detector.detect(image, args.method)
            
            # Save visualization
            vis = detector.visualize(image, segments)
            vis_path = output_dir / f'{img_path.stem}_detected.jpg'
            cv2.imwrite(str(vis_path), vis)
            
            results[img_path.name] = {
                'num_lines': len(segments),
                'lines': [seg.to_tuple() for seg in segments]
            }
        
        # Save results JSON
        results_path = output_dir / 'detection_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nProcessed {len(results)} images")
        print(f"Results saved to {output_dir}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
