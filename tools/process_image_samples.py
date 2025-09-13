import cv2
import numpy as np
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

# Ensure the geometry_engine package is discoverable
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geometry_engine.api import GeometryEngine
from geometry_engine.primitives import Point, LineSegment

IMAGE_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'image_samples')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'image_tests')

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def extract_lines_from_image(image_path: str, 
                             canny_low_threshold: int = 30, 
                             canny_high_threshold: int = 100,
                             hough_rho: float = 1, 
                             hough_theta: float = np.pi / 180, 
                             hough_threshold: int = 30,
                             hough_min_line_length: int = 30, 
                             hough_max_line_gap: int = 15) -> Tuple[List[List[float]], int, int]:
    """
    Extracts line segments from an image using Canny edge detection and Hough Line Transform.
    Returns raw lines, image width, and image height.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return [], 0, 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use adaptive thresholding for better line detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Detect edges
    edges = cv2.Canny(thresh, canny_low_threshold, canny_high_threshold, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, hough_rho, hough_theta, hough_threshold, 
                            minLineLength=hough_min_line_length, maxLineGap=hough_max_line_gap)
    
    raw_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            raw_lines.append([float(x1), float(y1), float(x2), float(y2)])
            
    return raw_lines, img.shape[1], img.shape[0] # Return lines, width, height

def plot_results(original_img: np.ndarray, raw_lines: List[List[float]], solved_lines: List[LineSegment], filename: str):
    """Plots original image, raw lines, and solved lines side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot original image
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot raw lines
    axes[1].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)) # Use original image as background
    for line in raw_lines:
        axes[1].plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=1)
    axes[1].set_title('Detected Raw Lines')
    axes[1].axis('off')

    # Plot solved lines
    axes[2].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)) # Use original image as background
    for line in solved_lines:
        axes[2].plot([line.start.x, line.end.x], [line.start.y, line.end.y], 'g-', linewidth=2)
    axes[2].set_title('Solved Geometry')
    axes[2].axis('off')
    
    plt.tight_layout()
    ensure_dir(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_processed.png"))
    plt.close(fig)

def main():
    ensure_dir(IMAGE_SAMPLES_DIR)
    ensure_dir(OUTPUT_DIR)
    
    image_files = [f for f in os.listdir(IMAGE_SAMPLES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {IMAGE_SAMPLES_DIR}. Please place images there.")
        return

    engine = GeometryEngine()

    for img_file in image_files:
        print(f"Processing {img_file}...")
        image_path = os.path.join(IMAGE_SAMPLES_DIR, img_file)
        
        raw_lines, img_width, img_height = extract_lines_from_image(image_path)
        print(f"  Extracted {len(raw_lines)} raw lines.")

        if not raw_lines:
            print(f"  No lines detected in {img_file}. Skipping geometry processing.")
            continue

        # Process with geometry engine
        result = engine.process_raw_geometry(raw_lines)
        solved_lines = result['lines']
        constraints = result['constraints']
        
        print(f"  Solved to {len(solved_lines)} lines.")
        print(f"  Detected constraints: {constraints.keys()}")
        
        # Plotting
        original_img = cv2.imread(image_path)
        plot_results(original_img, raw_lines, solved_lines, os.path.splitext(img_file)[0])
        print(f"  Results saved to {os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0])}_processed.png")

if __name__ == '__main__':
    main()
