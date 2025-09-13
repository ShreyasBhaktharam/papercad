import cv2
import numpy as np
import os

def create_synthetic_floorplan1():
    """Create a synthetic floor plan similar to the dormitory layout described"""
    # Create a white canvas
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Draw the main structure - 5 vertical units
    unit_width = 200
    unit_height = 700
    
    for i in range(5):
        x_start = 50 + i * unit_width
        
        # Main unit rectangle
        cv2.rectangle(img, (x_start, 50), (x_start + unit_width - 20, 50 + unit_height), (0, 0, 0), 2)
        
        # Interior walls
        # Horizontal divider
        cv2.line(img, (x_start, 200), (x_start + unit_width - 20, 200), (0, 0, 0), 1)
        cv2.line(img, (x_start, 400), (x_start + unit_width - 20, 400), (0, 0, 0), 1)
        cv2.line(img, (x_start, 600), (x_start + unit_width - 20, 600), (0, 0, 0), 1)
        
        # Vertical divider
        cv2.line(img, (x_start + 90, 200), (x_start + 90, 400), (0, 0, 0), 1)
        
        # Doors (arcs)
        cv2.ellipse(img, (x_start + 45, 300), (15, 15), 0, 0, 90, (0, 0, 0), 2)
        cv2.ellipse(img, (x_start + 45, 500), (15, 15), 0, 0, 90, (0, 0, 0), 2)
        
        # Furniture rectangles
        cv2.rectangle(img, (x_start + 10, 220), (x_start + 80, 250), (0, 0, 0), 1)  # bed
        cv2.rectangle(img, (x_start + 10, 420), (x_start + 80, 450), (0, 0, 0), 1)  # bed
        cv2.rectangle(img, (x_start + 100, 220), (x_start + 170, 250), (0, 0, 0), 1)  # cabinet
        
        # Windows
        cv2.rectangle(img, (x_start + 20, 30), (x_start + 160, 40), (0, 0, 0), 1)
    
    return img

def create_synthetic_floorplan2():
    """Create a single unit floor plan"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Main room rectangle
    cv2.rectangle(img, (50, 50), (750, 550), (0, 0, 0), 2)
    
    # Interior walls
    cv2.line(img, (200, 50), (200, 550), (0, 0, 0), 1)  # vertical divider
    cv2.line(img, (50, 300), (200, 300), (0, 0, 0), 1)   # horizontal divider
    
    # Bathroom area
    cv2.rectangle(img, (50, 300), (200, 550), (0, 0, 0), 1)
    cv2.line(img, (125, 300), (125, 550), (0, 0, 0), 1)  # bathroom divider
    
    # Doors
    cv2.ellipse(img, (175, 275), (20, 20), 0, 0, 90, (0, 0, 0), 2)
    cv2.ellipse(img, (100, 275), (20, 20), 0, 0, 90, (0, 0, 0), 2)
    
    # Furniture
    cv2.rectangle(img, (250, 100), (350, 200), (0, 0, 0), 1)  # bed
    cv2.rectangle(img, (400, 100), (500, 200), (0, 0, 0), 1)  # tv cabinet
    cv2.rectangle(img, (250, 400), (350, 500), (0, 0, 0), 1)  # wardrobe
    
    # Bathroom fixtures
    cv2.rectangle(img, (70, 350), (100, 380), (0, 0, 0), 1)  # toilet
    cv2.rectangle(img, (150, 350), (180, 380), (0, 0, 0), 1)  # sink
    
    # Windows
    cv2.rectangle(img, (300, 30), (500, 40), (0, 0, 0), 1)
    
    return img

def main():
    output_dir = "/Users/nishanthkotla/Desktop/papercad/data/image_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic floor plans
    img1 = create_synthetic_floorplan1()
    img2 = create_synthetic_floorplan2()
    
    # Save images
    cv2.imwrite(os.path.join(output_dir, "synthetic_floorplan1.png"), img1)
    cv2.imwrite(os.path.join(output_dir, "synthetic_floorplan2.png"), img2)
    
    print(f"Generated synthetic floor plans:")
    print(f"  - {os.path.join(output_dir, 'synthetic_floorplan1.png')}")
    print(f"  - {os.path.join(output_dir, 'synthetic_floorplan2.png')}")
    print("\nTo test with your actual images:")
    print("1. Save @0000-0002 (1).png as floorplan1.png")
    print("2. Save @0000-0003 (1).png as floorplan2.png")
    print("3. Run: PYTHONPATH=/Users/nishanthkotla/Desktop/papercad python tools/process_image_samples.py")

if __name__ == "__main__":
    main()
