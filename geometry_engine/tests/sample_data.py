import numpy as np

def get_perfect_rectangle():
    """Perfect rectangle for baseline testing"""
    return [
        [0, 0, 100, 0],      # Bottom
        [100, 0, 100, 50],   # Right
        [100, 50, 0, 50],    # Top
        [0, 50, 0, 0]        # Left
    ]

def get_messy_rectangle():
    """Hand-drawn rectangle with typical human errors"""
    return [
        [1.2, 0.8, 98.7, 1.1],    # Bottom (slightly tilted)
        [98.7, 1.1, 99.2, 49.3],  # Right (slightly tilted)
        [99.2, 49.3, 2.1, 50.8],  # Top (slightly tilted)
        [2.1, 50.8, 1.2, 0.8]     # Left (slightly tilted)
    ]

def get_l_shaped_room():
    """L-shaped floor plan for complex constraint testing"""
    return [
        [0, 0, 60, 0],       # Bottom main
        [60, 0, 60, 30],     # Vertical main
        [60, 30, 100, 30],   # Extension bottom
        [100, 30, 100, 50],  # Extension right
        [100, 50, 0, 50],    # Top
        [0, 50, 0, 0]        # Left main
    ]

def get_room_with_door():
    """Rectangle with door opening and swing arc"""
    lines = [
        [0, 0, 40, 0],       # Bottom left
        [60, 0, 100, 0],     # Bottom right
        [100, 0, 100, 50],   # Right
        [100, 50, 0, 50],    # Top
        [0, 50, 0, 0],       # Left
        [40, 0, 40, 20],     # Door frame left
        [60, 0, 60, 20]      # Door frame right
    ]
    
    symbols = [
        {
            'class': 'door',
            'bbox': [40, 0, 60, 20],
            'confidence': 0.95
        }
    ]
    
    text = [
        {
            'text': '20 ft',
            'bbox': [[105, 25], [125, 25], [125, 35], [105, 35]]
        }
    ]
    
    return lines, symbols, text

def get_architectural_plan():
    """More complex architectural drawing with multiple rooms"""
    return [
        # Main room
        [0, 0, 150, 0],
        [150, 0, 150, 100],
        [150, 100, 0, 100],
        [0, 100, 0, 0],
        
        # Interior wall
        [50, 0, 50, 60],
        [50, 80, 50, 100],
        
        # Second room division
        [0, 60, 50, 60],
        [70, 60, 150, 60],
        
        # Window representations
        [75, 0, 125, 0],  # Thicker line for window
        [0, 30, 0, 50]    # Side window
    ]

def get_noisy_sketch():
    """Very rough sketch with significant noise"""
    base_rect = get_messy_rectangle()
    
    # Add random noise to coordinates
    noisy_rect = []
    for line in base_rect:
        noise = np.random.normal(0, 0.5, 4)
        noisy_line = [coord + n for coord, n in zip(line, noise)]
        noisy_rect.append(noisy_line)
    
    # Add some random short lines (noise)
    noise_lines = [
        [25, 15, 27, 18],
        [75, 35, 76, 33],
        [15, 40, 16, 42]
    ]
    
    return noisy_rect + noise_lines

def get_engineering_diagram():
    """Technical diagram with precise constraints"""
    return [
        # Main structure
        [0, 0, 100, 0],
        [100, 0, 100, 60],
        [100, 60, 0, 60],
        [0, 60, 0, 0],
        
        # Interior features with equal spacing
        [20, 0, 20, 60],    # Vertical 1
        [40, 0, 40, 60],    # Vertical 2 (equal spacing)
        [60, 0, 60, 60],    # Vertical 3 (equal spacing)
        [80, 0, 80, 60],    # Vertical 4 (equal spacing)
        
        # Horizontal divisions
        [0, 20, 100, 20],   # Horizontal 1
        [0, 40, 100, 40],   # Horizontal 2 (equal spacing)
        
        # Diagonal braces
        [0, 0, 20, 20],
        [80, 0, 100, 20]
    ]

def get_floor_plan_with_dimensions():
    """Floor plan with dimension text"""
    lines = [
        [10, 10, 110, 10],   # 100 unit bottom wall
        [110, 10, 110, 70],  # 60 unit right wall
        [110, 70, 10, 70],   # 100 unit top wall
        [10, 70, 10, 10]     # 60 unit left wall
    ]
    
    text = [
        {'text': '100 ft', 'bbox': [[50, 5], [70, 5], [70, 8], [50, 8]]},
        {'text': '60 ft', 'bbox': [[112, 35], [125, 35], [125, 45], [112, 45]]},
        {'text': 'Living Room', 'bbox': [[40, 35], [80, 35], [80, 45], [40, 45]]},
        {'text': 'Scale: 1"=10ft', 'bbox': [[10, 75], [50, 75], [50, 80], [10, 80]]}
    ]
    
    return lines, [], text

# Real-world inspired test data from online floor plan datasets
def get_simple_apartment():
    """Based on typical small apartment layouts"""
    return [
        # Outer walls
        [50, 50, 250, 50],   # Bottom
        [250, 50, 250, 180], # Right  
        [250, 180, 50, 180], # Top
        [50, 180, 50, 50],   # Left
        
        # Bedroom wall
        [50, 120, 150, 120],
        [150, 120, 150, 180],
        
        # Bathroom wall
        [150, 50, 150, 90],
        [150, 90, 200, 90],
        [200, 90, 200, 50],
        
        # Kitchen counter
        [200, 90, 200, 120],
        [200, 120, 230, 120]
    ]

def get_office_layout():
    """Office space with cubicles"""
    return [
        # Perimeter
        [0, 0, 300, 0],
        [300, 0, 300, 200],
        [300, 200, 0, 200],
        [0, 200, 0, 0],
        
        # Conference room
        [50, 150, 150, 150],
        [150, 150, 150, 200],
        
        # Cubicle grid
        [100, 0, 100, 100],
        [200, 0, 200, 100],
        [0, 50, 300, 50],
        [0, 100, 300, 100],
        
        # Interior corridors
        [50, 100, 50, 150],
        [250, 100, 250, 200]
    ]

def get_u_shaped_room():
    """U-shaped layout to test multiple corners and preserved orientations"""
    return [
        [0, 0, 200, 0],
        [200, 0, 200, 150],
        [200, 150, 150, 150],
        [150, 150, 150, 50],
        [150, 50, 50, 50],
        [50, 50, 50, 150],
        [50, 150, 0, 150],
        [0, 150, 0, 0]
    ]

def get_corridor_with_doors():
    """Long corridor with three door openings (frames only)"""
    return [
        # Corridor walls
        [0, 0, 300, 0],
        [0, 40, 300, 40],
        [0, 0, 0, 40],
        [300, 0, 300, 40],
        
        # Door 1 (left wall)
        [0, 10, 20, 10],
        [0, 30, 20, 30],
        
        # Door 2 (right wall)
        [300, 12, 280, 12],
        [300, 28, 280, 28],
        
        # Door 3 (middle)
        [140, 0, 140, 20],
        [160, 0, 160, 20]
    ]

def get_rotated_noisy_rect(angle_deg: float = 8.0):
    """Rectangle rotated slightly and with jitter; solver should axis-align it"""
    import math
    theta = math.radians(angle_deg)
    rect = [
        [-50, -30, 50, -30],
        [50, -30, 50, 30],
        [50, 30, -50, 30],
        [-50, 30, -50, -30]
    ]
    rot = []
    for x1, y1, x2, y2 in rect:
        rx1 = x1 * math.cos(theta) - y1 * math.sin(theta)
        ry1 = x1 * math.sin(theta) + y1 * math.cos(theta)
        rx2 = x2 * math.cos(theta) - y2 * math.sin(theta)
        ry2 = x2 * math.sin(theta) + y2 * math.cos(theta)
        # add small noise
        n = np.random.normal(0, 0.6, 4)
        rot.append([rx1 + n[0], ry1 + n[1], rx2 + n[2], ry2 + n[3]])
    return rot
