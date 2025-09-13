import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
from geometry_engine.api import GeometryEngine
from geometry_engine.primitives import Point, LineSegment
from geometry_engine.constraint_detector import ConstraintDetector
from geometry_engine.constraint_solver import ConstraintSolver
from geometry_engine.tests.sample_data import *

def test_basic_primitives():
    """Test basic geometric primitive operations"""
    print("Testing basic primitives...")
    
    # Test points
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert abs(p1.distance_to(p2) - 5.0) < 0.001
    
    # Test line segments
    line = LineSegment(p1, p2)
    assert abs(line.length() - 5.0) < 0.001
    assert abs(line.angle() - np.arctan2(4, 3)) < 0.001
    
    midpoint = line.midpoint()
    assert abs(midpoint.x - 1.5) < 0.001
    assert abs(midpoint.y - 2.0) < 0.001
    
    print("✓ Basic primitives working correctly")

def test_constraint_detection():
    """Test constraint detection on known geometry"""
    print("Testing constraint detection...")
    
    detector = ConstraintDetector()
    
    # Test perpendicular detection
    lines = [
        LineSegment(Point(0, 0), Point(10, 0)),     # Horizontal
        LineSegment(Point(10, 0), Point(10, 10))    # Vertical
    ]
    
    constraints = detector.detect_all_constraints(lines)
    assert len(constraints['perpendicular']) == 1
    assert constraints['perpendicular'][0] == (0, 1)
    
    # Test parallel detection
    lines = [
        LineSegment(Point(0, 0), Point(10, 0)),     # Horizontal 1
        LineSegment(Point(0, 5), Point(10, 5))      # Horizontal 2
    ]
    
    constraints = detector.detect_all_constraints(lines)
    assert len(constraints['parallel']) == 1
    
    print("✓ Constraint detection working correctly")

def test_messy_rectangle_cleanup():
    """Test the full pipeline on a messy hand-drawn rectangle"""
    print("Testing messy rectangle cleanup...")
    
    engine = GeometryEngine()
    messy_lines = get_messy_rectangle()
    
    result = engine.process_raw_geometry(messy_lines)
    
    # Check that we have 4 lines
    assert len(result['lines']) == 4
    
    # Check that perpendicular constraints were detected
    assert len(result['constraints']['perpendicular']) > 0
    
    # Verify angles are close to cardinal directions
    angles = [line.angle() for line in result['lines']]
    cardinal_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    
    for angle in angles:
        # Normalize angle to [0, 2π]
        normalized_angle = angle % (2 * np.pi)
        min_distance = min(abs(normalized_angle - ca) for ca in cardinal_angles)
        assert min_distance < 0.1, f"Angle {np.degrees(normalized_angle)} not close to cardinal direction"
    
    print("✓ Messy rectangle successfully cleaned up")
    print(f"  - Applied {result['statistics']['constraints_applied']} constraints")
    print(f"  - Found {result['statistics']['perpendicular_pairs']} perpendicular pairs")

def test_room_with_door():
    """Test processing a room with door symbols"""
    print("Testing room with door processing...")
    
    engine = GeometryEngine()
    lines, symbols, text = get_room_with_door()
    
    result = engine.process_raw_geometry(lines, symbols, text)
    
    # Should have walls and door arc
    assert len(result['lines']) > 5
    assert len(result['arcs']) == 1
    
    # Should have detected text
    assert len(result['metadata']['text']) == 1
    assert result['metadata']['text'][0]['text'] == '20 ft'
    
    print("✓ Room with door processed successfully")
    print(f"  - Lines: {len(result['lines'])}")
    print(f"  - Arcs: {len(result['arcs'])}")
    print(f"  - Text elements: {len(result['metadata']['text'])}")

def test_engineering_diagram():
    """Test processing of precise engineering diagram"""
    print("Testing engineering diagram...")
    
    engine = GeometryEngine()
    lines = get_engineering_diagram()
    
    result = engine.process_raw_geometry(lines)
    
    # Should detect many equal length constraints
    equal_length_pairs = result['constraints']['equal_length']
    assert len(equal_length_pairs) > 2, "Should detect multiple equal length constraints"
    
    # Should detect parallel constraints
    parallel_pairs = result['constraints']['parallel']
    assert len(parallel_pairs) > 3, "Should detect multiple parallel constraints"
    
    print("✓ Engineering diagram processed successfully")
    print(f"  - Equal length pairs: {len(equal_length_pairs)}")
    print(f"  - Parallel pairs: {len(parallel_pairs)}")

def visualize_before_after(raw_lines, processed_result, title="Geometry Processing"):
    """Visualize geometry before and after processing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before
    ax1.set_title("Before Processing")
    for line in raw_lines:
        ax1.plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=1, alpha=0.7)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # After
    ax2.set_title("After Processing")
    for line in processed_result['lines']:
        ax2.plot([line.start.x, line.end.x], [line.start.y, line.end.y], 'b-', linewidth=2)
    
    # Show arcs
    for arc in processed_result.get('arcs', []):
        theta = np.linspace(arc.start_angle, arc.end_angle, 50)
        x = arc.center.x + arc.radius * np.cos(theta)
        y = arc.center.y + arc.radius * np.sin(theta)
        ax2.plot(x, y, 'g-', linewidth=2)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{title} - {processed_result['statistics']['constraints_applied']} constraints applied")
    plt.tight_layout()
    plt.show()

def run_comprehensive_test():
    """Run all test cases and generate visualizations"""
    print("Running comprehensive geometry engine test suite...")
    print("=" * 60)
    
    # Basic tests
    test_basic_primitives()
    test_constraint_detection()
    
    # Integration tests
    test_messy_rectangle_cleanup()
    test_room_with_door()
    test_engineering_diagram()
    
    # Additional scenarios
    print("\nTesting U-shaped room...")
    engine = GeometryEngine()
    u_lines = get_u_shaped_room()
    u_result = engine.process_raw_geometry(u_lines)
    assert len(u_result['lines']) == len(u_lines)
    assert len(u_result['constraints']['perpendicular']) >= 6
    print("✓ U-shaped room processed")
    
    print("\nTesting corridor with doors...")
    corridor = get_corridor_with_doors()
    c_result = engine.process_raw_geometry(corridor)
    assert len(c_result['lines']) == len(corridor)
    assert len(c_result['constraints']['parallel']) >= 6
    print("✓ Corridor processed")
    
    print("\nTesting rotated noisy rectangle alignment...")
    rot = get_rotated_noisy_rect()
    r_result = engine.process_raw_geometry(rot)
    # After solving, expect near-cardinal angles
    for line in r_result['lines']:
        ang = line.angle() % (2*np.pi)
        assert min(abs(ang-0), abs(ang-np.pi/2), abs(ang-np.pi), abs(ang-3*np.pi/2)) < 0.15
    print("✓ Rotated noisy rectangle aligned")
    
    print("\n" + "=" * 60)
    print("All tests passed! Generating visualizations...")
    
    engine = GeometryEngine()
    
    # Test cases with visualization
    test_cases = [
        ("Messy Rectangle", get_messy_rectangle()),
        ("L-Shaped Room", get_l_shaped_room()),
        ("Engineering Diagram", get_engineering_diagram()),
        ("Simple Apartment", get_simple_apartment()),
        ("Office Layout", get_office_layout())
    ]
    
    for name, raw_lines in test_cases:
        print(f"\nProcessing {name}...")
        result = engine.process_raw_geometry(raw_lines)
        
        print(f"  Original lines: {result['statistics']['original_lines']}")
        print(f"  Final lines: {result['statistics']['final_lines']}")
        print(f"  Constraints applied: {result['statistics']['constraints_applied']}")
        
        # Show DXF entities that would be generated
        entities = engine.get_dxf_entities(result)
        print(f"  DXF entities: {len(entities)}")
        
        # Visualize if matplotlib is available
        try:
            visualize_before_after(raw_lines, result, name)
        except:
            print("  (Visualization skipped)")
    
    print("\n" + "=" * 60)
    print("Geometry Engine Phase 1 Complete!")
    print("✓ All core modules implemented and tested")
    print("✓ Ready for integration with CV pipeline")

if __name__ == "__main__":
    run_comprehensive_test()
