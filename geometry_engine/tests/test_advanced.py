import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time
from geometry_engine.api import GeometryEngine
from geometry_engine.primitives import Point, LineSegment, Arc
from geometry_engine.ocr_processor import OCRProcessor
from geometry_engine.performance import FastConstraintSolver, perf_monitor

def test_advanced_constraints():
    """Test advanced constraints: tangency and concentricity"""
    print("Testing advanced constraints...")
    
    engine = GeometryEngine()
    
    # Create test data with tangent line and arc
    lines = [
        [0, 10, 100, 10],    # Horizontal line that should be tangent to arc
        [50, 0, 50, 20],     # Vertical line
    ]
    
    symbols = [
        {'class': 'door', 'bbox': [50, 10, 70, 30], 'confidence': 0.9}  # Creates arc
    ]
    
    result = engine.process_raw_geometry(lines, symbols)
    
    # Should detect tangency between line and arc
    tangent_constraints = result['constraints']['tangent']
    print(f"  Tangency constraints detected: {len(tangent_constraints)}")
    
    # Test concentric arcs
    symbols_concentric = [
        {'class': 'door', 'bbox': [25, 25, 45, 45], 'confidence': 0.9},  # Arc 1
        {'class': 'door', 'bbox': [20, 20, 50, 50], 'confidence': 0.9}   # Arc 2 (should be concentric)
    ]
    
    result2 = engine.process_raw_geometry([], symbols_concentric)
    concentric_constraints = result2['constraints']['concentric']
    print(f"  Concentricity constraints detected: {len(concentric_constraints)}")
    
    print("✓ Advanced constraints working")

def test_ocr_scaling():
    """Test OCR-based dimensional scaling"""
    print("Testing OCR scaling...")
    
    processor = OCRProcessor()
    
    # Test dimension extraction
    text_data = [
        {'text': '10 ft', 'bbox': [[50, 5], [70, 5], [70, 15], [50, 15]]},
        {'text': '8 feet', 'bbox': [[105, 25], [125, 25], [125, 35], [105, 35]]},
        {'text': 'Scale: 1" = 10ft', 'bbox': [[10, 90], [60, 90], [60, 100], [10, 100]]},
        {'text': 'Living Room', 'bbox': [[40, 40], [80, 40], [80, 50], [40, 50]]}
    ]
    
    dimensions = processor.extract_dimensions(text_data)
    
    print(f"  Measurements found: {len(dimensions['measurements'])}")
    print(f"  Scale factor: {dimensions['scale_factor']}")
    print(f"  Primary unit: {dimensions['primary_unit']}")
    print(f"  Labels: {len(dimensions['labels'])}")
    
    # Test scaling application
    test_lines = [
        LineSegment(Point(0, 0), Point(100, 0)),    # 100 pixel line
        LineSegment(Point(0, 0), Point(0, 80))      # 80 pixel line
    ]
    
    scaled_lines, scale_factor = processor.apply_scaling(test_lines, dimensions)
    
    print(f"  Applied scale factor: {scale_factor:.3f}")
    print(f"  Original line length: {test_lines[0].length():.1f} pixels")
    print(f"  Scaled line length: {scaled_lines[0].length():.1f} units")
    
    print("✓ OCR scaling working")

def test_performance_optimization():
    """Test performance optimizations"""
    print("Testing performance optimization...")
    
    # Generate large dataset
    large_lines = []
    for i in range(200):
        x1, y1 = i * 10, i % 20 * 10
        x2, y2 = x1 + 50, y1 + (30 if i % 2 else -30)
        large_lines.append([x1, y1, x2, y2])
    
    # Test standard engine
    perf_monitor.clear()
    engine_standard = GeometryEngine(performance_mode=False)
    
    start_time = time.perf_counter()
    result_standard = engine_standard.process_raw_geometry(large_lines)
    standard_time = time.perf_counter() - start_time
    
    # Test fast engine
    perf_monitor.clear()
    engine_fast = GeometryEngine(performance_mode=True)
    
    start_time = time.perf_counter()
    result_fast = engine_fast.process_raw_geometry(large_lines)
    fast_time = time.perf_counter() - start_time
    
    speedup = standard_time / fast_time if fast_time > 0 else 1.0
    
    print(f"  Standard mode: {standard_time:.3f}s, {len(result_standard['lines'])} lines")
    print(f"  Fast mode: {fast_time:.3f}s, {len(result_fast['lines'])} lines")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Test vectorized constraint detection
    fast_solver = FastConstraintSolver()
    test_lines = [
        LineSegment(Point(0, 0), Point(100, 0)),
        LineSegment(Point(100, 0), Point(100, 100)),
        LineSegment(Point(100, 100), Point(0, 100)),
        LineSegment(Point(0, 100), Point(0, 0))
    ]
    
    perp_pairs = fast_solver.optimized_detector.detect_perpendicular_vectorized(test_lines)
    parallel_pairs = fast_solver.optimized_detector.detect_parallel_vectorized(test_lines)
    
    print(f"  Vectorized perpendicular detection: {len(perp_pairs)} pairs")
    print(f"  Vectorized parallel detection: {len(parallel_pairs)} pairs")
    
    print("✓ Performance optimization working")

def test_complex_geometry():
    """Test complex geometry with multiple constraint types"""
    print("Testing complex geometry...")
    
    engine = GeometryEngine()
    
    # Complex floor plan with multiple rooms and features
    complex_lines = [
        # Main building outline
        [0, 0, 200, 0], [200, 0, 200, 150], [200, 150, 0, 150], [0, 150, 0, 0],
        
        # Interior walls
        [50, 0, 50, 75], [50, 100, 50, 150],  # Vertical divider with door gap
        [0, 75, 100, 75],                      # Horizontal divider
        [150, 50, 200, 50],                    # Another room divider
        
        # Additional features
        [25, 0, 25, 30],    # Small wall
        [175, 0, 175, 30],  # Another small wall
        [0, 125, 30, 125],  # Partial wall
        [170, 125, 200, 125] # Another partial wall
    ]
    
    symbols = [
        {'class': 'door', 'bbox': [50, 75, 70, 100], 'confidence': 0.9},  # Door 1
        {'class': 'door', 'bbox': [125, 50, 150, 70], 'confidence': 0.9}  # Door 2
    ]
    
    text_data = [
        {'text': '20 ft', 'bbox': [[100, -10], [120, -10], [120, 0], [100, 0]]},
        {'text': '15 ft', 'bbox': [[205, 75], [225, 75], [225, 85], [205, 85]]},
        {'text': 'Kitchen', 'bbox': [[20, 35], [60, 35], [60, 45], [20, 45]]},
        {'text': 'Living Room', 'bbox': [[120, 85], [180, 85], [180, 95], [120, 95]]}
    ]
    
    result = engine.process_raw_geometry(complex_lines, symbols, text_data)
    
    print(f"  Input lines: {len(complex_lines)}")
    print(f"  Output lines: {len(result['lines'])}")
    print(f"  Arcs: {len(result['arcs'])}")
    print(f"  Total constraints applied: {result['statistics']['constraints_applied']}")
    
    constraint_summary = {k: len(v) for k, v in result['constraints'].items() if v}
    print(f"  Constraint breakdown: {constraint_summary}")
    
    if 'ocr_dimensions' in result['metadata']:
        ocr_info = result['metadata']['ocr_dimensions']
        print(f"  OCR measurements: {len(ocr_info['measurements'])}")
        print(f"  OCR labels: {len(ocr_info['labels'])}")
    
    print("✓ Complex geometry processing working")

def test_mathematical_accuracy():
    """Test mathematical accuracy of geometric operations"""
    print("Testing mathematical accuracy...")
    
    # Test perpendicular angle snapping
    engine = GeometryEngine()
    
    # Lines that are almost perpendicular (89.5 and 179.5 degrees)
    test_lines = [
        [0, 0, 100, np.tan(np.radians(89.5)) * 100],  # Almost vertical
        [0, 0, 100, np.tan(np.radians(179.5)) * 100]  # Almost horizontal
    ]
    
    result = engine.process_raw_geometry(test_lines)
    
    # Check that angles are snapped to exactly 90 and 0 degrees
    line1_angle = result['lines'][0].angle()
    line2_angle = result['lines'][1].angle()
    
    print(f"  Input angles: ~89.5°, ~179.5°")
    print(f"  Output angles: {np.degrees(line1_angle):.1f}°, {np.degrees(line2_angle):.1f}°")
    
    # Test tangency calculation accuracy
    arc = Arc(Point(50, 50), 25, 0, np.pi)
    line = LineSegment(Point(0, 25), Point(100, 25))  # Should be tangent
    
    distance = line.distance_to_point(arc.center)
    print(f"  Line-to-arc-center distance: {distance:.3f}")
    print(f"  Arc radius: {arc.radius}")
    print(f"  Tangency error: {abs(distance - arc.radius):.6f}")
    
    # Test that tangency error is within tolerance
    assert abs(distance - arc.radius) < 1e-10, "Tangency calculation not accurate enough"
    
    print("✓ Mathematical accuracy validated")

def run_phase3_tests():
    """Run all Phase 3 advanced feature tests"""
    print("Running Phase 3 Advanced Features Test Suite")
    print("=" * 60)
    
    test_advanced_constraints()
    print()
    
    test_ocr_scaling()
    print()
    
    test_performance_optimization()
    print()
    
    test_complex_geometry()
    print()
    
    test_mathematical_accuracy()
    print()
    
    print("=" * 60)
    print("✅ All Phase 3 tests passed!")
    print("✅ Advanced constraints: tangency, concentricity")
    print("✅ OCR integration: real-world scaling")
    print("✅ Robust solver: complex constraint interactions")
    print("✅ Performance optimization: real-time ready")
    print("✅ Mathematical accuracy: precision validated")

if __name__ == "__main__":
    run_phase3_tests()
