import numpy as np
from typing import List, Dict, Tuple, Optional
from .primitives import Point, LineSegment, Arc
from .vectorization import vectorize_from_raw, merge_nearby_endpoints
from .constraint_detector import ConstraintDetector
from .constraint_solver import ConstraintSolver
from .ocr_processor import OCRProcessor
from .performance import FastConstraintSolver, perf_monitor
from .conflict_resolver import ConflictResolver
from .room_detector import RoomDetector

class GeometryEngine:
    """Main API interface for the geometry processing pipeline"""
    
    def __init__(self, 
                 angle_tolerance: float = 0.1,
                 distance_tolerance: float = 2.0,
                 length_tolerance: float = 0.1,
                 performance_mode: bool = False):
        self.constraint_detector = ConstraintDetector(
            angle_tolerance=angle_tolerance,
            distance_tolerance=distance_tolerance,
            length_tolerance=length_tolerance
        )
        self.constraint_solver = ConstraintSolver()
        self.fast_solver = FastConstraintSolver()
        self.ocr_processor = OCRProcessor()
        self.conflict_resolver = ConflictResolver()
        self.room_detector = RoomDetector()
        self.performance_mode = performance_mode
    
    def process_raw_geometry(self, 
                           raw_lines: List[List[float]],
                           raw_symbols: Optional[List[dict]] = None,
                           raw_text: Optional[List[dict]] = None) -> Dict:
        """
        Main processing function that takes raw CV output and returns clean parametric geometry
        
        Args:
            raw_lines: List of [x1, y1, x2, y2] line coordinates
            raw_symbols: List of detected symbols with bbox and class
            raw_text: List of OCR results with text and bbox
            
        Returns:
            Dictionary containing processed geometry and metadata
        """
        # Step 1: Vectorization
        line_segments, arcs, metadata = vectorize_from_raw(raw_lines, raw_symbols, raw_text)
        
        # Step 2: Merge nearby endpoints (disabled for now to debug)
        # line_segments = merge_nearby_endpoints(line_segments)
        
        # Step 3: Process OCR for scaling
        ocr_dimensions = self.ocr_processor.extract_dimensions(raw_text or [])
        
        # Step 4: Detect constraints
        constraints = self.constraint_detector.detect_all_constraints(line_segments, arcs)
        
        # Step 4.1: Resolve constraint conflicts
        constraints = self.conflict_resolver.resolve_conflicts(constraints)
        
        # Step 5: Solve constraints (with performance optimization)
        if self.performance_mode and len(line_segments) > 100:
            solved_lines = self.fast_solver.solve_fast(line_segments, constraints)
            solved_arcs = arcs  # Fast mode doesn't process arcs for speed
        else:
            solved_lines, solved_arcs = self.constraint_solver.solve_constraints(
                line_segments, constraints, arcs
            )
        
        # Step 6: Apply OCR-based scaling
        if ocr_dimensions['measurements'] or ocr_dimensions['scale_factor']:
            solved_lines, scale_factor = self.ocr_processor.apply_scaling(
                solved_lines, ocr_dimensions, target_unit='feet'
            )
            metadata['scale_factor'] = scale_factor
            metadata['ocr_dimensions'] = ocr_dimensions
        
        # Step 7: Detect rooms (closed polygons)
        rooms = self.room_detector.detect_rooms(solved_lines)
        if raw_text:
            rooms = self.room_detector.classify_rooms(rooms, raw_text)
        
        # Step 8: Prepare output
        result = {
            'lines': solved_lines,
            'arcs': solved_arcs,
            'rooms': rooms,
            'constraints': constraints,
            'metadata': metadata,
            'statistics': {
                'original_lines': len(line_segments),
                'final_lines': len(solved_lines),
                'rooms_detected': len(rooms),
                'total_floor_area': sum(room.area for room in rooms),
                'constraints_applied': sum(len(v) for v in constraints.values()),
                'perpendicular_pairs': len(constraints.get('perpendicular', [])),
                'parallel_pairs': len(constraints.get('parallel', [])),
                'equal_length_pairs': len(constraints.get('equal_length', []))
            }
        }
        
        return result
    
    def get_dxf_entities(self, processed_geometry: Dict) -> List[Dict]:
        """
        Convert processed geometry to DXF-ready entity descriptions
        
        Args:
            processed_geometry: Output from process_raw_geometry
            
        Returns:
            List of DXF entity dictionaries
        """
        entities = []
        
        # Add line segments
        for i, line in enumerate(processed_geometry['lines']):
            entities.append({
                'type': 'LINE',
                'layer': 'WALLS',
                'start': (line.start.x, line.start.y),
                'end': (line.end.x, line.end.y),
                'id': f'line_{i}'
            })
        
        # Add arcs
        for i, arc in enumerate(processed_geometry['arcs']):
            entities.append({
                'type': 'ARC',
                'layer': 'DOORS',
                'center': (arc.center.x, arc.center.y),
                'radius': arc.radius,
                'start_angle': np.degrees(arc.start_angle),
                'end_angle': np.degrees(arc.end_angle),
                'id': f'arc_{i}'
            })
        
        # Add symbols as blocks
        for i, symbol in enumerate(processed_geometry['metadata']['symbols']):
            entities.append({
                'type': 'INSERT',
                'layer': 'SYMBOLS',
                'block_name': symbol['class'].upper(),
                'position': (symbol['bbox'][0], symbol['bbox'][1]),
                'scale': (1.0, 1.0),
                'rotation': 0,
                'id': f'symbol_{i}'
            })
        
        # Add text
        for i, text_item in enumerate(processed_geometry['metadata']['text']):
            entities.append({
                'type': 'TEXT',
                'layer': 'DIMENSIONS',
                'text': text_item['text'],
                'position': (text_item['bbox'][0][0], text_item['bbox'][0][1]),
                'height': 2.0,
                'id': f'text_{i}'
            })
        
        return entities
    
    def get_performance_metrics(self) -> Dict:
        """Get performance statistics for optimization"""
        return perf_monitor.get_stats()
    
    def clear_performance_metrics(self):
        """Clear performance monitoring data"""
        perf_monitor.clear()
