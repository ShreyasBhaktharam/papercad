import numpy as np
from typing import List, Tuple, Dict, Set
from .primitives import Point, LineSegment, Arc
from .symmetry_detector import SymmetryDetector

class ConstraintDetector:
    def __init__(self, 
                 angle_tolerance: float = 0.1,
                 distance_tolerance: float = 2.0,
                 length_tolerance: float = 0.1):
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
        self.length_tolerance = length_tolerance
        self.symmetry_detector = SymmetryDetector(angle_tolerance, distance_tolerance)
    
    def detect_all_constraints(self, 
                             line_segments: List[LineSegment], 
                             arcs: List[Arc] = None) -> Dict[str, List[Tuple]]:
        """Detect all geometric constraints in the given primitives"""
        constraints = {
            'perpendicular': [],
            'parallel': [],
            'collinear': [],
            'equal_length': [],
            'tangent': [],
            'concentric': [],
            'symmetric': []
        }
        
        if not line_segments:
            return constraints
        
        constraints['perpendicular'] = self.detect_perpendicular_pairs(line_segments)
        constraints['parallel'] = self.detect_parallel_pairs(line_segments)
        constraints['collinear'] = self.detect_collinear_segments(line_segments)
        constraints['equal_length'] = self.detect_equal_length_pairs(line_segments)
        
        if arcs:
            constraints['tangent'] = self.detect_tangency(line_segments, arcs)
            constraints['concentric'] = self.detect_concentric_arcs(arcs)
        
        # Detect symmetries
        symmetry_axes = self.symmetry_detector.detect_symmetries(line_segments, arcs)
        if symmetry_axes:
            constraints['symmetric'] = self.symmetry_detector.get_symmetry_constraints(line_segments, symmetry_axes)
        
        return constraints
    
    def detect_perpendicular_pairs(self, line_segments: List[LineSegment]) -> List[Tuple[int, int]]:
        """Find pairs of lines that are approximately perpendicular"""
        perpendicular_pairs = []
        
        for i, line1 in enumerate(line_segments):
            for j, line2 in enumerate(line_segments[i+1:], start=i+1):
                angle1 = line1.angle()
                angle2 = line2.angle()
                angle_diff = abs(angle1 - angle2)
                
                # Normalize angle difference to [0, pi]
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                if abs(angle_diff - np.pi/2) < self.angle_tolerance:
                    perpendicular_pairs.append((i, j))
        
        return perpendicular_pairs
    
    def detect_parallel_pairs(self, line_segments: List[LineSegment]) -> List[Tuple[int, int]]:
        """Find pairs of lines that are approximately parallel"""
        parallel_pairs = []
        
        for i, line1 in enumerate(line_segments):
            for j, line2 in enumerate(line_segments[i+1:], start=i+1):
                angle1 = line1.angle()
                angle2 = line2.angle()
                angle_diff = abs(angle1 - angle2)
                
                # Normalize angle difference
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                
                if angle_diff < self.angle_tolerance or abs(angle_diff - np.pi) < self.angle_tolerance:
                    parallel_pairs.append((i, j))
        
        return parallel_pairs
    
    def detect_collinear_segments(self, line_segments: List[LineSegment]) -> List[Tuple[int, int]]:
        """Find line segments that should be joined as they are collinear"""
        collinear_pairs = []
        
        for i, line1 in enumerate(line_segments):
            for j, line2 in enumerate(line_segments[i+1:], start=i+1):
                if self._are_collinear(line1, line2):
                    collinear_pairs.append((i, j))
        
        return collinear_pairs
    
    def detect_equal_length_pairs(self, line_segments: List[LineSegment]) -> List[Tuple[int, int]]:
        """Find pairs of lines with approximately equal length"""
        equal_length_pairs = []
        
        for i, line1 in enumerate(line_segments):
            for j, line2 in enumerate(line_segments[i+1:], start=i+1):
                len1 = line1.length()
                len2 = line2.length()
                
                if len1 > 0 and len2 > 0:
                    relative_diff = abs(len1 - len2) / max(len1, len2)
                    if relative_diff < self.length_tolerance:
                        equal_length_pairs.append((i, j))
        
        return equal_length_pairs
    
    def detect_tangency(self, line_segments: List[LineSegment], arcs: List[Arc]) -> List[Tuple]:
        """Find line segments that are tangent to arcs"""
        tangent_pairs = []
        
        for i, line in enumerate(line_segments):
            for j, arc in enumerate(arcs):
                if self._is_tangent_to_arc(line, arc):
                    tangent_pairs.append(('line', i, 'arc', j))
        
        # Check arc-to-arc tangency
        for i, arc1 in enumerate(arcs):
            for j, arc2 in enumerate(arcs[i+1:], start=i+1):
                if self._are_arcs_tangent(arc1, arc2):
                    tangent_pairs.append(('arc', i, 'arc', j))
        
        return tangent_pairs
    
    def detect_concentric_arcs(self, arcs: List[Arc]) -> List[Tuple[int, int]]:
        """Find pairs of arcs with the same center"""
        concentric_pairs = []
        
        for i, arc1 in enumerate(arcs):
            for j, arc2 in enumerate(arcs[i+1:], start=i+1):
                if arc1.center.distance_to(arc2.center) < self.distance_tolerance:
                    concentric_pairs.append((i, j))
        
        return concentric_pairs
    
    def _are_collinear(self, line1: LineSegment, line2: LineSegment) -> bool:
        """Check if two line segments are collinear and close enough to merge"""
        # Check if angles are similar
        angle1 = line1.angle()
        angle2 = line2.angle()
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        if angle_diff > self.angle_tolerance and abs(angle_diff - np.pi) > self.angle_tolerance:
            return False
        
        # Check if endpoints are close
        endpoints = [line1.start, line1.end, line2.start, line2.end]
        min_distance = float('inf')
        
        for i, p1 in enumerate(endpoints):
            for j, p2 in enumerate(endpoints[i+1:], start=i+1):
                if (i < 2) != (j < 2):  # Different line segments
                    min_distance = min(min_distance, p1.distance_to(p2))
        
        return min_distance < self.distance_tolerance
    
    def _is_tangent_to_arc(self, line: LineSegment, arc: Arc) -> bool:
        """Check if a line segment is tangent to an arc using precise geometric calculation"""
        # Vector from arc center to line start
        cx, cy = arc.center.x, arc.center.y
        x1, y1 = line.start.x, line.start.y
        x2, y2 = line.end.x, line.end.y
        
        # Line direction vector
        dx = x2 - x1
        dy = y2 - y1
        line_length_sq = dx * dx + dy * dy
        
        if line_length_sq < 1e-10:  # Degenerate line
            return False
        
        # Project center onto line
        t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / line_length_sq))
        
        # Closest point on line segment to arc center
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from center to closest point
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        dist = np.sqrt(dist_sq)
        
        # Check if tangent (distance equals radius)
        return abs(dist - arc.radius) < self.distance_tolerance
    
    def _are_arcs_tangent(self, arc1: Arc, arc2: Arc) -> bool:
        """Check if two arcs are tangent (externally or internally)"""
        center_distance = arc1.center.distance_to(arc2.center)
        
        # External tangency: distance = r1 + r2
        external_tangent = abs(center_distance - (arc1.radius + arc2.radius)) < self.distance_tolerance
        
        # Internal tangency: distance = |r1 - r2|
        internal_tangent = abs(center_distance - abs(arc1.radius - arc2.radius)) < self.distance_tolerance
        
        return external_tangent or internal_tangent
