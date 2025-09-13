import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .primitives import Point, LineSegment, Arc

@dataclass
class SymmetryAxis:
    """Represents a line of symmetry"""
    point: Point  # Point on the axis
    angle: float  # Angle of the axis (radians)
    confidence: float  # How strong the symmetry is (0-1)
    
    def reflect_point(self, p: Point) -> Point:
        """Reflect a point across this symmetry axis"""
        # Translate so axis passes through origin
        translated_x = p.x - self.point.x
        translated_y = p.y - self.point.y
        
        # Rotation matrix to align axis with x-axis
        cos_theta = np.cos(-self.angle)
        sin_theta = np.sin(-self.angle)
        
        # Rotate point
        rotated_x = cos_theta * translated_x - sin_theta * translated_y
        rotated_y = sin_theta * translated_x + cos_theta * translated_y
        
        # Reflect across x-axis (negate y)
        reflected_x = rotated_x
        reflected_y = -rotated_y
        
        # Rotate back
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        final_x = cos_theta * reflected_x - sin_theta * reflected_y
        final_y = sin_theta * reflected_x + cos_theta * reflected_y
        
        # Translate back
        return Point(final_x + self.point.x, final_y + self.point.y)

class SymmetryDetector:
    """Detects and enforces symmetry in geometric drawings"""
    
    def __init__(self, angle_tolerance: float = 0.1, distance_tolerance: float = 3.0):
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
    
    def detect_symmetries(self, line_segments: List[LineSegment], arcs: List[Arc] = None) -> List[SymmetryAxis]:
        """
        Detect lines of symmetry in the geometry
        
        Args:
            line_segments: List of line segments
            arcs: Optional list of arcs
            
        Returns:
            List of detected symmetry axes
        """
        if len(line_segments) < 2:
            return []
        
        # Get all geometric points
        points = self._extract_points(line_segments, arcs or [])
        
        # Try common symmetry axes
        axes = []
        
        # 1. Vertical symmetry (most common in architectural drawings)
        vertical_axis = self._detect_vertical_symmetry(points, line_segments)
        if vertical_axis:
            axes.append(vertical_axis)
        
        # 2. Horizontal symmetry
        horizontal_axis = self._detect_horizontal_symmetry(points, line_segments)
        if horizontal_axis:
            axes.append(horizontal_axis)
        
        # 3. Diagonal symmetries (45° angles)
        diagonal_axes = self._detect_diagonal_symmetries(points, line_segments)
        axes.extend(diagonal_axes)
        
        # 4. Custom angle symmetries
        custom_axes = self._detect_custom_symmetries(points, line_segments)
        axes.extend(custom_axes)
        
        # Sort by confidence
        axes.sort(key=lambda x: x.confidence, reverse=True)
        
        return axes
    
    def _extract_points(self, line_segments: List[LineSegment], arcs: List[Arc]) -> List[Point]:
        """Extract all unique points from geometry"""
        points = []
        
        # Add line endpoints
        for line in line_segments:
            points.extend([line.start, line.end])
        
        # Add arc centers and endpoints
        for arc in arcs:
            points.append(arc.center)
            points.append(arc.start_point())
            points.append(arc.end_point())
        
        # Remove duplicates (within tolerance)
        unique_points = []
        for point in points:
            is_duplicate = False
            for existing in unique_points:
                if point.distance_to(existing) < self.distance_tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points
    
    def _detect_vertical_symmetry(self, points: List[Point], line_segments: List[LineSegment]) -> Optional[SymmetryAxis]:
        """Detect vertical line of symmetry"""
        if len(points) < 4:
            return None
        
        # Find the center x-coordinate
        x_coords = [p.x for p in points]
        center_x = (min(x_coords) + max(x_coords)) / 2
        
        # Test vertical axis at center
        axis = SymmetryAxis(
            point=Point(center_x, 0),
            angle=np.pi/2,  # Vertical
            confidence=0.0
        )
        
        confidence = self._calculate_symmetry_confidence(points, line_segments, axis)
        axis.confidence = confidence
        
        return axis if confidence > 0.3 else None
    
    def _detect_horizontal_symmetry(self, points: List[Point], line_segments: List[LineSegment]) -> Optional[SymmetryAxis]:
        """Detect horizontal line of symmetry"""
        if len(points) < 4:
            return None
        
        # Find the center y-coordinate
        y_coords = [p.y for p in points]
        center_y = (min(y_coords) + max(y_coords)) / 2
        
        # Test horizontal axis at center
        axis = SymmetryAxis(
            point=Point(0, center_y),
            angle=0,  # Horizontal
            confidence=0.0
        )
        
        confidence = self._calculate_symmetry_confidence(points, line_segments, axis)
        axis.confidence = confidence
        
        return axis if confidence > 0.3 else None
    
    def _detect_diagonal_symmetries(self, points: List[Point], line_segments: List[LineSegment]) -> List[SymmetryAxis]:
        """Detect diagonal lines of symmetry (45° and 135°)"""
        axes = []
        
        if len(points) < 4:
            return axes
        
        # Get center point
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        center = Point(
            (min(x_coords) + max(x_coords)) / 2,
            (min(y_coords) + max(y_coords)) / 2
        )
        
        # Test 45° and 135° diagonals
        for angle in [np.pi/4, 3*np.pi/4]:
            axis = SymmetryAxis(
                point=center,
                angle=angle,
                confidence=0.0
            )
            
            confidence = self._calculate_symmetry_confidence(points, line_segments, axis)
            axis.confidence = confidence
            
            if confidence > 0.3:
                axes.append(axis)
        
        return axes
    
    def _detect_custom_symmetries(self, points: List[Point], line_segments: List[LineSegment]) -> List[SymmetryAxis]:
        """Detect symmetries at custom angles"""
        axes = []
        
        if len(points) < 6:
            return axes
        
        # Get center point
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        center = Point(
            (min(x_coords) + max(x_coords)) / 2,
            (min(y_coords) + max(y_coords)) / 2
        )
        
        # Test angles every 15 degrees
        for angle_deg in range(0, 180, 15):
            if angle_deg in [0, 45, 90, 135]:  # Skip already tested angles
                continue
            
            angle = np.radians(angle_deg)
            axis = SymmetryAxis(
                point=center,
                angle=angle,
                confidence=0.0
            )
            
            confidence = self._calculate_symmetry_confidence(points, line_segments, axis)
            axis.confidence = confidence
            
            if confidence > 0.4:  # Higher threshold for custom angles
                axes.append(axis)
        
        return axes
    
    def _calculate_symmetry_confidence(self, points: List[Point], line_segments: List[LineSegment], axis: SymmetryAxis) -> float:
        """Calculate how well the geometry matches the proposed symmetry axis"""
        if len(points) < 2:
            return 0.0
        
        matched_points = 0
        total_points = 0
        
        # Check point symmetry
        for point in points:
            reflected = axis.reflect_point(point)
            
            # Find if reflected point exists
            found_match = False
            for other_point in points:
                if reflected.distance_to(other_point) < self.distance_tolerance:
                    found_match = True
                    break
            
            if found_match:
                matched_points += 1
            total_points += 1
        
        point_confidence = matched_points / total_points if total_points > 0 else 0
        
        # Check line symmetry
        matched_lines = 0
        total_lines = 0
        
        for line in line_segments:
            reflected_start = axis.reflect_point(line.start)
            reflected_end = axis.reflect_point(line.end)
            
            # Find if reflected line exists
            found_match = False
            for other_line in line_segments:
                if (self._points_match(reflected_start, other_line.start, self.distance_tolerance) and
                    self._points_match(reflected_end, other_line.end, self.distance_tolerance)) or \
                   (self._points_match(reflected_start, other_line.end, self.distance_tolerance) and
                    self._points_match(reflected_end, other_line.start, self.distance_tolerance)):
                    found_match = True
                    break
            
            if found_match:
                matched_lines += 1
            total_lines += 1
        
        line_confidence = matched_lines / total_lines if total_lines > 0 else 0
        
        # Combined confidence (weighted average)
        return 0.6 * line_confidence + 0.4 * point_confidence
    
    def _points_match(self, p1: Point, p2: Point, tolerance: float) -> bool:
        """Check if two points match within tolerance"""
        return p1.distance_to(p2) < tolerance
    
    def enforce_symmetry(self, line_segments: List[LineSegment], axes: List[SymmetryAxis]) -> List[LineSegment]:
        """
        Enforce detected symmetries by adjusting geometry
        
        Args:
            line_segments: Original line segments
            axes: Detected symmetry axes
            
        Returns:
            Adjusted line segments with enforced symmetry
        """
        if not axes:
            return line_segments[:]
        
        # Use the highest confidence axis
        primary_axis = axes[0]
        
        adjusted_lines = []
        processed_indices = set()
        
        for i, line in enumerate(line_segments):
            if i in processed_indices:
                continue
            
            # Find symmetric partner
            reflected_start = primary_axis.reflect_point(line.start)
            reflected_end = primary_axis.reflect_point(line.end)
            
            partner_idx = None
            for j, other_line in enumerate(line_segments):
                if j == i or j in processed_indices:
                    continue
                
                if (self._points_match(reflected_start, other_line.start, self.distance_tolerance * 2) and
                    self._points_match(reflected_end, other_line.end, self.distance_tolerance * 2)) or \
                   (self._points_match(reflected_start, other_line.end, self.distance_tolerance * 2) and
                    self._points_match(reflected_end, other_line.start, self.distance_tolerance * 2)):
                    partner_idx = j
                    break
            
            if partner_idx is not None:
                # Enforce perfect symmetry by averaging positions
                partner = line_segments[partner_idx]
                
                # Calculate average line
                avg_start = Point(
                    (line.start.x + primary_axis.reflect_point(partner.start).x) / 2,
                    (line.start.y + primary_axis.reflect_point(partner.start).y) / 2
                )
                avg_end = Point(
                    (line.end.x + primary_axis.reflect_point(partner.end).x) / 2,
                    (line.end.y + primary_axis.reflect_point(partner.end).y) / 2
                )
                
                # Create symmetric pair
                adjusted_lines.append(LineSegment(avg_start, avg_end))
                reflected_avg_start = primary_axis.reflect_point(avg_start)
                reflected_avg_end = primary_axis.reflect_point(avg_end)
                adjusted_lines.append(LineSegment(reflected_avg_start, reflected_avg_end))
                
                processed_indices.update([i, partner_idx])
            else:
                # No symmetric partner, keep original
                adjusted_lines.append(line)
                processed_indices.add(i)
        
        return adjusted_lines
    
    def get_symmetry_constraints(self, line_segments: List[LineSegment], axes: List[SymmetryAxis]) -> List[Tuple]:
        """
        Generate symmetry constraints for the constraint solver
        
        Returns:
            List of tuples (line_idx1, line_idx2, 'symmetric', axis)
        """
        constraints = []
        
        for axis in axes:
            for i, line1 in enumerate(line_segments):
                for j, line2 in enumerate(line_segments[i+1:], start=i+1):
                    if self._are_symmetric(line1, line2, axis):
                        constraints.append((i, j, 'symmetric', axis))
        
        return constraints
    
    def _are_symmetric(self, line1: LineSegment, line2: LineSegment, axis: SymmetryAxis) -> bool:
        """Check if two lines are symmetric across an axis"""
        reflected_start = axis.reflect_point(line1.start)
        reflected_end = axis.reflect_point(line1.end)
        
        # Check if reflected line1 matches line2
        return ((self._points_match(reflected_start, line2.start, self.distance_tolerance) and
                 self._points_match(reflected_end, line2.end, self.distance_tolerance)) or
                (self._points_match(reflected_start, line2.end, self.distance_tolerance) and
                 self._points_match(reflected_end, line2.start, self.distance_tolerance)))
