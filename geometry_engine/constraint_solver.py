import numpy as np
from typing import List, Dict, Tuple, Optional
from .primitives import Point, LineSegment, Arc

class ConstraintSolver:
    def __init__(self, max_iterations: int = 10, convergence_threshold: float = 0.1):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def solve_constraints(self, 
                         line_segments: List[LineSegment],
                         constraints: Dict[str, List[Tuple]],
                         arcs: List[Arc] = None) -> Tuple[List[LineSegment], List[Arc]]:
        """Apply constraint solving to snap geometry to ideal positions"""
        if not line_segments:
            return line_segments, arcs or []
        
        solved_lines = [self._copy_line(line) for line in line_segments]
        solved_arcs = [arc for arc in (arcs or [])]
        
        # Lock each line's initial orientation to preserve vertical and horizontal members.
        # This prevents the optimizer from averaging angles and flattening geometry.
        orientation_targets = self._compute_orientation_targets(solved_lines)
        
        for iteration in range(self.max_iterations):
            previous_lines = [self._copy_line(line) for line in solved_lines]
            previous_arcs = [self._copy_arc(arc) for arc in solved_arcs]
            
            # Apply constraints in order of importance
            solved_lines = self._apply_perpendicular_constraints(solved_lines, constraints.get('perpendicular', []), orientation_targets)
            solved_lines = self._apply_parallel_constraints(solved_lines, constraints.get('parallel', []), orientation_targets)
            solved_lines = self._apply_equal_length_constraints(solved_lines, constraints.get('equal_length', []))
            
            # Apply advanced constraints
            solved_lines, solved_arcs = self._apply_tangency_constraints(solved_lines, solved_arcs, constraints.get('tangent', []))
            solved_arcs = self._apply_concentricity_constraints(solved_arcs, constraints.get('concentric', []))
            solved_lines = self._apply_symmetry_constraints(solved_lines, constraints.get('symmetric', []))
            
            # Final cleanup
            solved_lines = self._snap_endpoints(solved_lines)
            
            if self._has_converged(previous_lines, solved_lines) and self._arcs_converged(previous_arcs, solved_arcs):
                break
        
        return solved_lines, solved_arcs
    
    def _apply_perpendicular_constraints(self, 
                                       line_segments: List[LineSegment], 
                                       perpendicular_pairs: List[Tuple[int, int]],
                                       orientation_targets: List[float]) -> List[LineSegment]:
        """Snap lines to be exactly perpendicular"""
        for i, j in perpendicular_pairs:
            if i < len(line_segments) and j < len(line_segments):
                line1 = line_segments[i]
                line2 = line_segments[j]
                
                # Respect locked orientation targets
                target_angle1 = orientation_targets[i]
                target_angle2 = orientation_targets[j]
                
                # Apply adjustments
                line_segments[i] = self._rotate_line_to_angle(line1, target_angle1)
                line_segments[j] = self._rotate_line_to_angle(line2, target_angle2)
        
        return line_segments
    
    def _apply_parallel_constraints(self, 
                                  line_segments: List[LineSegment], 
                                  parallel_pairs: List[Tuple[int, int]],
                                  orientation_targets: List[float]) -> List[LineSegment]:
        """Snap lines to be exactly parallel"""
        for i, j in parallel_pairs:
            if i < len(line_segments) and j < len(line_segments):
                line1 = line_segments[i]
                line2 = line_segments[j]
                
                # Use each line's own locked target; keeps verticals vertical and horizontals horizontal
                line_segments[i] = self._rotate_line_to_angle(line1, orientation_targets[i])
                line_segments[j] = self._rotate_line_to_angle(line2, orientation_targets[j])
        
        return line_segments
    
    def _apply_equal_length_constraints(self, 
                                      line_segments: List[LineSegment], 
                                      equal_length_pairs: List[Tuple[int, int]]) -> List[LineSegment]:
        """Make lines have exactly equal length"""
        for i, j in equal_length_pairs:
            if i < len(line_segments) and j < len(line_segments):
                line1 = line_segments[i]
                line2 = line_segments[j]
                
                len1 = line1.length()
                len2 = line2.length()
                target_length = (len1 + len2) / 2
                
                line_segments[i] = self._scale_line_to_length(line1, target_length)
                line_segments[j] = self._scale_line_to_length(line2, target_length)
        
        return line_segments
    
    def _snap_endpoints(self, line_segments: List[LineSegment], tolerance: float = 3.0) -> List[LineSegment]:
        """Snap nearby endpoints together"""
        snapped_lines = []
        
        for line in line_segments:
            new_start = line.start
            new_end = line.end
            
            # Find closest endpoints for start and end
            for other_line in line_segments:
                if other_line == line:
                    continue
                
                if new_start.distance_to(other_line.start) < tolerance:
                    new_start = other_line.start
                elif new_start.distance_to(other_line.end) < tolerance:
                    new_start = other_line.end
                
                if new_end.distance_to(other_line.start) < tolerance:
                    new_end = other_line.start
                elif new_end.distance_to(other_line.end) < tolerance:
                    new_end = other_line.end
            
            snapped_lines.append(LineSegment(new_start, new_end))
        
        return snapped_lines
    
    def _compute_orientation_targets(self, line_segments: List[LineSegment]) -> List[float]:
        """Compute target angle (0 or pi/2) for each line based on initial orientation.
        Lines nearer to horizontal snap to 0/π, lines nearer to vertical snap to π/2/3π/2.
        """
        targets: List[float] = []
        for line in line_segments:
            angle = line.angle() % (2 * np.pi)
            # Distance to horizontal and vertical
            dist_to_horizontal = min(abs(angle - 0), abs(angle - np.pi))
            dist_to_vertical = min(abs(angle - np.pi/2), abs(angle - 3*np.pi/2))
            if dist_to_horizontal <= dist_to_vertical:
                targets.append(0.0)
            else:
                targets.append(np.pi/2)
        return targets
    
    def _apply_tangency_constraints(self, line_segments: List[LineSegment], 
                                   arcs: List[Arc], 
                                   tangent_pairs: List[Tuple]) -> Tuple[List[LineSegment], List[Arc]]:
        """Apply tangency constraints between lines and arcs"""
        for constraint in tangent_pairs:
            if len(constraint) == 4:
                type1, idx1, type2, idx2 = constraint
                
                if type1 == 'line' and type2 == 'arc':
                    if idx1 < len(line_segments) and idx2 < len(arcs):
                        line = line_segments[idx1]
                        arc = arcs[idx2]
                        adjusted_line = self._make_line_tangent_to_arc(line, arc)
                        if adjusted_line:
                            line_segments[idx1] = adjusted_line
                
                elif type1 == 'arc' and type2 == 'arc':
                    if idx1 < len(arcs) and idx2 < len(arcs):
                        arc1 = arcs[idx1]
                        arc2 = arcs[idx2]
                        adjusted_arcs = self._make_arcs_tangent(arc1, arc2)
                        if adjusted_arcs:
                            arcs[idx1], arcs[idx2] = adjusted_arcs
        
        return line_segments, arcs
    
    def _apply_concentricity_constraints(self, arcs: List[Arc], 
                                        concentric_pairs: List[Tuple[int, int]]) -> List[Arc]:
        """Make arcs concentric by aligning their centers"""
        for i, j in concentric_pairs:
            if i < len(arcs) and j < len(arcs):
                arc1 = arcs[i]
                arc2 = arcs[j]
                
                # Use the average center position
                avg_center = Point(
                    (arc1.center.x + arc2.center.x) / 2,
                    (arc1.center.y + arc2.center.y) / 2
                )
                
                arcs[i] = Arc(avg_center, arc1.radius, arc1.start_angle, arc1.end_angle)
                arcs[j] = Arc(avg_center, arc2.radius, arc2.start_angle, arc2.end_angle)
        
        return arcs
    
    def _make_line_tangent_to_arc(self, line: LineSegment, arc: Arc) -> Optional[LineSegment]:
        """Adjust line to be tangent to arc while preserving orientation"""
        # Calculate the perpendicular distance from arc center to line
        cx, cy = arc.center.x, arc.center.y
        x1, y1 = line.start.x, line.start.y
        x2, y2 = line.end.x, line.end.y
        
        # Line direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        
        if length < 1e-10:
            return None
        
        # Unit direction vector
        ux = dx / length
        uy = dy / length
        
        # Unit normal vector (perpendicular to line)
        nx = -uy
        ny = ux
        
        # Move line so it's tangent to the arc
        # Distance from current line to center
        current_dist = abs((cy - y1) * ux - (cx - x1) * uy)
        
        # Offset needed to make tangent
        offset = arc.radius - current_dist
        
        # Adjust line position
        new_start = Point(x1 + offset * nx, y1 + offset * ny)
        new_end = Point(x2 + offset * nx, y2 + offset * ny)
        
        return LineSegment(new_start, new_end)
    
    def _make_arcs_tangent(self, arc1: Arc, arc2: Arc) -> Optional[Tuple[Arc, Arc]]:
        """Adjust arc positions to make them tangent"""
        center_dist = arc1.center.distance_to(arc2.center)
        
        if center_dist < 1e-10:
            return None
        
        # Determine if external or internal tangency is closer
        external_target = arc1.radius + arc2.radius
        internal_target = abs(arc1.radius - arc2.radius)
        
        target_dist = external_target if abs(center_dist - external_target) < abs(center_dist - internal_target) else internal_target
        
        if target_dist < 1e-10:
            return None
        
        # Scale factor to achieve target distance
        scale = target_dist / center_dist
        
        # Calculate new centers
        mid_x = (arc1.center.x + arc2.center.x) / 2
        mid_y = (arc1.center.y + arc2.center.y) / 2
        
        offset_x = (arc1.center.x - mid_x) * scale
        offset_y = (arc1.center.y - mid_y) * scale
        
        new_center1 = Point(mid_x + offset_x, mid_y + offset_y)
        new_center2 = Point(mid_x - offset_x, mid_y - offset_y)
        
        new_arc1 = Arc(new_center1, arc1.radius, arc1.start_angle, arc1.end_angle)
        new_arc2 = Arc(new_center2, arc2.radius, arc2.start_angle, arc2.end_angle)
        
        return new_arc1, new_arc2
    
    def _apply_symmetry_constraints(self, line_segments: List[LineSegment], symmetric_constraints: List[Tuple]) -> List[LineSegment]:
        """Apply symmetry constraints to enforce perfect symmetry"""
        if not symmetric_constraints:
            return line_segments
        
        lines_copy = [LineSegment(Point(line.start.x, line.start.y), Point(line.end.x, line.end.y)) 
                     for line in line_segments]
        
        for constraint in symmetric_constraints:
            if len(constraint) >= 4:
                idx1, idx2, constraint_type, axis = constraint
                
                if (constraint_type == 'symmetric' and 
                    0 <= idx1 < len(lines_copy) and 
                    0 <= idx2 < len(lines_copy)):
                    
                    line1 = lines_copy[idx1]
                    line2 = lines_copy[idx2]
                    
                    # Calculate average position and enforce symmetry
                    reflected_start = axis.reflect_point(line2.start)
                    reflected_end = axis.reflect_point(line2.end)
                    
                    # Average the symmetric pair
                    avg_start1 = Point(
                        (line1.start.x + reflected_start.x) / 2,
                        (line1.start.y + reflected_start.y) / 2
                    )
                    avg_end1 = Point(
                        (line1.end.x + reflected_end.x) / 2,
                        (line1.end.y + reflected_end.y) / 2
                    )
                    
                    # Update both lines to be perfectly symmetric
                    lines_copy[idx1] = LineSegment(avg_start1, avg_end1)
                    lines_copy[idx2] = LineSegment(
                        axis.reflect_point(avg_start1),
                        axis.reflect_point(avg_end1)
                    )
        
        return lines_copy
    
    def _copy_arc(self, arc: Arc) -> Arc:
        """Create a deep copy of an arc"""
        return Arc(
            Point(arc.center.x, arc.center.y),
            arc.radius,
            arc.start_angle,
            arc.end_angle
        )
    
    def _arcs_converged(self, previous_arcs: List[Arc], current_arcs: List[Arc]) -> bool:
        """Check if arc constraints have converged"""
        if len(previous_arcs) != len(current_arcs):
            return False
        
        total_movement = 0.0
        for prev_arc, curr_arc in zip(previous_arcs, current_arcs):
            total_movement += prev_arc.center.distance_to(curr_arc.center)
            total_movement += abs(prev_arc.radius - curr_arc.radius)
        
        return total_movement < self.convergence_threshold

    def _round_to_cardinal(self, angle: float) -> float:
        """Round angle to nearest cardinal direction (0, 90, 180, 270 degrees)"""
        # Convert to degrees for easier calculation
        degrees = np.degrees(angle) % 360
        
        # Find nearest cardinal direction
        cardinals = [0, 90, 180, 270]
        nearest_cardinal = min(cardinals, key=lambda x: min(abs(degrees - x), abs(degrees - x - 360), abs(degrees - x + 360)))
        
        return np.radians(nearest_cardinal)
    
    def _rotate_line_to_angle(self, line: LineSegment, target_angle: float) -> LineSegment:
        """Rotate line around its midpoint to target angle"""
        midpoint = line.midpoint()
        length = line.length()
        
        # Calculate new endpoints
        half_length = length / 2
        new_start = Point(
            midpoint.x - half_length * np.cos(target_angle),
            midpoint.y - half_length * np.sin(target_angle)
        )
        new_end = Point(
            midpoint.x + half_length * np.cos(target_angle),
            midpoint.y + half_length * np.sin(target_angle)
        )
        
        return LineSegment(new_start, new_end)
    
    def _scale_line_to_length(self, line: LineSegment, target_length: float) -> LineSegment:
        """Scale line to target length while maintaining direction"""
        if line.length() == 0:
            return line
        
        midpoint = line.midpoint()
        angle = line.angle()
        half_length = target_length / 2
        
        new_start = Point(
            midpoint.x - half_length * np.cos(angle),
            midpoint.y - half_length * np.sin(angle)
        )
        new_end = Point(
            midpoint.x + half_length * np.cos(angle),
            midpoint.y + half_length * np.sin(angle)
        )
        
        return LineSegment(new_start, new_end)
    
    def _copy_line(self, line: LineSegment) -> LineSegment:
        """Create a deep copy of a line segment"""
        return LineSegment(
            Point(line.start.x, line.start.y),
            Point(line.end.x, line.end.y)
        )
    
    def _has_converged(self, previous_lines: List[LineSegment], current_lines: List[LineSegment]) -> bool:
        """Check if the constraint solving has converged"""
        if len(previous_lines) != len(current_lines):
            return False
        
        total_movement = 0.0
        for prev_line, curr_line in zip(previous_lines, current_lines):
            total_movement += prev_line.start.distance_to(curr_line.start)
            total_movement += prev_line.end.distance_to(curr_line.end)
        
        return total_movement < self.convergence_threshold
