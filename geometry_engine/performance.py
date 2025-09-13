import numpy as np
import time
from typing import List, Dict, Any
from functools import wraps
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Container for performance measurement data"""
    operation: str
    duration: float
    input_size: int
    memory_usage: float = 0.0
    iterations: int = 1

class PerformanceMonitor:
    """Monitor and optimize geometry engine performance"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.enabled = True
    
    def measure(self, operation_name: str):
        """Decorator to measure function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Estimate input size
                input_size = self._estimate_input_size(args, kwargs)
                
                metric = PerformanceMetrics(
                    operation=operation_name,
                    duration=end_time - start_time,
                    input_size=input_size
                )
                self.metrics.append(metric)
                
                return result
            return wrapper
        return decorator
    
    def _estimate_input_size(self, args, kwargs) -> int:
        """Estimate the complexity of input data"""
        total_size = 0
        
        for arg in args:
            if hasattr(arg, '__len__'):
                total_size += len(arg)
            elif hasattr(arg, '__sizeof__'):
                total_size += 1
        
        for value in kwargs.values():
            if hasattr(value, '__len__'):
                total_size += len(value)
            elif hasattr(value, '__sizeof__'):
                total_size += 1
        
        return total_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics:
            return {}
        
        stats = {}
        for operation in set(m.operation for m in self.metrics):
            op_metrics = [m for m in self.metrics if m.operation == operation]
            durations = [m.duration for m in op_metrics]
            input_sizes = [m.input_size for m in op_metrics]
            
            stats[operation] = {
                'count': len(op_metrics),
                'total_time': sum(durations),
                'avg_time': np.mean(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'avg_input_size': np.mean(input_sizes),
                'throughput': np.mean(input_sizes) / np.mean(durations) if np.mean(durations) > 0 else 0
            }
        
        return stats
    
    def clear(self):
        """Clear performance metrics"""
        self.metrics.clear()

# Global performance monitor instance
perf_monitor = PerformanceMonitor()

class OptimizedConstraintDetector:
    """Optimized version of constraint detection using vectorized operations"""
    
    def __init__(self, angle_tolerance: float = 0.1, distance_tolerance: float = 2.0):
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance
    
    @perf_monitor.measure("vectorized_perpendicular_detection")
    def detect_perpendicular_vectorized(self, line_segments: List) -> List:
        """Vectorized perpendicular detection for better performance"""
        if len(line_segments) < 2:
            return []
        
        # Convert to numpy arrays for vectorized operations
        angles = np.array([self._line_angle(line) for line in line_segments])
        
        # Create pairwise angle difference matrix
        angle_diff_matrix = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
        
        # Normalize angle differences
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2*np.pi - angle_diff_matrix)
        angle_diff_matrix = np.minimum(angle_diff_matrix, np.pi - angle_diff_matrix)
        
        # Find perpendicular pairs
        perpendicular_mask = np.abs(angle_diff_matrix - np.pi/2) < self.angle_tolerance
        
        # Get indices of perpendicular pairs (upper triangle only)
        i_indices, j_indices = np.where(np.triu(perpendicular_mask, k=1))
        
        return list(zip(i_indices, j_indices))
    
    @perf_monitor.measure("vectorized_parallel_detection")
    def detect_parallel_vectorized(self, line_segments: List) -> List:
        """Vectorized parallel detection for better performance"""
        if len(line_segments) < 2:
            return []
        
        angles = np.array([self._line_angle(line) for line in line_segments])
        
        # Create pairwise angle difference matrix
        angle_diff_matrix = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2*np.pi - angle_diff_matrix)
        
        # Find parallel pairs
        parallel_mask = (angle_diff_matrix < self.angle_tolerance) | \
                       (np.abs(angle_diff_matrix - np.pi) < self.angle_tolerance)
        
        # Get indices (upper triangle only, exclude diagonal)
        i_indices, j_indices = np.where(np.triu(parallel_mask, k=1))
        
        return list(zip(i_indices, j_indices))
    
    def _line_angle(self, line) -> float:
        """Calculate line angle efficiently"""
        return np.arctan2(line.end.y - line.start.y, line.end.x - line.start.x)

class GeometryCache:
    """Cache for expensive geometric computations"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_count: Dict[str, int] = {}
    
    def get(self, key: str) -> Any:
        """Get cached value"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()

# Global geometry cache
geo_cache = GeometryCache()

def hash_geometry(geometry) -> str:
    """Create hash key for geometry objects"""
    if hasattr(geometry, 'start') and hasattr(geometry, 'end'):
        # LineSegment
        return f"line_{geometry.start.x:.3f}_{geometry.start.y:.3f}_{geometry.end.x:.3f}_{geometry.end.y:.3f}"
    elif hasattr(geometry, 'center') and hasattr(geometry, 'radius'):
        # Arc
        return f"arc_{geometry.center.x:.3f}_{geometry.center.y:.3f}_{geometry.radius:.3f}_{geometry.start_angle:.3f}_{geometry.end_angle:.3f}"
    else:
        return str(hash(str(geometry)))

class FastConstraintSolver:
    """Performance-optimized constraint solver"""
    
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.1):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.optimized_detector = OptimizedConstraintDetector()
    
    @perf_monitor.measure("fast_constraint_solving")
    def solve_fast(self, line_segments: List, constraints: Dict) -> List:
        """Fast constraint solving with early termination"""
        if not line_segments:
            return line_segments
        
        solved_lines = [self._copy_line(line) for line in line_segments]
        
        # Use vectorized operations for large numbers of constraints
        if len(line_segments) > 50:
            # Re-detect constraints using vectorized methods for better accuracy
            constraints['perpendicular'] = self.optimized_detector.detect_perpendicular_vectorized(solved_lines)
            constraints['parallel'] = self.optimized_detector.detect_parallel_vectorized(solved_lines)
        
        for iteration in range(self.max_iterations):
            previous_lines = [self._copy_line(line) for line in solved_lines]
            
            # Apply only the most important constraints for speed
            solved_lines = self._fast_apply_constraints(solved_lines, constraints)
            
            # Early termination check
            if self._fast_convergence_check(previous_lines, solved_lines):
                break
        
        return solved_lines
    
    def _fast_apply_constraints(self, line_segments: List, constraints: Dict) -> List:
        """Apply constraints with performance optimizations"""
        # Only apply perpendicular and parallel constraints for speed
        # These are the most visually important
        
        # Batch process perpendicular constraints
        perp_pairs = constraints.get('perpendicular', [])
        if perp_pairs:
            line_segments = self._batch_apply_perpendicular(line_segments, perp_pairs)
        
        # Batch process parallel constraints
        parallel_pairs = constraints.get('parallel', [])
        if parallel_pairs:
            line_segments = self._batch_apply_parallel(line_segments, parallel_pairs)
        
        return line_segments
    
    def _batch_apply_perpendicular(self, line_segments: List, perp_pairs: List) -> List:
        """Apply perpendicular constraints in batch for performance"""
        if not perp_pairs:
            return line_segments
        
        # Convert to numpy for vectorized operations
        angles = np.array([self._line_angle(line) for line in line_segments])
        
        for i, j in perp_pairs:
            if i < len(angles) and j < len(angles):
                # Snap both to cardinal directions
                angle1 = self._round_to_cardinal(angles[i])
                angle2 = angle1 + np.pi/2
                
                angles[i] = angle1
                angles[j] = angle2
        
        # Update line segments with new angles
        for idx, angle in enumerate(angles):
            if idx < len(line_segments):
                line_segments[idx] = self._rotate_line_to_angle(line_segments[idx], angle)
        
        return line_segments
    
    def _batch_apply_parallel(self, line_segments: List, parallel_pairs: List) -> List:
        """Apply parallel constraints in batch for performance"""
        if not parallel_pairs:
            return line_segments
        
        angles = np.array([self._line_angle(line) for line in line_segments])
        
        for i, j in parallel_pairs:
            if i < len(angles) and j < len(angles):
                # Use the angle that's closer to cardinal direction
                angle1 = angles[i]
                angle2 = angles[j]
                
                cardinal1 = self._round_to_cardinal(angle1)
                cardinal2 = self._round_to_cardinal(angle2)
                
                # Choose the angle with smaller adjustment needed
                if abs(angle1 - cardinal1) <= abs(angle2 - cardinal2):
                    target_angle = cardinal1
                else:
                    target_angle = cardinal2
                
                angles[i] = target_angle
                angles[j] = target_angle
        
        # Update line segments
        for idx, angle in enumerate(angles):
            if idx < len(line_segments):
                line_segments[idx] = self._rotate_line_to_angle(line_segments[idx], angle)
        
        return line_segments
    
    def _line_angle(self, line) -> float:
        """Calculate line angle efficiently"""
        return np.arctan2(line.end.y - line.start.y, line.end.x - line.start.x)
    
    def _round_to_cardinal(self, angle: float) -> float:
        """Round to nearest cardinal direction"""
        degrees = np.degrees(angle) % 360
        cardinals = [0, 90, 180, 270]
        nearest = min(cardinals, key=lambda x: min(abs(degrees - x), abs(degrees - x - 360), abs(degrees - x + 360)))
        return np.radians(nearest)
    
    def _rotate_line_to_angle(self, line, target_angle: float):
        """Rotate line to target angle around midpoint"""
        midpoint_x = (line.start.x + line.end.x) / 2
        midpoint_y = (line.start.y + line.end.y) / 2
        length = np.sqrt((line.end.x - line.start.x)**2 + (line.end.y - line.start.y)**2)
        
        half_length = length / 2
        
        from .primitives import Point, LineSegment
        new_start = Point(
            midpoint_x - half_length * np.cos(target_angle),
            midpoint_y - half_length * np.sin(target_angle)
        )
        new_end = Point(
            midpoint_x + half_length * np.cos(target_angle),
            midpoint_y + half_length * np.sin(target_angle)
        )
        
        return LineSegment(new_start, new_end)
    
    def _copy_line(self, line):
        """Fast line copy"""
        from .primitives import Point, LineSegment
        return LineSegment(
            Point(line.start.x, line.start.y),
            Point(line.end.x, line.end.y)
        )
    
    def _fast_convergence_check(self, prev_lines: List, curr_lines: List) -> bool:
        """Fast convergence check using vectorized operations"""
        if len(prev_lines) != len(curr_lines):
            return False
        
        # Vectorized distance calculation
        prev_coords = np.array([[line.start.x, line.start.y, line.end.x, line.end.y] for line in prev_lines])
        curr_coords = np.array([[line.start.x, line.start.y, line.end.x, line.end.y] for line in curr_lines])
        
        total_movement = np.sum(np.abs(prev_coords - curr_coords))
        return total_movement < self.convergence_threshold
