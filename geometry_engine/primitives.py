import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))

@dataclass
class LineSegment:
    start: Point
    end: Point
    
    def length(self) -> float:
        return self.start.distance_to(self.end)
    
    def angle(self) -> float:
        """Returns angle in radians"""
        return np.arctan2(self.end.y - self.start.y, self.end.x - self.start.x)
    
    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)
    
    def distance_to_point(self, point: Point) -> float:
        """Calculate minimum distance from point to line segment"""
        A = point.x - self.start.x
        B = point.y - self.start.y
        C = self.end.x - self.start.x
        D = self.end.y - self.start.y
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return point.distance_to(self.start)
        
        param = dot / len_sq
        
        if param < 0:
            xx = self.start.x
            yy = self.start.y
        elif param > 1:
            xx = self.end.x
            yy = self.end.y
        else:
            xx = self.start.x + param * C
            yy = self.start.y + param * D
        
        dx = point.x - xx
        dy = point.y - yy
        return np.sqrt(dx * dx + dy * dy)
    
    def intersects_with(self, other: 'LineSegment') -> Optional[Point]:
        """Find intersection point between two line segments"""
        x1, y1 = self.start.x, self.start.y
        x2, y2 = self.end.x, self.end.y
        x3, y3 = other.start.x, other.start.y
        x4, y4 = other.end.x, other.end.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            return Point(px, py)
        
        return None

@dataclass
class Arc:
    center: Point
    radius: float
    start_angle: float
    end_angle: float
    
    def length(self) -> float:
        angle_diff = abs(self.end_angle - self.start_angle)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        return self.radius * angle_diff
    
    def start_point(self) -> Point:
        return Point(
            self.center.x + self.radius * np.cos(self.start_angle),
            self.center.y + self.radius * np.sin(self.start_angle)
        )
    
    def end_point(self) -> Point:
        return Point(
            self.center.x + self.radius * np.cos(self.end_angle),
            self.center.y + self.radius * np.sin(self.end_angle)
        )
    
    def is_point_on_arc(self, point: Point, tolerance: float = 1e-6) -> bool:
        """Check if a point lies on the arc"""
        # Check distance to center
        dist_to_center = point.distance_to(self.center)
        if abs(dist_to_center - self.radius) > tolerance:
            return False
        
        # Check angle
        angle = np.arctan2(point.y - self.center.y, point.x - self.center.x)
        angle = angle % (2 * np.pi)
        start_angle = self.start_angle % (2 * np.pi)
        end_angle = self.end_angle % (2 * np.pi)
        
        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        else:  # Arc crosses 0 radians
            return angle >= start_angle or angle <= end_angle
    
    def distance_to_point(self, point: Point) -> float:
        """Minimum distance from point to arc"""
        dist_to_center = point.distance_to(self.center)
        
        # If point is at center
        if dist_to_center < 1e-10:
            return self.radius
        
        # Angle to point
        angle = np.arctan2(point.y - self.center.y, point.x - self.center.x)
        angle = angle % (2 * np.pi)
        start_angle = self.start_angle % (2 * np.pi)
        end_angle = self.end_angle % (2 * np.pi)
        
        # Check if angle is within arc
        in_arc = False
        if start_angle <= end_angle:
            in_arc = start_angle <= angle <= end_angle
        else:
            in_arc = angle >= start_angle or angle <= end_angle
        
        if in_arc:
            return abs(dist_to_center - self.radius)
        else:
            # Distance to closest endpoint
            start_pt = self.start_point()
            end_pt = self.end_point()
            return min(point.distance_to(start_pt), point.distance_to(end_pt))
