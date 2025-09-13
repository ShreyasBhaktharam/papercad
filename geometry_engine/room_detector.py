import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from .primitives import Point, LineSegment

@dataclass
class Room:
    """Represents a detected room/closed polygon"""
    vertices: List[Point]
    boundary_lines: List[int]  # Indices of line segments forming the boundary
    area: float
    centroid: Point
    room_type: Optional[str] = None  # 'bedroom', 'bathroom', 'kitchen', etc.

class RoomDetector:
    """Detects rooms as closed polygons from line segments"""
    
    def __init__(self, junction_tolerance: float = 3.0):
        self.junction_tolerance = junction_tolerance
    
    def detect_rooms(self, line_segments: List[LineSegment]) -> List[Room]:
        """
        Detect closed polygons (rooms) from line segments
        
        Args:
            line_segments: List of line segments representing walls
            
        Returns:
            List of detected Room objects
        """
        if len(line_segments) < 3:
            return []
        
        # Build connectivity graph
        graph = self._build_connectivity_graph(line_segments)
        
        # Find closed cycles
        cycles = self._find_closed_cycles(graph, line_segments)
        
        # Convert cycles to Room objects
        rooms = []
        for cycle in cycles:
            room = self._cycle_to_room(cycle, line_segments)
            if room and room.area > 100:  # Filter out tiny polygons
                rooms.append(room)
        
        # Sort by area (largest first)
        rooms.sort(key=lambda r: r.area, reverse=True)
        
        return rooms
    
    def _build_connectivity_graph(self, line_segments: List[LineSegment]) -> Dict[int, List[int]]:
        """Build a graph where nodes are line endpoints and edges connect nearby endpoints"""
        graph = {}
        
        for i, line1 in enumerate(line_segments):
            graph[i] = []
            
            for j, line2 in enumerate(line_segments):
                if i == j:
                    continue
                
                # Check if lines share an endpoint or have nearby endpoints
                if self._lines_connected(line1, line2):
                    graph[i].append(j)
        
        return graph
    
    def _lines_connected(self, line1: LineSegment, line2: LineSegment) -> bool:
        """Check if two lines are connected at their endpoints"""
        endpoints1 = [line1.start, line1.end]
        endpoints2 = [line2.start, line2.end]
        
        for p1 in endpoints1:
            for p2 in endpoints2:
                if p1.distance_to(p2) < self.junction_tolerance:
                    return True
        return False
    
    def _find_closed_cycles(self, graph: Dict[int, List[int]], line_segments: List[LineSegment]) -> List[List[int]]:
        """Find closed cycles in the connectivity graph using DFS"""
        cycles = []
        visited_global = set()
        
        for start_node in graph:
            if start_node in visited_global:
                continue
            
            # Find cycles starting from this node
            node_cycles = self._dfs_cycles(graph, start_node, [], set(), start_node)
            
            # Filter and validate cycles
            for cycle in node_cycles:
                if len(cycle) >= 3 and self._is_valid_cycle(cycle, line_segments):
                    # Avoid duplicate cycles (same cycle with different starting points)
                    normalized_cycle = self._normalize_cycle(cycle)
                    if normalized_cycle not in [self._normalize_cycle(c) for c in cycles]:
                        cycles.append(cycle)
                        visited_global.update(cycle)
        
        return cycles
    
    def _dfs_cycles(self, graph: Dict[int, List[int]], current: int, path: List[int], 
                   visited: Set[int], start: int, max_depth: int = 8) -> List[List[int]]:
        """Depth-first search to find cycles"""
        if len(path) > max_depth:  # Prevent infinite loops
            return []
        
        if current in visited and current == start and len(path) >= 3:
            return [path[:]]  # Found a cycle
        
        if current in visited:
            return []  # Already visited, not a valid cycle
        
        visited.add(current)
        path.append(current)
        
        cycles = []
        for neighbor in graph.get(current, []):
            if neighbor == start and len(path) >= 3:
                cycles.append(path + [neighbor])
            elif neighbor not in visited or neighbor == start:
                cycles.extend(self._dfs_cycles(graph, neighbor, path[:], visited.copy(), start, max_depth))
        
        return cycles
    
    def _is_valid_cycle(self, cycle: List[int], line_segments: List[LineSegment]) -> bool:
        """Check if a cycle represents a valid closed polygon"""
        if len(cycle) < 3:
            return False
        
        # Check that consecutive lines in the cycle are actually connected
        for i in range(len(cycle)):
            line1_idx = cycle[i]
            line2_idx = cycle[(i + 1) % len(cycle)]
            
            if line1_idx >= len(line_segments) or line2_idx >= len(line_segments):
                return False
            
            line1 = line_segments[line1_idx]
            line2 = line_segments[line2_idx]
            
            if not self._lines_connected(line1, line2):
                return False
        
        return True
    
    def _normalize_cycle(self, cycle: List[int]) -> Tuple[int, ...]:
        """Normalize cycle representation to avoid duplicates"""
        if not cycle:
            return tuple()
        
        # Start from the smallest index and choose the lexicographically smaller direction
        min_idx = min(cycle)
        start_pos = cycle.index(min_idx)
        
        # Two possible directions
        forward = cycle[start_pos:] + cycle[:start_pos]
        backward = list(reversed(cycle[start_pos+1:] + cycle[:start_pos+1]))
        
        return tuple(min(forward, backward))
    
    def _cycle_to_room(self, cycle: List[int], line_segments: List[LineSegment]) -> Optional[Room]:
        """Convert a cycle of line indices to a Room object"""
        if len(cycle) < 3:
            return None
        
        # Get vertices by following the cycle
        vertices = []
        for i in range(len(cycle)):
            line1_idx = cycle[i]
            line2_idx = cycle[(i + 1) % len(cycle)]
            
            line1 = line_segments[line1_idx]
            line2 = line_segments[line2_idx]
            
            # Find the shared vertex between consecutive lines
            shared_vertex = self._find_shared_vertex(line1, line2)
            if shared_vertex:
                vertices.append(shared_vertex)
        
        if len(vertices) < 3:
            return None
        
        # Calculate area using shoelace formula
        area = self._calculate_polygon_area(vertices)
        
        # Calculate centroid
        centroid = self._calculate_centroid(vertices)
        
        return Room(
            vertices=vertices,
            boundary_lines=cycle,
            area=abs(area),
            centroid=centroid
        )
    
    def _find_shared_vertex(self, line1: LineSegment, line2: LineSegment) -> Optional[Point]:
        """Find the shared vertex between two connected lines"""
        tolerance = self.junction_tolerance
        
        for p1 in [line1.start, line1.end]:
            for p2 in [line2.start, line2.end]:
                if p1.distance_to(p2) < tolerance:
                    # Return the average position for better precision
                    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        
        return None
    
    def _calculate_polygon_area(self, vertices: List[Point]) -> float:
        """Calculate polygon area using the shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y
            area -= vertices[j].x * vertices[i].y
        
        return area / 2.0
    
    def _calculate_centroid(self, vertices: List[Point]) -> Point:
        """Calculate the centroid of a polygon"""
        if not vertices:
            return Point(0, 0)
        
        x_sum = sum(v.x for v in vertices)
        y_sum = sum(v.y for v in vertices)
        
        return Point(x_sum / len(vertices), y_sum / len(vertices))
    
    def classify_rooms(self, rooms: List[Room], text_labels: List[Dict] = None) -> List[Room]:
        """Classify rooms based on area, shape, and text labels"""
        classified_rooms = []
        
        for room in rooms:
            room_copy = Room(
                vertices=room.vertices,
                boundary_lines=room.boundary_lines,
                area=room.area,
                centroid=room.centroid,
                room_type=self._classify_room_type(room, text_labels)
            )
            classified_rooms.append(room_copy)
        
        return classified_rooms
    
    def _classify_room_type(self, room: Room, text_labels: List[Dict] = None) -> str:
        """Classify room type based on area and nearby text labels"""
        # Check for nearby text labels first
        if text_labels:
            for label in text_labels:
                if 'text' in label and 'bbox' in label:
                    label_text = label['text'].lower()
                    label_center = self._bbox_center(label['bbox'])
                    
                    if self._point_in_polygon(label_center, room.vertices):
                        # Common room type keywords
                        room_keywords = {
                            'bedroom': ['bedroom', 'bed', 'sleep'],
                            'bathroom': ['bathroom', 'bath', 'toilet', 'wc'],
                            'kitchen': ['kitchen', 'cook'],
                            'living': ['living', 'lounge', 'family'],
                            'dining': ['dining', 'eat'],
                            'office': ['office', 'study', 'work'],
                            'closet': ['closet', 'wardrobe', 'storage']
                        }
                        
                        for room_type, keywords in room_keywords.items():
                            if any(keyword in label_text for keyword in keywords):
                                return room_type
        
        # Fallback classification based on area
        if room.area < 500:
            return 'closet'
        elif room.area < 2000:
            return 'bathroom'
        elif room.area < 8000:
            return 'bedroom'
        else:
            return 'living_area'
    
    def _bbox_center(self, bbox: List) -> Point:
        """Calculate center of bounding box"""
        if isinstance(bbox[0], list):
            # [[x1,y1], [x2,y2], ...] format
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        else:
            # [x1, y1, x2, y2] format
            return Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _point_in_polygon(self, point: Point, vertices: List[Point]) -> bool:
        """Check if a point is inside a polygon using ray casting"""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0].x, vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n].x, vertices[i % n].y
            
            if point.y > min(p1y, p2y):
                if point.y <= max(p1y, p2y):
                    if point.x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point.y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point.x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
