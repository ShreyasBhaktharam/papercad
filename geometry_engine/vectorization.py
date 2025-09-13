import numpy as np
from typing import List, Tuple
from .primitives import Point, LineSegment, Arc

def vectorize_from_raw(raw_lines: List[List[float]], 
                      raw_symbols: List[dict] = None,
                      raw_text: List[dict] = None) -> Tuple[List[LineSegment], List[Arc], List[dict]]:
    """
    Convert raw CV output into geometric primitive objects
    
    Args:
        raw_lines: List of [x1, y1, x2, y2] coordinates
        raw_symbols: List of symbol detections with bbox and class
        raw_text: List of OCR results with text and bbox
    
    Returns:
        Tuple of (line_segments, arcs, metadata)
    """
    line_segments = []
    arcs = []
    metadata = {
        'symbols': raw_symbols or [],
        'text': raw_text or [],
        'scale_factor': 1.0,
        'units': 'pixels'
    }
    
    for raw_line in raw_lines:
        if len(raw_line) >= 4:
            start = Point(raw_line[0], raw_line[1])
            end = Point(raw_line[2], raw_line[3])
            
            if start.distance_to(end) > 1.0:
                line_segments.append(LineSegment(start, end))
    
    if raw_symbols:
        for symbol in raw_symbols:
            if symbol.get('class') == 'door' and 'bbox' in symbol:
                arc = _extract_door_arc(symbol['bbox'])
                if arc:
                    arcs.append(arc)
    
    return line_segments, arcs, metadata

def _extract_door_arc(bbox: List[float]) -> Arc:
    """Extract door swing arc from door symbol bounding box"""
    x1, y1, x2, y2 = bbox
    center = Point(x1, (y1 + y2) / 2)
    radius = abs(x2 - x1)
    return Arc(center, radius, 0, np.pi/2)

def merge_nearby_endpoints(line_segments: List[LineSegment], 
                          tolerance: float = 2.0) -> List[LineSegment]:
    """Merge line segments with endpoints that are very close"""
    if not line_segments:
        return []
    
    merged_segments = []
    used_indices = set()
    
    for i, segment in enumerate(line_segments):
        if i in used_indices:
            continue
            
        current_segment = segment
        used_indices.add(i)
        
        for j, other_segment in enumerate(line_segments[i+1:], start=i+1):
            if j in used_indices:
                continue
                
            if (current_segment.end.distance_to(other_segment.start) < tolerance):
                current_segment = LineSegment(current_segment.start, other_segment.end)
                used_indices.add(j)
            elif (current_segment.end.distance_to(other_segment.end) < tolerance):
                current_segment = LineSegment(current_segment.start, other_segment.start)
                used_indices.add(j)
            elif (current_segment.start.distance_to(other_segment.start) < tolerance):
                current_segment = LineSegment(other_segment.end, current_segment.end)
                used_indices.add(j)
            elif (current_segment.start.distance_to(other_segment.end) < tolerance):
                current_segment = LineSegment(other_segment.start, current_segment.end)
                used_indices.add(j)
        
        merged_segments.append(current_segment)
    
    return merged_segments
