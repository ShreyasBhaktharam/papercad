import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from .primitives import Point, LineSegment

class OCRProcessor:
    """Process OCR text data and extract dimensional information"""
    
    def __init__(self):
        # Common unit patterns
        self.unit_patterns = {
            'feet': r'(\d+\.?\d*)\s*[\'\"]*\s*(?:ft|feet|f)\b',
            'inches': r'(\d+\.?\d*)\s*[\'\"]*\s*(?:in|inch|inches|\")\b',
            'meters': r'(\d+\.?\d*)\s*(?:m|meter|meters)\b',
            'centimeters': r'(\d+\.?\d*)\s*(?:cm|centimeter|centimeters)\b',
            'millimeters': r'(\d+\.?\d*)\s*(?:mm|millimeter|millimeters)\b',
            'points': r'(\d+\.?\d*)\s*(?:pt|points?)\b',
            'pixels': r'(\d+\.?\d*)\s*(?:px|pixel|pixels)\b'
        }
        
        # Scale text patterns
        self.scale_patterns = [
            r'scale:\s*1\s*[\'\"]*\s*=\s*(\d+\.?\d*)\s*(?:ft|feet)',
            r'1\s*[\'\"]*\s*=\s*(\d+\.?\d*)\s*(?:ft|feet)',
            r'(\d+\.?\d*)\s*(?:ft|feet)\s*per\s*(?:inch|in)',
            r'1:(\d+)',
            r'(\d+):1'
        ]
    
    def extract_dimensions(self, text_data: List[Dict]) -> Dict:
        """Extract dimensional information from OCR text data"""
        dimensions = {
            'measurements': [],
            'scale_factor': None,
            'primary_unit': None,
            'labels': []
        }
        
        for text_item in text_data:
            text = text_item.get('text', '').lower().strip()
            bbox = text_item.get('bbox', [])
            
            if not text:
                continue
            
            # Extract measurements
            measurement = self._extract_measurement(text)
            if measurement:
                dimensions['measurements'].append({
                    'value': measurement['value'],
                    'unit': measurement['unit'],
                    'text': text,
                    'bbox': bbox,
                    'position': self._bbox_center(bbox)
                })
            
            # Extract scale information
            scale = self._extract_scale(text)
            if scale:
                dimensions['scale_factor'] = scale
            
            # Store non-dimensional labels
            if not measurement and not scale:
                dimensions['labels'].append({
                    'text': text,
                    'bbox': bbox,
                    'position': self._bbox_center(bbox)
                })
        
        # Determine primary unit
        if dimensions['measurements']:
            unit_counts = {}
            for m in dimensions['measurements']:
                unit_counts[m['unit']] = unit_counts.get(m['unit'], 0) + 1
            dimensions['primary_unit'] = max(unit_counts, key=unit_counts.get)
        
        return dimensions
    
    def apply_scaling(self, line_segments: List[LineSegment], 
                     dimensions: Dict, 
                     target_unit: str = 'feet') -> Tuple[List[LineSegment], float]:
        """Apply real-world scaling to geometry based on OCR dimensions"""
        
        if not dimensions['measurements']:
            return line_segments, 1.0
        
        # Find the best dimension to use for scaling
        scale_factor = self._calculate_scale_factor(line_segments, dimensions, target_unit)
        
        if scale_factor == 1.0:
            return line_segments, scale_factor
        
        # Apply scaling to all line segments
        scaled_lines = []
        for line in line_segments:
            scaled_start = Point(line.start.x * scale_factor, line.start.y * scale_factor)
            scaled_end = Point(line.end.x * scale_factor, line.end.y * scale_factor)
            scaled_lines.append(LineSegment(scaled_start, scaled_end))
        
        return scaled_lines, scale_factor
    
    def _extract_measurement(self, text: str) -> Optional[Dict]:
        """Extract numerical measurement with unit from text"""
        for unit, pattern in self.unit_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    return {'value': value, 'unit': unit}
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_scale(self, text: str) -> Optional[float]:
        """Extract scale factor from text"""
        for pattern in self.scale_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    scale_value = float(match.group(1))
                    if 'per' in pattern or ':' in pattern:
                        return scale_value
                    else:
                        return scale_value
                except (ValueError, IndexError):
                    continue
        return None
    
    def _bbox_center(self, bbox: List) -> Point:
        """Calculate center point of bounding box"""
        if len(bbox) >= 4:
            # Handle [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
            if isinstance(bbox[0], list):
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]
                return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
            # Handle [x1, y1, x2, y2] format
            else:
                return Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        return Point(0, 0)
    
    def _calculate_scale_factor(self, line_segments: List[LineSegment], 
                               dimensions: Dict, target_unit: str) -> float:
        """Calculate scale factor by matching OCR dimensions to geometry"""
        
        measurements = dimensions['measurements']
        if not measurements:
            return 1.0
        
        # Convert all measurements to target unit
        converted_measurements = []
        for m in measurements:
            converted_value = self._convert_units(m['value'], m['unit'], target_unit)
            if converted_value is not None:
                converted_measurements.append({
                    **m,
                    'converted_value': converted_value
                })
        
        if not converted_measurements:
            return 1.0
        
        # Find the best matching line segment for each measurement
        best_scale = 1.0
        best_confidence = 0
        
        for measurement in converted_measurements:
            pos = measurement['position']
            target_length = measurement['converted_value']
            
            # Find closest line segment
            closest_line = None
            min_distance = float('inf')
            
            for line in line_segments:
                # Check if the text position is near this line
                line_center = line.midpoint()
                distance = pos.distance_to(line_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_line = line
            
            if closest_line and min_distance < 50:  # Reasonable proximity threshold
                pixel_length = closest_line.length()
                if pixel_length > 10:  # Avoid tiny lines
                    scale = target_length / pixel_length
                    confidence = 1.0 / (1.0 + min_distance / 10.0)  # Higher confidence for closer text
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_scale = scale
        
        return best_scale if best_confidence > 0.1 else 1.0
    
    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert between different units"""
        # Conversion factors to meters
        to_meters = {
            'feet': 0.3048,
            'inches': 0.0254,
            'meters': 1.0,
            'centimeters': 0.01,
            'millimeters': 0.001,
            'points': 0.000352778,  # 1/72 inch
            'pixels': None  # Can't convert without DPI
        }
        
        if from_unit == to_unit:
            return value
        
        if from_unit not in to_meters or to_unit not in to_meters:
            return None
        
        if to_meters[from_unit] is None or to_meters[to_unit] is None:
            return None
        
        # Convert via meters
        meters = value * to_meters[from_unit]
        return meters / to_meters[to_unit]
