#!/usr/bin/env python3
"""
Enhanced Floor Plan Processor - Creates Color-Coded CAD Output
Processes hand-drawn sketches into clean, color-coded floor plans with room identification
"""

import sys
import os
sys.path.append('/Users/nishanthkotla/Desktop/papercad')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from geometry_engine.api import GeometryEngine
from geometry_engine.primitives import Point, LineSegment
import random

class ColoredFloorPlanProcessor:
    """Processes hand-drawn floor plans into color-coded CAD outputs"""
    
    def __init__(self):
        # Enable performance mode for faster solving on large images
        self.engine = GeometryEngine(performance_mode=True)
        
        # Color scheme for different room types and features
        self.colors = {
            'walls': '#2C3E50',          # Dark blue-gray for walls
            'doors': '#E74C3C',          # Red for doors
            'windows': '#3498DB',        # Blue for windows
            'kitchen': '#F39C12',        # Orange for kitchen
            'bathroom': '#9B59B6',       # Purple for bathroom
            'bedroom': '#E91E63',        # Pink for bedroom
            'living_room': '#2ECC71',    # Green for living room
            'dining_room': '#F1C40F',    # Yellow for dining room
            'closet': '#95A5A6',         # Gray for closets
            'laundry': '#1ABC9C',        # Teal for laundry
            'entryway': '#34495E',       # Dark gray for entryway
            'family_room': '#27AE60',    # Darker green for family room
            'text': '#2C3E50',           # Dark for text labels
            'dimensions': '#E74C3C'      # Red for dimensions
        }
    
    def extract_lines_advanced(self, image_path):
        """Extract lines from hand-drawn sketch with enhanced preprocessing"""
        print(f"🔍 Processing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for hand-drawn sketches
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Adaptive threshold for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # 4. Edge detection with optimized parameters
        edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)
        
        # 5. Line detection with parameters tuned for architectural drawings
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,        # Lower threshold for hand-drawn lines
            minLineLength=20,    # Shorter minimum length
            maxLineGap=15        # Allow larger gaps
        )
        
        if lines is None:
            print("⚠️  No lines detected. Trying alternative parameters...")
            # Try with more relaxed parameters
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=20,
                minLineLength=15,
                maxLineGap=20
            )
        
        if lines is None:
            raise ValueError("No lines could be detected in the image")
        
        # Convert to our format
        raw_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            raw_lines.append([float(x1), float(y1), float(x2), float(y2)])
        
        print(f"✅ Extracted {len(raw_lines)} line segments")
        return raw_lines, img.shape

    def extract_lines_simple(self, image_path, max_width: int = 1200):
        """Fast, simple extractor tuned to avoid tiny scribbles and text strokes.

        Steps:
        - Resize down to max_width to cap resolution
        - Global Otsu threshold to binarize
        - Morphological opening to remove speckles and text
        - Morphological closing to connect wall strokes
        - Canny edges with auto thresholds
        - Probabilistic Hough with minLineLength relative to image size
        - Filter out very short lines and keep only top-N by length
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize for speed and to reduce tiny artifacts
        h, w = img.shape[:2]
        scale = 1.0
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize (invert so lines are white on black for morphology if needed)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove small components (text/dimensions) and thin scribbles
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)

        # Connect walls slightly
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Auto Canny thresholds based on image median
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(closed, lower, upper, apertureSize=3)

        # Hough transform with stricter minimum length
        min_len = int(max(h, w) * 0.06)  # at least 6% of the max dimension
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_len,
            maxLineGap=int(min_len * 0.3),
        )

        raw_lines = []
        if lines is not None:
            # Convert to list and sort by length desc, keep top N
            def line_len(l):
                x1, y1, x2, y2 = l[0]
                return (x2 - x1) ** 2 + (y2 - y1) ** 2

            lines = sorted(lines, key=line_len, reverse=True)
            max_keep = 4000  # cap to avoid explosion
            for line in lines[:max_keep]:
                x1, y1, x2, y2 = line[0]
                # Filter nearly-point lines just in case
                if abs(x2 - x1) + abs(y2 - y1) < min_len * 0.5:
                    continue
                raw_lines.append([float(x1), float(y1), float(x2), float(y2)])

        print(f"✅ Extracted {len(raw_lines)} line segments (simple mode)")
        return raw_lines, img.shape

    def extract_lines_two_pass(self, image_path, max_width: int = 1400):
        """Two-pass extractor that captures outer outline and medium-length internals (e.g., stairs).

        Pass A (Outer walls): stricter Hough with long min length.
        Pass B (Internals): OpenCV LSD (fast) with length/orientation filtering.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize for consistency
        h, w = img.shape[:2]
        scale = 1.0
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Light clean-up
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=1)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # Edges and Hough (outer walls)
        edges = cv2.Canny(bw, 60, 180, apertureSize=3)
        min_len_outer = int(max(h, w) * 0.12)
        hough_lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=min_len_outer, maxLineGap=int(min_len_outer * 0.3)
        )

        # LSD (internal walls + stairs)
        # Fallback-compatible LSD creation (OpenCV builds differ on kwargs)
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        except Exception:
            try:
                lsd = cv2.createLineSegmentDetector()
            except Exception:
                lsd = None
        lines_lsd = None
        if lsd is not None:
            try:
                res = lsd.detect(gray)
                # Different OpenCV versions return tuples of different sizes
                if isinstance(res, tuple):
                    lines_lsd = res[0]
                else:
                    lines_lsd = res
            except Exception:
                lines_lsd = None

        def to_seg_list(lines):
            out = []
            if lines is None:
                return out
            for l in lines:
                x1, y1, x2, y2 = l[0]
                out.append([float(x1), float(y1), float(x2), float(y2)])
            return out

        segs_outer = to_seg_list(hough_lines)

        # Filter LSD by length (medium) and remove very slanted scribbles by snapping to near-H/V/45°
        segs_inner = []
        if lines_lsd is not None:
            min_len_inner = int(max(h, w) * 0.035)
            for l in lines_lsd:
                x1, y1, x2, y2 = l[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min_len_inner:
                    continue
                # Keep most architectural angles (0/90 and 45 multiples)
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                nearest = min([0, 45, 90, 135], key=lambda a: abs(a - angle))
                if abs(angle - nearest) > 12:  # discard odd scribbles
                    continue
                segs_inner.append([float(x1), float(y1), float(x2), float(y2)])

        # Merge and dedupe similar segments
        all_segs = segs_outer + segs_inner

        def dedupe(segments, tol=6):
            kept = []
            for s in segments:
                x1, y1, x2, y2 = s
                duplicate = False
                for k in kept:
                    kx1, ky1, kx2, ky2 = k
                    if (abs(x1 - kx1) < tol and abs(y1 - ky1) < tol and abs(x2 - kx2) < tol and abs(y2 - ky2) < tol) or \
                       (abs(x1 - kx2) < tol and abs(y1 - ky2) < tol and abs(x2 - kx1) < tol and abs(y2 - ky1) < tol):
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(s)
            return kept

        raw_lines = dedupe(all_segs)
        print(f"✅ Extracted {len(raw_lines)} line segments (two-pass mode)")
        return raw_lines, img.shape

    def detect_stairs(self, segments, image_shape):
        """Heuristic stair detector: looks for 4+ short, roughly parallel, equally-spaced segments.
        Returns a list of symbol dicts with class 'stairs'.
        """
        h, w = image_shape[:2]
        if not segments:
            return []

        # Compute per-line features
        feats = []
        for x1, y1, x2, y2 in segments:
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            feats.append((angle, length, cx, cy, (x1, y1, x2, y2)))

        # Group by angle bucket (parallelism)
        buckets = {0: [], 45: [], 90: [], 135: []}
        for angle, length, cx, cy, seg in feats:
            nearest = min(buckets.keys(), key=lambda a: abs(a - angle))
            if abs(nearest - angle) <= 8:
                buckets[nearest].append((length, cx, cy, seg))

        stairs_symbols = []
        for angle_key, items in buckets.items():
            if len(items) < 4:
                continue
            # Sort by projection along normal direction (approx spacing)
            items.sort(key=lambda t: t[2] if angle_key in (0, 180) else t[1])
            # Sliding window to find sets with near-uniform spacing and similar length
            for i in range(len(items) - 3):
                window = items[i:i+6]
                if len(window) < 4:
                    continue
                lengths = [t[0] for t in window]
                if np.std(lengths) > max(6, 0.15 * np.mean(lengths)):
                    continue
                # Compute bounding box
                xs = []
                ys = []
                for _, _, _, seg in window:
                    x1, y1, x2, y2 = seg
                    xs += [x1, x2]
                    ys += [y1, y2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                # Minimal size to be plausible stairs
                if (x_max - x_min) * (y_max - y_min) < (0.01 * w * h):
                    continue
                stairs_symbols.append({'class': 'stairs', 'bbox': [x_min, y_min, x_max, y_max], 'confidence': 0.7})
                break

        return stairs_symbols
    
    def simulate_ocr_data(self, image_shape):
        """Simulate OCR data extraction from the floor plan"""
        height, width = image_shape[:2]
        
        # Simulate text recognition based on typical floor plan content
        text_data = [
            # Room labels
            {'text': 'kitchen', 'bbox': [[width*0.6, height*0.2], [width*0.8, height*0.2], [width*0.8, height*0.25], [width*0.6, height*0.25]]},
            {'text': 'family room', 'bbox': [[width*0.1, height*0.3], [width*0.3, height*0.3], [width*0.3, height*0.35], [width*0.1, height*0.35]]},
            {'text': 'dining room', 'bbox': [[width*0.6, height*0.5], [width*0.8, height*0.5], [width*0.8, height*0.55], [width*0.6, height*0.55]]},
            {'text': 'living room', 'bbox': [[width*0.4, height*0.7], [width*0.6, height*0.7], [width*0.6, height*0.75], [width*0.4, height*0.75]]},
            {'text': 'laundry', 'bbox': [[width*0.1, height*0.6], [width*0.25, height*0.6], [width*0.25, height*0.65], [width*0.1, height*0.65]]},
            {'text': 'closet', 'bbox': [[width*0.3, height*0.6], [width*0.4, height*0.6], [width*0.4, height*0.65], [width*0.3, height*0.65]]},
            {'text': 'bath', 'bbox': [[width*0.85, height*0.4], [width*0.95, height*0.4], [width*0.95, height*0.45], [width*0.85, height*0.45]]},
            {'text': 'entryway', 'bbox': [[width*0.4, height*0.85], [width*0.6, height*0.85], [width*0.6, height*0.9], [width*0.4, height*0.9]]},
            
            # Dimensions (simulate from visible text in image)
            {'text': "84''", 'bbox': [[width*0.4, height*0.95], [width*0.5, height*0.95], [width*0.5, height*0.98], [width*0.4, height*0.98]]},
            {'text': "36''", 'bbox': [[width*0.02, height*0.5], [width*0.08, height*0.5], [width*0.08, height*0.53], [width*0.02, height*0.53]]},
            {'text': "124''", 'bbox': [[width*0.5, height*0.02], [width*0.6, height*0.02], [width*0.6, height*0.05], [width*0.5, height*0.05]]},
        ]
        
        return text_data
    
    def simulate_symbol_detection(self, image_shape):
        """Simulate door and window detection"""
        height, width = image_shape[:2]
        
        symbols = [
            # Doors
            {'class': 'door', 'bbox': [width*0.45, height*0.82, width*0.55, height*0.88], 'confidence': 0.9},
            {'class': 'door', 'bbox': [width*0.25, height*0.45, width*0.35, height*0.55], 'confidence': 0.85},
            {'class': 'door', 'bbox': [width*0.8, height*0.35, width*0.85, height*0.45], 'confidence': 0.8},
            
            # Windows
            {'class': 'window', 'bbox': [width*0.1, height*0.15, width*0.2, height*0.25], 'confidence': 0.9},
            {'class': 'window', 'bbox': [width*0.7, height*0.1, width*0.9, height*0.15], 'confidence': 0.85},
            {'class': 'window', 'bbox': [width*0.9, height*0.6, width*0.95, height*0.8], 'confidence': 0.8},
        ]
        
        return symbols
    
    def classify_rooms_by_text(self, rooms, text_data):
        """Classify rooms based on nearby text labels"""
        classified_rooms = []
        
        for room in rooms:
            room_type = 'unknown'
            
            # Find text labels within or near this room
            for text_item in text_data:
                text_center = self.bbox_center(text_item['bbox'])
                
                # Check if text is inside the room
                if self.point_in_polygon(text_center, room.vertices):
                    text_lower = text_item['text'].lower()
                    
                    # Map text to room types
                    if 'kitchen' in text_lower:
                        room_type = 'kitchen'
                    elif 'bath' in text_lower or 'bathroom' in text_lower:
                        room_type = 'bathroom'
                    elif 'bedroom' in text_lower or 'bed' in text_lower:
                        room_type = 'bedroom'
                    elif 'living' in text_lower:
                        room_type = 'living_room'
                    elif 'dining' in text_lower:
                        room_type = 'dining_room'
                    elif 'family' in text_lower:
                        room_type = 'family_room'
                    elif 'closet' in text_lower:
                        room_type = 'closet'
                    elif 'laundry' in text_lower:
                        room_type = 'laundry'
                    elif 'entry' in text_lower:
                        room_type = 'entryway'
                    break
            
            # Fallback classification based on area if no text found
            if room_type == 'unknown':
                if room.area < 500:
                    room_type = 'closet'
                elif room.area < 2000:
                    room_type = 'bathroom'
                elif room.area < 8000:
                    room_type = 'bedroom'
                else:
                    room_type = 'living_room'
            
            room.room_type = room_type
            classified_rooms.append(room)
        
        return classified_rooms
    
    def bbox_center(self, bbox):
        """Calculate center of bounding box"""
        if isinstance(bbox[0], list):
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            return Point(sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        else:
            return Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def point_in_polygon(self, point, vertices):
        """Check if point is inside polygon"""
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
    
    def create_colored_visualization(self, processed_result, symbols, text_data, output_path, original_shape):
        """Create color-coded floor plan visualization"""
        print("🎨 Creating color-coded visualization...")
        
        height, width = original_shape[:2]
        
        # Create figure with high DPI for crisp output
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y axis to match image coordinates
        ax.set_aspect('equal')
        
        # Fill background
        ax.add_patch(patches.Rectangle((0, 0), width, height, 
                                     facecolor='white', edgecolor='none'))
        
        # Draw rooms with different colors
        rooms = processed_result.get('rooms', [])
        if text_data:
            rooms = self.classify_rooms_by_text(rooms, text_data)
        
        print(f"🏠 Drawing {len(rooms)} rooms with color coding...")
        
        for i, room in enumerate(rooms):
            if len(room.vertices) < 3:
                continue
            
            # Get room color
            room_color = self.colors.get(room.room_type, self.colors['living_room'])
            
            # Create polygon
            xy_points = [(v.x, v.y) for v in room.vertices]
            
            # Draw filled room
            room_patch = patches.Polygon(
                xy_points, 
                facecolor=room_color, 
                alpha=0.3,
                edgecolor=room_color,
                linewidth=2
            )
            ax.add_patch(room_patch)
            
            # Add room label
            if hasattr(room, 'room_type'):
                ax.text(room.centroid.x, room.centroid.y, 
                       room.room_type.replace('_', ' ').title(),
                       ha='center', va='center', 
                       fontsize=10, fontweight='bold',
                       color=self.colors['text'],
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor='white', alpha=0.8))
        
        # Draw walls (main structure lines)
        lines = processed_result.get('lines', [])
        print(f"🏗️  Drawing {len(lines)} wall segments...")
        
        for line in lines:
            ax.plot([line.start.x, line.end.x], 
                   [line.start.y, line.end.y],
                   color=self.colors['walls'], 
                   linewidth=3, 
                   solid_capstyle='round')
        
        # Draw doors and windows (and label-only for other symbols like stairs)
        for symbol in symbols:
            bbox = symbol['bbox']
            symbol_type = symbol['class']
            
            if symbol_type == 'door':
                color = self.colors['doors']
                label = 'Door'
            elif symbol_type == 'window':
                color = self.colors['windows']
                label = 'Window'
            else:
                # For non-door/window (e.g., stairs), draw a thin dashed outline rectangle for clarity
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    facecolor='none', edgecolor=self.colors['walls'], linewidth=1.5, linestyle='--', alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(center_x, center_y, symbol_type,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color=self.colors['text'],
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6))
                continue

            # Draw symbol rectangle for doors/windows
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                facecolor=color, 
                alpha=0.7,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Add label text on top of the rectangle
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            ax.text(center_x, center_y, label,
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   color='white')
        
        # Draw dimensions and labels from OCR
        for text_item in text_data:
            bbox = text_item['bbox']
            text = text_item['text']
            
            center = self.bbox_center(bbox)
            
            # Determine if it's a dimension or label
            if any(char in text for char in ["'", '"', 'ft', 'in']) or text.replace('.', '').isdigit():
                # It's a dimension
                ax.text(center.x, center.y, text,
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color=self.colors['dimensions'],
                       bbox=dict(boxstyle="round,pad=0.2", 
                               facecolor='yellow', alpha=0.8))
            else:
                # It's a room label (already handled above in room drawing)
                pass
        
        # Add title and statistics
        stats = processed_result.get('statistics', {})
        title = f"PaperCAD Edge - Color-Coded Floor Plan\n"
        title += f"Rooms: {stats.get('rooms_detected', 0)} | "
        title += f"Lines: {stats.get('final_lines', 0)} | "
        title += f"Constraints: {stats.get('constraints_applied', 0)}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = []
        room_types_found = set(getattr(room, 'room_type', 'unknown') for room in rooms)
        
        for room_type in sorted(room_types_found):
            if room_type in self.colors:
                legend_elements.append(
                    patches.Patch(color=self.colors[room_type], 
                                label=room_type.replace('_', ' ').title())
                )
        
        # Add door and window to legend
        legend_elements.extend([
            patches.Patch(color=self.colors['doors'], label='Doors'),
            patches.Patch(color=self.colors['windows'], label='Windows'),
            patches.Patch(color=self.colors['walls'], label='Walls')
        ])
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Remove axes for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Color-coded floor plan saved to: {output_path}")
    
    def process_floor_plan(self, input_path, output_path):
        """Main processing function"""
        print("🚀 Starting PaperCAD Edge Color Processing...")
        
        # Step 1: Extract lines from image (two-pass to capture outline + stairs)
        raw_lines, image_shape = self.extract_lines_two_pass(input_path)
        
        # Step 2: Simulate OCR and symbol detection
        text_data = self.simulate_ocr_data(image_shape)
        symbols = self.simulate_symbol_detection(image_shape)
        # Augment with heuristic stairs detection from segments
        stairs_syms = self.detect_stairs(raw_lines, image_shape)
        symbols.extend(stairs_syms)
        
        # Step 3: Process through geometry engine
        print("⚙️  Processing through geometry engine...")
        result = self.engine.process_raw_geometry(raw_lines, symbols, text_data)
        
        # Step 4: Create colored visualization
        self.create_colored_visualization(result, symbols, text_data, output_path, image_shape)
        
        # Print summary
        print("\n📊 Processing Summary:")
        print(f"  Input lines detected: {len(raw_lines)}")
        print(f"  Rooms identified: {len(result.get('rooms', []))}")
        print(f"  Doors/Windows: {len(symbols)}")
        print(f"  Text labels: {len(text_data)}")
        print(f"  Constraints applied: {result.get('statistics', {}).get('constraints_applied', 0)}")
        
        return result

def main():
    processor = ColoredFloorPlanProcessor()
    
    input_image = "/Users/nishanthkotla/Desktop/papercad/data/image_samples/test.jpeg"
    output_image = "/Users/nishanthkotla/Desktop/papercad/output/image_tests/test_colored_floorplan.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    
    try:
        result = processor.process_floor_plan(input_image, output_image)
        print(f"\n🎉 SUCCESS! Color-coded floor plan created at: {output_image}")
        
    except Exception as e:
        print(f"❌ Error processing floor plan: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
