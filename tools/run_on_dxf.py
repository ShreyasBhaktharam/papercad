import os
from typing import List
import ezdxf

from geometry_engine.api import GeometryEngine
from geometry_engine.primitives import Point, LineSegment


def dxf_to_raw_lines(path: str) -> List[List[float]]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    lines: List[List[float]] = []
    
    for e in msp.query('LINE'):
        lines.append([float(e.dxf.start.x), float(e.dxf.start.y), float(e.dxf.end.x), float(e.dxf.end.y)])
    
    # Convert LWPOLYLINE and POLYLINE segments to lines
    for pl in msp.query('LWPOLYLINE'):  # lightweight polyline
        pts = [tuple(p[:2]) for p in pl.get_points()]  # x, y
        if pl.closed and pts:
            pts.append(pts[0])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            lines.append([float(x1), float(y1), float(x2), float(y2)])
    for pl in msp.query('POLYLINE'):
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in pl.vertices]
        if pl.is_closed and pts:
            pts.append(pts[0])
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            lines.append([float(x1), float(y1), float(x2), float(y2)])
    return lines


def run_on_file(path: str):
    print(f"Processing DXF: {path}")
    lines = dxf_to_raw_lines(path)
    print(f"  Extracted {len(lines)} line segments")
    
    engine = GeometryEngine()
    result = engine.process_raw_geometry(lines)
    
    print(f"  Output lines: {len(result['lines'])}")
    print(f"  Constraints applied: {result['statistics']['constraints_applied']}")


if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'samples')
    for fname in os.listdir(base):
        if fname.lower().endswith('.dxf'):
            run_on_file(os.path.join(base, fname))


