#!/usr/bin/env python3
"""
Post-process and smooth extracted floor-plan geometry for professional rendering.

Pipeline:
1) Extract raw segments using the two-pass extractor from create_colored_floorplan
2) Snap angles to {0, 45, 90, 135} deg
3) Bridge small endpoint gaps and deduplicate
4) Merge collinear segments
5) Remove tiny fragments
6) Render clean, color-coded output (reusing the visualizer)
"""

import sys
import os
import math
import numpy as np
import cv2

sys.path.append('/Users/nishanthkotla/Desktop/papercad')

from geometry_engine.primitives import Point, LineSegment
from tools.create_colored_floorplan import ColoredFloorPlanProcessor


def angle_of_segment(seg):
    x1, y1, x2, y2 = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def length_of_segment(seg):
    x1, y1, x2, y2 = seg
    return math.hypot(x2 - x1, y2 - y1)


def snap_angle(seg, allowed=(0, 90)):
    """Snap the segment direction to the nearest allowed angle by rotating about midpoint."""
    x1, y1, x2, y2 = seg
    ang = angle_of_segment(seg)
    target = min(allowed, key=lambda a: abs(a - ang))
    # compute length and midpoint
    L = length_of_segment(seg)
    if L <= 1e-3:
        return seg
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    theta = math.radians(target)
    dx = (L / 2.0) * math.cos(theta)
    dy = (L / 2.0) * math.sin(theta)
    # New endpoints, keep original orientation sign by quadrant matching
    # Use sign from original vector projected on target axis
    vx, vy = x2 - x1, y2 - y1
    if vx * math.cos(theta) + vy * math.sin(theta) < 0:
        dx, dy = -dx, -dy
    return [mx - dx, my - dy, mx + dx, my + dy]


def connect_endpoints(segments, join_tol=8):
    """Snap endpoints within join_tol pixels to their average location."""
    if not segments:
        return []
    pts = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        pts.append((i, 0, x1, y1))
        pts.append((i, 1, x2, y2))

    # Union-Find for clusters
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(pts)):
        _, _, xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            _, _, xj, yj = pts[j]
            if abs(xi - xj) <= join_tol and abs(yi - yj) <= join_tol:
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= join_tol ** 2:
                    union(i, j)

    # Compute cluster averages
    clusters = {}
    for idx, (_, _, x, y) in enumerate(pts):
        r = find(idx)
        clusters.setdefault(r, []).append((x, y))
    average = {r: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for r, v in clusters.items()}

    # Apply snapping
    snapped = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in segments]
    for idx, (seg_idx, end_id, _, _) in enumerate(pts):
        r = find(idx)
        sx, sy = average[r]
        if end_id == 0:
            snapped[seg_idx][0] = sx
            snapped[seg_idx][1] = sy
        else:
            snapped[seg_idx][2] = sx
            snapped[seg_idx][3] = sy
    return snapped


def are_collinear(a, b, angle_tol=4, dist_tol=6):
    """Check if two segments are roughly collinear and close to each other."""
    ang_a = angle_of_segment(a)
    ang_b = angle_of_segment(b)
    if min(abs(ang_a - ang_b), 180 - abs(ang_a - ang_b)) > angle_tol:
        return False
    # distance between infinite lines (midpoints projection)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # line a in ax + by + c = 0
    A = ay2 - ay1
    B = ax1 - ax2
    C = -(A * ax1 + B * ay1)
    def point_line_dist(x, y):
        return abs(A * x + B * y + C) / max(1e-6, math.hypot(A, B))
    mbx, mby = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return point_line_dist(mbx, mby) <= dist_tol


def merge_collinear_endpoint_connected(segments, edges=None, angle_tol=3, dist_tol=4, support_tau=0.35):
    """Merge only endpoint-connected, nearly collinear segments. Avoids spanning across corners.

    If edges is provided (binary), validate that the merged line is supported by image edges.
    """
    if not segments:
        return []

    # Build adjacency by shared endpoints (exact after snapping)
    def key(p, tol=1e-3):
        return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

    endpoints_map = {}
    for idx, (x1, y1, x2, y2) in enumerate(segments):
        endpoints_map.setdefault(key((x1, y1)), []).append((idx, 0))
        endpoints_map.setdefault(key((x2, y2)), []).append((idx, 1))

    used = [False] * len(segments)
    merged = []

    def edge_support(seg):
        if edges is None:
            return 1.0
        h, w = edges.shape[:2]
        x1, y1, x2, y2 = map(int, seg)
        samples = max(10, int(length_of_segment(seg) / 6))
        ok = 0
        for t in np.linspace(0, 1, samples):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            x0, x1w = max(0, x - 2), min(w, x + 3)
            y0, y1w = max(0, y - 2), min(h, y + 3)
            if np.any(edges[y0:y1w, x0:x1w] > 0):
                ok += 1
        return ok / float(samples)

    for i, seg in enumerate(segments):
        if used[i]:
            continue
        x1, y1, x2, y2 = seg
        chain = [seg]
        used[i] = True
        extended = True
        while extended:
            extended = False
            # try to extend at both ends
            for endpt in [(x1, y1), (x2, y2)]:
                k = key(endpt)
                for (j, which) in endpoints_map.get(k, []):
                    if used[j] or j == i:
                        continue
                    s2 = segments[j]
                    # Check collinearity with last segment touching this endpoint
                    base = chain[0] if (endpt == (x1, y1)) else chain[-1]
                    if not are_collinear(base, s2, angle_tol, dist_tol):
                        continue
                    # Candidate merged segment from farthest endpoints
                    candidates = [chain[0][0:2], chain[0][2:4], chain[-1][0:2], chain[-1][2:4], s2[0:2], s2[2:4]]
                    xs = [p[0] for p in candidates]
                    ys = [p[1] for p in candidates]
                    # choose extreme pair along the angle of base
                    ang = math.radians(min([0, 90], key=lambda k2: abs(k2 - angle_of_segment(base))))
                    ux, uy = math.cos(ang), math.sin(ang)
                    ts = [(p[0] * ux + p[1] * uy, p) for p in candidates]
                    ts.sort(key=lambda t: t[0])
                    merged_seg = [ts[0][1][0], ts[0][1][1], ts[-1][1][0], ts[-1][1][1]]
                    if edge_support(merged_seg) < support_tau:
                        continue
                    # accept merge
                    chain.append(s2)
                    used[j] = True
                    x1, y1, x2, y2 = merged_seg
                    extended = True
                    break
                if extended:
                    break
        merged.append([x1, y1, x2, y2])
    return merged


def dedupe(segments, tol=3):
    out = []
    for s in segments:
        x1, y1, x2, y2 = s
        dup = False
        for t in out:
            tx1, ty1, tx2, ty2 = t
            same_dir = (abs(x1 - tx1) < tol and abs(y1 - ty1) < tol and abs(x2 - tx2) < tol and abs(y2 - ty2) < tol)
            opp_dir = (abs(x1 - tx2) < tol and abs(y1 - ty2) < tol and abs(x2 - tx1) < tol and abs(y2 - ty1) < tol)
            if same_dir or opp_dir:
                dup = True
                break
        if not dup:
            out.append(s)
    return out


def remove_tiny(segments, min_len=14):
    return [s for s in segments if length_of_segment(s) >= min_len]


def _perp_distance_between_lines(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    A = ay2 - ay1
    B = ax1 - ax2
    C = -(A * ax1 + B * ay1)
    def dpt(x, y):
        return abs(A * x + B * y + C) / max(1e-6, math.hypot(A, B))
    mbx, mby = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return dpt(mbx, mby)


def _overlap_ratio(a, b):
    """Return projected overlap ratio along principal axis (0..1)."""
    ang = math.radians(min([0, 90], key=lambda k: abs(k - angle_of_segment(a))))
    ux, uy = math.cos(ang), math.sin(ang)
    def proj(seg):
        x1, y1, x2, y2 = seg
        t1 = x1 * ux + y1 * uy
        t2 = x2 * ux + y2 * uy
        return (min(t1, t2), max(t1, t2))
    a1, a2 = proj(a)
    b1, b2 = proj(b)
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = max(1e-6, min(a2 - a1, b2 - b1))
    return inter / shorter


def remove_parallel_duplicates(segments, angle_tol=2.0, offset_tol=4.0, overlap_thresh=0.6):
    """Collapse double/triple strokes: keep the longest when near-parallel and co-located."""
    keep = [True] * len(segments)
    for i in range(len(segments)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(segments)):
            if not keep[j]:
                continue
            a, b = segments[i], segments[j]
            if min(abs(angle_of_segment(a) - angle_of_segment(b)), 180 - abs(angle_of_segment(a) - angle_of_segment(b))) > angle_tol:
                continue
            if _perp_distance_between_lines(a, b) > offset_tol:
                continue
            if _overlap_ratio(a, b) < overlap_thresh:
                continue
            # drop shorter
            if length_of_segment(a) >= length_of_segment(b):
                keep[j] = False
            else:
                keep[i] = False
                break
    return [s for s, k in zip(segments, keep) if k]


def smooth_segments(raw_segments, canvas_shape, reference_image_path=None):
    """Run the full smoothing pipeline on raw float segments list [[x1,y1,x2,y2],...]."""
    if not raw_segments:
        return []
    h, w = canvas_shape[:2]
    # Determine allowed angles: if many diagonals present, include 45/135 else stick to H/V
    angles = [angle_of_segment(s) for s in raw_segments]
    diag_ratio = sum(1 for a in angles if min(abs(a - 45), abs(a - 135)) < 10) / max(1, len(angles))
    allowed = (0, 90) if diag_ratio < 0.15 else (0, 45, 90, 135)

    # 1) angle snap
    snapped = [snap_angle(s, allowed=allowed) for s in raw_segments]
    # 2) connect endpoints
    joined = connect_endpoints(snapped, join_tol=max(6, int(0.006 * max(w, h))))
    # 2.5) remove parallel duplicates (double/triple strokes)
    joined = remove_parallel_duplicates(joined)

    # 3) merge collinear but only if endpoint-connected and supported by edges
    edges = None
    if reference_image_path is not None:
        img = cv2.imread(reference_image_path)
        if img is not None:
            # resize to canvas_shape to align sampling
            h, w = canvas_shape[:2]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            v = np.median(gray)
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))
            edges = cv2.Canny(gray, lower, upper, apertureSize=3)

    merged = merge_collinear_endpoint_connected(joined, edges=edges)
    # 4) dedupe and remove tiny
    merged = dedupe(merged)
    merged = remove_tiny(merged, min_len=max(10, int(0.01 * max(w, h))))
    return merged


def render_smoothed(input_image, output_image):
    processor = ColoredFloorPlanProcessor()
    # Reuse extractor (two-pass) to get primitives
    raw_lines, image_shape = processor.extract_lines_two_pass(input_image)
    # Smooth
    smoothed = smooth_segments(raw_lines, image_shape, reference_image_path=input_image)
    # Convert to LineSegment objects for visualizer
    smoothed_lines = [LineSegment(Point(s[0], s[1]), Point(s[2], s[3])) for s in smoothed]

    # Fake minimal result dict expected by create_colored_visualization
    result = {
        'lines': smoothed_lines,
        'rooms': [],
        'constraints': {},
        'statistics': {
            'final_lines': len(smoothed_lines),
            'rooms_detected': 0,
            'constraints_applied': 0
        }
    }

    # Symbols: reuse simulated ones to mark windows/doors mildly
    text_data = processor.simulate_ocr_data(image_shape)
    symbols = processor.simulate_symbol_detection(image_shape)

    # Symbol-aware cleanup: collapse multi-strokes inside door/window boxes and add a single canonical line
    def symbol_cleanup(line_objs, symbols, shape):
        h, w = shape[:2]
        margin = max(6, int(0.012 * max(w, h)))
        filtered = []
        for ln in line_objs:
            mx = (ln.start.x + ln.end.x) / 2.0
            my = (ln.start.y + ln.end.y) / 2.0
            keep = True
            for sym in symbols:
                if sym.get('class') not in ('door', 'window'):
                    continue
                bx0, by0, bx1, by1 = sym['bbox']
                bx0 -= margin; by0 -= margin; bx1 += margin; by1 += margin
                if bx0 <= mx <= bx1 and by0 <= my <= by1:
                    keep = False
                    break
            if keep:
                filtered.append(ln)

        # Add canonical center line per symbol
        canonical = []
        for sym in symbols:
            if sym.get('class') not in ('door', 'window'):
                continue
            bx0, by0, bx1, by1 = sym['bbox']
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            wlen, hlen = (bx1 - bx0), (by1 - by0)
            if sym['class'] == 'window':
                if wlen >= hlen:
                    canonical.append(LineSegment(Point(bx0 + 0.15 * wlen, cy), Point(bx1 - 0.15 * wlen, cy)))
                else:
                    canonical.append(LineSegment(Point(cx, by0 + 0.15 * hlen), Point(cx, by1 - 0.15 * hlen)))
            else:  # door
                if wlen < hlen:
                    canonical.append(LineSegment(Point(bx0 + 0.25 * wlen, cy), Point(bx1 - 0.25 * wlen, cy)))
                else:
                    canonical.append(LineSegment(Point(cx, by0 + 0.25 * hlen), Point(cx, by1 - 0.25 * hlen)))

        return filtered + canonical

    smoothed_lines = symbol_cleanup(smoothed_lines, symbols, image_shape)
    result['lines'] = smoothed_lines
    result['statistics']['final_lines'] = len(smoothed_lines)

    # Render
    processor.create_colored_visualization(result, symbols, text_data, output_image, image_shape)
    return result


def main():
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        input_path = "/Users/nishanthkotla/Desktop/papercad/data/image_samples/test_3.png"
        output_path = "/Users/nishanthkotla/Desktop/papercad/output/image_tests/test_3_colored_smoothed.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("🧽 Smoothing and post-processing...")
    res = render_smoothed(input_path, output_path)
    print(f"✅ Smoothed plan saved to: {output_path}")
    print(f"  Final lines: {res['statistics']['final_lines']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


