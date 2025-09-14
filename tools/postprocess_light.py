#!/usr/bin/env python3
import sys, os, math
import numpy as np
import cv2

sys.path.append('/Users/nishanthkotla/Desktop/papercad')
from geometry_engine.primitives import Point, LineSegment
from tools.create_colored_floorplan import ColoredFloorPlanProcessor


def angle(seg):
    x1,y1,x2,y2 = seg
    return math.degrees(math.atan2(y2-y1, x2-x1)) % 180

def length(seg):
    x1,y1,x2,y2 = seg
    return math.hypot(x2-x1, y2-y1)

def perp_dist(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    A=ay2-ay1; B=ax1-ax2; C=-(A*ax1+B*ay1)
    denom=max(1e-6, math.hypot(A,B))
    mx,my=(bx1+bx2)/2.0,(by1+by2)/2.0
    return abs(A*mx+B*my+C)/denom

def overlap_ratio(a,b):
    ang = math.radians(min([0,90], key=lambda k: abs(k-angle(a))))
    ux,uy=math.cos(ang), math.sin(ang)
    def proj(s):
        x1,y1,x2,y2=s
        t1=x1*ux+y1*uy; t2=x2*ux+y2*uy
        return (min(t1,t2), max(t1,t2))
    a1,a2=proj(a); b1,b2=proj(b)
    inter=max(0.0, min(a2,b2)-max(a1,b1))
    shorter=max(1e-6, min(a2-a1,b2-b1))
    return inter/shorter

def remove_parallel_duplicates(segments, angle_tol=2.0, offset_tol=4.0, overlap_thresh=0.6):
    keep=[True]*len(segments)
    for i in range(len(segments)):
        if not keep[i]:
            continue
        for j in range(i+1, len(segments)):
            if not keep[j]:
                continue
            a,b=segments[i], segments[j]
            if min(abs(angle(a)-angle(b)), 180-abs(angle(a)-angle(b)))>angle_tol:
                continue
            if perp_dist(a,b)>offset_tol:
                continue
            if overlap_ratio(a,b)<overlap_thresh:
                continue
            if length(a)>=length(b):
                keep[j]=False
            else:
                keep[i]=False
                break
    return [s for s,k in zip(segments, keep) if k]


def light_postprocess(input_path, output_path):
    proc=ColoredFloorPlanProcessor()
    raw, shape = proc.extract_lines_two_pass(input_path)
    # Only remove tiny and duplicates
    min_len=max(10, int(0.01*max(shape[0], shape[1])))
    raw=[s for s in raw if length(s)>=min_len]
    raw=remove_parallel_duplicates(raw)
    # To LineSegment
    lines=[LineSegment(Point(s[0],s[1]), Point(s[2],s[3])) for s in raw]
    # Symbols and text
    text=proc.simulate_ocr_data(shape)
    symbols=proc.simulate_symbol_detection(shape)
    # Clean inside door/window boxes and add canonical center line
    margin=max(6, int(0.012*max(shape[0], shape[1])))
    filtered=[]
    for ln in lines:
        mx=(ln.start.x+ln.end.x)/2.0; my=(ln.start.y+ln.end.y)/2.0
        keep=True
        for sym in symbols:
            if sym.get('class') not in ('door','window'): continue
            bx0,by0,bx1,by1=sym['bbox']
            bx0-=margin; by0-=margin; bx1+=margin; by1+=margin
            if bx0<=mx<=bx1 and by0<=my<=by1:
                keep=False; break
        if keep: filtered.append(ln)

    canonical=[]
    for sym in symbols:
        if sym.get('class') not in ('door','window'): continue
        bx0,by0,bx1,by1=sym['bbox']
        cx,cy=(bx0+bx1)/2.0,(by0+by1)/2.0
        wlen,hlen=(bx1-bx0),(by1-by0)
        if sym['class']=='window':
            if wlen>=hlen:
                canonical.append(LineSegment(Point(bx0+0.15*wlen, cy), Point(bx1-0.15*wlen, cy)))
            else:
                canonical.append(LineSegment(Point(cx, by0+0.15*hlen), Point(cx, by1-0.15*hlen)))
        else:
            if wlen<hlen:
                canonical.append(LineSegment(Point(bx0+0.25*wlen, cy), Point(bx1-0.25*wlen, cy)))
            else:
                canonical.append(LineSegment(Point(cx, by0+0.25*hlen), Point(cx, by1-0.25*hlen)))

    result={
        'lines': filtered+canonical,
        'rooms': [],
        'constraints': {},
        'statistics': {'final_lines': len(filtered)+len(canonical), 'rooms_detected':0, 'constraints_applied':0}
    }
    proc.create_colored_visualization(result, symbols, text, output_path, shape)


def main():
    if len(sys.argv)>=3:
        inp=sys.argv[1]; outp=sys.argv[2]
    else:
        inp='/Users/nishanthkotla/Desktop/papercad/data/image_samples/test_4.jpeg'
        outp='/Users/nishanthkotla/Desktop/papercad/output/image_tests/test_4_colored_light.png'
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    light_postprocess(inp, outp)
    print('✅ Light smoothed saved to:', outp)

if __name__=='__main__':
    sys.exit(main())


