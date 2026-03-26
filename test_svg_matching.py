import numpy as np
from svgpathtools import svg2paths
from shapely.geometry import Polygon
import shapely.affinity as affinity
import sys
from pathlib import Path

script_dir = Path(__file__).parent
sys.path.append(str(script_dir))
from scripts.generate_nested_layouts import load_shelves, Sheet, export_sheet_to_dxf

def path_to_poly(path, num_samples=150):
    if not path.isclosed():
        try: path.append(path[-1].reversed())
        except: pass
    points = []
    for i in range(num_samples):
        try:
            pt = path.point(i / float(num_samples - 1))
            points.append((pt.real, pt.imag))
        except: pass
    if len(points) < 3: return None
    poly = Polygon(points)
    if not poly.is_valid: poly = poly.buffer(0)
    return poly

def normalize_poly(poly):
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx+maxx)/2, (miny+maxy)/2
    p = affinity.translate(poly, -cx, -cy)
    scale = 1.0 / np.sqrt(p.area)
    p = affinity.scale(p, xfact=scale, yfact=scale, origin=(0,0))
    return p, cx, cy, scale

def get_best_match(svg_poly, dxf_shelves):
    svg_norm, scx, scy, s_scale = normalize_poly(svg_poly)
    
    # SVG has Y axis going DOWN. DXF has Y going UP.
    # We must mirror the SVG to match DXF standard orientation
    svg_norm = affinity.scale(svg_norm, xfact=1, yfact=-1, origin=(0,0))
    
    best_score = float('inf')
    best_shelf = None
    best_angle = 0
    best_mirrored = False
    
    # Standard rotations since manual packing likely uses 90 degree increments
    # Might also use arbitrary, but let's test 360 degrees in small increments if we want,
    # or just rely on bounding box aspect ratio to narrow it down!
    rotations = [0, 90, 180, 270]
    
    for shelf in dxf_shelves:
        dxf_norm, dcx, dcy, d_scale = normalize_poly(shelf.poly)
        
        for angle in rotations:
            for mirror in [False, True]:
                test_p = affinity.rotate(dxf_norm, angle, origin=(0,0))
                if mirror:
                    test_p = affinity.scale(test_p, xfact=-1, yfact=1, origin=(0,0))
                
                score = svg_norm.symmetric_difference(test_p).area
                if score < best_score:
                    best_score = score
                    best_shelf = shelf
                    best_angle = angle
                    best_mirrored = mirror
                    
    return best_shelf, best_score, best_angle, best_mirrored, scx, scy, s_scale

def parse_and_match(filename, dxf_shelves):
    print(f"\nMatching {filename}...")
    paths, _ = svg2paths(filename)
    svg_polys = []
    for path in paths:
        try:
            p = path_to_poly(path)
            if p and p.area > 500: # filter noise
                svg_polys.append(p)
        except: pass
        
    print(f"Extracted {len(svg_polys)} valid shelves from SVG.")
    matches = []
    available_shelves = list(dxf_shelves)
    
    for sp in svg_polys:
        shelf, score, angle, mirror, cx, cy, s_scale = get_best_match(sp, available_shelves)
        if shelf:
            print(f"  Matched to {shelf.name} (Score: {score:.3f}, Angle: {angle}, Mirror: {mirror})")
            matches.append({
                'svg_poly': sp,
                'shelf': shelf,
                'cx': cx,
                'cy': cy,
                'scale': 1.0 / s_scale,
                'angle': angle,
                'mirror': mirror
            })
            available_shelves.remove(shelf)
    return matches

if __name__ == '__main__':
    dxf_shelves = load_shelves()
    matches_1 = parse_and_match('output/sheet_1.svg', dxf_shelves)
    matches_2 = parse_and_match('output/sheet_2.svg', [s for s in dxf_shelves if s not in [m['shelf'] for m in matches_1]])
