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
    return p, cx, cy

def get_best_match(svg_poly, dxf_shelves):
    svg_norm, scx, scy = normalize_poly(svg_poly)
    svg_norm = affinity.scale(svg_norm, xfact=1, yfact=-1, origin=(0,0))
    best_score = float('inf')
    best_shelf = None
    best_angle = 0
    best_mirrored = False
    rotations = [0, 90, 180, 270]
    for shelf in dxf_shelves:
        dxf_norm, dcx, dcy = normalize_poly(shelf.poly)
        for angle in rotations:
            for mirror in [False, True]:
                test_p = affinity.rotate(dxf_norm, angle, origin=(0,0))
                if mirror: test_p = affinity.scale(test_p, xfact=-1, yfact=1, origin=(0,0))
                score = svg_norm.symmetric_difference(test_p).area
                if score < best_score:
                    best_score = score
                    best_shelf = shelf
                    best_angle = angle
                    best_mirrored = mirror
    return best_shelf, best_score, best_angle, best_mirrored, scx, scy

def get_global_placed_shelves(available_shelves):
    all_svg_polys = []
    
    # We load BOTH svgs to reconstitute the entire CAD canvas!
    for f in ['output/sheet_1.svg', 'output/sheet_2.svg']:
        paths, _ = svg2paths(f)
        for path in paths:
            try:
                p = path_to_poly(path)
                if p and p.area > 500: all_svg_polys.append(p)
            except: pass
            
    # There are two 48x96 ply sheets drawn. We'll find them (area ~65800)
    ply_sheets = [p for p in all_svg_polys if 60000 < p.area < 70000]
    for p in ply_sheets:
        all_svg_polys.remove(p)
        
    s_minx, s_miny, s_maxx, s_maxy = ply_sheets[0].bounds
    scale_x = 96.0 / (s_maxx - s_minx)
    scale_y = 48.0 / (s_maxy - s_miny)
    
    placed = []
    for sp in all_svg_polys:
        if not available_shelves: break
        shelf, score, angle, mirror, scx, scy = get_best_match(sp, available_shelves)
        if shelf and score < 0.8: # Must be a strong match (filters out garbage polys)
            # Use absolute scaling starting from 0, since the user's canvas implies clusters!
            dxf_cx = scx * scale_x
            dxf_cy = -(scy * scale_y) # SVG Y goes down
            
            minx, miny, maxx, maxy = shelf.bounds()
            dcx, dcy = (minx+maxx)/2, (miny+maxy)/2
            shelf.translate(-dcx, -dcy)
            shelf.rotate(angle)
            if mirror:
                shelf.poly = affinity.scale(shelf.poly, xfact=-1, yfact=1, origin=(0,0))
                shelf.bracket_polys = [affinity.scale(b, xfact=-1, yfact=1, origin=(0,0)) for b in shelf.bracket_polys]
                shelf.bracket_cuts = [affinity.scale(c, xfact=-1, yfact=1, origin=(0,0)) for c in shelf.bracket_cuts]
                
            shelf.translate(dxf_cx, dxf_cy)
            placed.append(shelf)
            available_shelves.remove(shelf)
            
    return placed

def relax_layout(shelves, max_iters=250):
    MARGIN = 0.5
    for it in range(max_iters):
        max_overlap = 0
        forces = {s: np.array([0.0, 0.0]) for s in shelves}
        for i, s1 in enumerate(shelves):
            for j, s2 in enumerate(shelves):
                if i >= j: continue
                d = s1.distance(s2)
                if d < MARGIN:
                    overlap = MARGIN - d
                    max_overlap = max(max_overlap, overlap)
                    c1x, c1y = (s1.bounds()[0]+s1.bounds()[2])/2, (s1.bounds()[1]+s1.bounds()[3])/2
                    c2x, c2y = (s2.bounds()[0]+s2.bounds()[2])/2, (s2.bounds()[1]+s2.bounds()[3])/2
                    dx, dy = c1x - c2x, c1y - c2y
                    dist = np.hypot(dx, dy)
                    if dist < 1e-5: dx, dy, dist = 1.0, 0.0, 1.0
                    push = (overlap / 2.0) * 1.5 
                    forces[s1] += np.array([(dx/dist)*push, (dy/dist)*push])
                    forces[s2] -= np.array([(dx/dist)*push, (dy/dist)*push])
                    
            minx, miny, maxx, maxy = s1.bounds()
            if minx < 0: 
                forces[s1][0] -= minx; max_overlap = max(max_overlap, -minx)
            if miny < 0: 
                forces[s1][1] -= miny; max_overlap = max(max_overlap, -miny)
            if maxx > 96: 
                forces[s1][0] -= (maxx - 96); max_overlap = max(max_overlap, maxx - 96)
            if maxy > 48: 
                forces[s1][1] -= (maxy - 48); max_overlap = max(max_overlap, maxy - 48)
            
        if max_overlap < 0.01:
            print(f"Converged beautifully after {it} iterations.")
            break
        for s in shelves:
            if np.linalg.norm(forces[s]) > 0:
                s.translate(forces[s][0], forces[s][1])

def main():
    dxf_shelves = load_shelves()
    available = list(dxf_shelves)
    
    global_placed = get_global_placed_shelves(available)
    print(f"Successfully recovered {len(global_placed)} shelves from the global canvas.")
    
    # We have 2 clear sheets.
    # The back shelves are at X ~ 0-96.
    # The side shelves are at X ~ 400 or something in the SVG!
    # Let's cleanly separate them based on their physical clustering.
    # We will cluster them using K-Means or just a hard split!
    
    xs = [(s.bounds()[0] + s.bounds()[2])/2 for s in global_placed]
    avg_x = np.median(xs)
    
    sheet_1_shelves = []
    sheet_2_shelves = []
    
    for s in global_placed:
        cx = (s.bounds()[0] + s.bounds()[2])/2
        if cx < avg_x:
            sheet_1_shelves.append(s)
        else:
            sheet_2_shelves.append(s)
            
    print(f"Divided into Sheet 1 ({len(sheet_1_shelves)} shelves) and Sheet 2 ({len(sheet_2_shelves)} shelves) by spatial coordinates!")
    
    # Now perfectly center each group into a 48x96 bounding box
    for sh_shelves in [sheet_1_shelves, sheet_2_shelves]:
        minx = min([s.bounds()[0] for s in sh_shelves])
        miny = min([s.bounds()[1] for s in sh_shelves])
        for s in sh_shelves:
            # Drop them perfectly inside the 0,0 frame before repulsion
            s.translate(-minx + 1.0, -miny + 1.0)
            
    # Relax and Export
    output_target = Path('output')
    
    sh1 = Sheet(1)
    sh1.placed_groups = sheet_1_shelves
    print("Relaxing Sheet 1...")
    relax_layout(sheet_1_shelves)
    export_sheet_to_dxf(sh1, output_target)
    
    sh2 = Sheet(2)
    sh2.placed_groups = sheet_2_shelves
    print("Relaxing Sheet 2...")
    relax_layout(sheet_2_shelves)
    export_sheet_to_dxf(sh2, output_target)

if __name__ == '__main__':
    main()
