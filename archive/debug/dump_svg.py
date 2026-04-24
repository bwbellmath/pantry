import numpy as np
from svgpathtools import svg2paths
from shapely.geometry import Polygon

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

def dump_svg(filename):
    print(f"\n--- {filename} Polys ---")
    paths, _ = svg2paths(filename)
    polys = []
    for i, path in enumerate(paths):
        try:
            p = path_to_poly(path)
            if p and p.area > 500:
                minx, miny, maxx, maxy = p.bounds
                print(f"Poly {i}: Area={p.area:.1f}, Bounds X:[{minx:.1f}, {maxx:.1f}], Y:[{miny:.1f}, {maxy:.1f}], AspectRatio: {(maxx-minx)/(maxy-miny) if maxy-miny>0 else 0:.2f}")
        except: pass

dump_svg('output/sheet_2.svg')
