import numpy as np
from svgpathtools import svg2paths
from shapely.geometry import Polygon
import math

def path_to_poly(path, num_samples=100):
    """Convert an svgpathtools Path to a Shapely Polygon."""
    if not path.isclosed():
        # Maybe force close it? Or skip.
        try:
            path.append(path[-1].reversed()) 
        except:
            pass
            
    points = []
    # Sample points along the path
    for i in range(num_samples):
        try:
            pt = path.point(i / float(num_samples - 1))
            points.append((pt.real, pt.imag))
        except:
            pass
    if len(points) < 3: return None
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def analyze_svg(filename):
    print(f"\n--- Analyzing {filename} ---")
    paths, attributes = svg2paths(filename)
    polys = []
    for path in paths:
        try:
            poly = path_to_poly(path)
            if poly and poly.area > 100:  # arbitrary threshold
                polys.append(poly)
        except Exception as e:
            print("Error parsing a path:", e)
            
    print(f"Found {len(polys)} valid, reasonably-sized polygons.")
    
    if polys:
        areas = [p.area for p in polys]
        print(f"Typical Areas: {areas[:5]}")
        # Bounds of the first one
        minx, miny, maxx, maxy = polys[0].bounds
        print(f"Polys[0] dimensions: {maxx-minx:.1f} x {maxy-miny:.1f}")

        # Check total bounding box of all paths
        all_x = []
        all_y = []
        for p in polys:
            mnx, mny, mxx, mxy = p.bounds
            all_x.extend([mnx, mxx])
            all_y.extend([mny, mxy])
        print(f"Overall bounding box: X=({min(all_x):.1f}, {max(all_x):.1f}), Y=({min(all_y):.1f}, {max(all_y):.1f})")

analyze_svg('output/sheet_1.svg')
analyze_svg('output/sheet_2.svg')
