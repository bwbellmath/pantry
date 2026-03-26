#!/usr/bin/env python3
"""
Export shelf geometries to JSON for use in the nesting UI.

Shelf polygons are:
  - Re-centered so the bounding-box centre is at (0, 0)
  - Y-flipped (DXF Y-up → SVG Y-down)

Run from the pantry project root:
  python scripts/export_shelf_geometry.py
Outputs: nesting_geometry.json
"""

import json
import sys
from pathlib import Path
from shapely.geometry import LineString, MultiLineString, GeometryCollection

sys.path.append(str(Path(__file__).parent))
from generate_nested_layouts import load_shelves

SHEET_W = 96.0
SHEET_H = 48.0


def bbox_center(coords):
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def transform_pts(coords, cx_dxf, cy_dxf):
    """Re-centre around (cx_dxf, cy_dxf) and flip Y."""
    return [[x - cx_dxf, -(y - cy_dxf)] for x, y in coords]


def extract_linestring_coords(geom):
    if isinstance(geom, LineString):
        return [list(geom.coords)]
    if isinstance(geom, (MultiLineString, GeometryCollection)):
        result = []
        for g in geom.geoms:
            result.extend(extract_linestring_coords(g))
        return result
    if hasattr(geom, 'exterior'):
        return [list(geom.exterior.coords)]
    return []


def main():
    print("Loading shelves...")
    shelves = load_shelves()
    print(f"  {len(shelves)} shelves loaded.")

    data = {}
    for s in shelves:
        outer = list(s.poly.exterior.coords)
        cx, cy = bbox_center(outer)

        poly_pts = transform_pts(outer, cx, cy)

        brackets = []
        for b in s.bracket_polys:
            brackets.append(transform_pts(list(b.exterior.coords), cx, cy))

        cuts = []
        for c in s.bracket_cuts:
            for seg in extract_linestring_coords(c):
                cuts.append(transform_pts(list(seg), cx, cy))

        # Bounding box in local (re-centred, Y-down) coords
        xs = [p[0] for p in poly_pts]
        ys = [p[1] for p in poly_pts]

        data[s.name] = {
            "poly":     poly_pts,
            "brackets": brackets,
            "cuts":     cuts,
            "bbox":     [min(xs), min(ys), max(xs), max(ys)],  # [minx, miny, maxx, maxy]
            "shelf_type": s.shelf_type,
        }

    out = Path(__file__).parent.parent / "nesting_geometry.json"
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
