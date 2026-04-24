#!/usr/bin/env python3
"""
Generate per-sheet DXF files from a nesting_layout.json produced by nesting_ui.html.

Usage:
    python generate_dxf_from_layout.py [nesting_layout.json]

Output:
    output/sheet_<N>_nesting.dxf  for each sheet that has shelves

Coordinate conventions
──────────────────────
nesting_ui.html stores shelf positions as:
  x, y   — centroid of the shelf in sheet-local SVG coordinates
             (y=0 = sheet top, y=48 = sheet bottom; Y increases downward)
  rotation  — 0..3 multiples of 90° clockwise (SVG Y-down sense)

The shelf geometry JSON (nesting_geometry.json, exported by
scripts/export_shelf_geometry.py) has polygon coordinates re-centred
around (0,0) with Y increasing downward.

To produce the DXF (Y increases upward):
  1. Rotate the shelf polygon by  rotation × 90°  clockwise in SVG (Y-down)
     → equivalent to  rotation × 90°  counter-clockwise in DXF (Y-up)
  2. Flip Y:  dxf_y = -(svg_y)
  3. Translate so the centroid lands at (x, SHEET_H - y) in DXF coords
"""

import argparse
import json
import math
import sys
from pathlib import Path

import ezdxf
from ezdxf import units

SHEET_W = 96.0
SHEET_H = 48.0

LAYER_BOUNDARY  = "PLYWOOD_BOUNDARY"
LAYER_SHELF     = "SHELF_OUTLINE"
LAYER_BRACKET   = "BRACKET_OUTLINE"
LAYER_CUT       = "BRACKET_CUTS"


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def rot_pt(x, y, turns):
    """Rotate (x,y) by turns×90° clockwise in SVG (Y-down) = CCW in DXF (Y-up)."""
    for _ in range(turns & 3):
        x, y = y, -x
    return x, y


def transform_pts(pts, rot, cx_dxf, cy_dxf):
    """
    pts    : list of [x, y] in SVG-centred (Y-down) local coords
    rot    : rotation turns (0-3)
    cx_dxf : shelf centroid X in DXF sheet coords
    cy_dxf : shelf centroid Y in DXF sheet coords
    """
    out = []
    for x, y in pts:
        rx, ry = rot_pt(x, y, rot)
        # Flip SVG Y-down → DXF Y-up
        out.append((rx + cx_dxf, -ry + cy_dxf))
    return out


# ─── DXF export ───────────────────────────────────────────────────────────────

def add_polyline(msp, pts, layer, closed=True):
    pts3 = [(x, y, 0) for x, y in pts]
    if closed and pts3[0] != pts3[-1]:
        pts3.append(pts3[0])
    msp.add_lwpolyline(pts3, close=closed, dxfattribs={"layer": layer})


def export_sheet_dxf(sheet_id, shelf_items, geom, output_dir):
    doc = ezdxf.new(dxfversion="R2010")
    doc.units = units.IN

    msp = doc.modelspace()

    for layer_name, color in [
        (LAYER_BOUNDARY, 7),
        (LAYER_SHELF,    3),
        (LAYER_BRACKET,  5),
        (LAYER_CUT,      1),
    ]:
        lyr = doc.layers.new(layer_name)
        lyr.color = color

    # Sheet boundary
    add_polyline(msp, [(0,0),(SHEET_W,0),(SHEET_W,SHEET_H),(0,SHEET_H)],
                 LAYER_BOUNDARY)

    for item in shelf_items:
        name  = item["name"]
        g     = geom.get(name)
        if g is None:
            print(f"  WARNING: geometry not found for {name}, skipping.")
            continue

        # SVG layout coords → DXF coords
        cx_svg = item["x"]   # centroid X in sheet (same in DXF)
        cy_svg = item["y"]   # centroid Y in sheet SVG (Y-down, 0=top)
        cy_dxf = SHEET_H - cy_svg   # flip Y
        cx_dxf = cx_svg

        rot = item.get("rotation", 0) & 3

        # Shelf outline
        pts = transform_pts(g["poly"], rot, cx_dxf, cy_dxf)
        add_polyline(msp, pts, LAYER_SHELF)

        # Bracket outlines
        for bpoly in g["brackets"]:
            pts = transform_pts(bpoly, rot, cx_dxf, cy_dxf)
            add_polyline(msp, pts, LAYER_BRACKET)

        # Bracket cuts (open polylines)
        for cpts in g["cuts"]:
            pts = transform_pts(cpts, rot, cx_dxf, cy_dxf)
            if len(pts) >= 2:
                add_polyline(msp, pts, LAYER_CUT, closed=False)

        print(f"  {name:20s}  centre=({cx_dxf:.2f}, {cy_dxf:.2f})  rot={rot*90}°")

    out_path = output_dir / f"sheet_{sheet_id}_nesting.dxf"
    doc.saveas(out_path)
    print(f"  → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate per-sheet DXF files from nesting layout.')
    parser.add_argument('layout', nargs='?', help='Path to nesting_layout.json (optional)')
    parser.add_argument('--project', default='pantry', help='Project name (default: pantry)')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    project_dir = repo_root / 'projects' / args.project

    layout_path = Path(args.layout) if args.layout else project_dir / "nesting_layout.json"
    geom_path   = project_dir / "nesting_geometry.json"
    output_dir  = project_dir / "output"

    if not layout_path.exists():
        print(f"Layout file not found: {layout_path}")
        sys.exit(1)
    if not geom_path.exists():
        print(f"Geometry file not found: {geom_path}")
        print(f"Run:  python tools/export_shelf_geometry.py --project {args.project}")
        sys.exit(1)

    with open(layout_path)  as f: layout = json.load(f)
    with open(geom_path)    as f: geom   = json.load(f)

    output_dir.mkdir(exist_ok=True)

    sheets = layout.get("sheets", [])
    if not sheets:
        print("No sheets found in layout JSON.")
        return

    for sh in sheets:
        shelf_items = sh.get("shelves", [])
        if not shelf_items:
            continue
        sid = sh.get("id", "?")
        print(f"\nSheet {sid}: {len(shelf_items)} shelf/shelves")
        export_sheet_dxf(sid, shelf_items, geom, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
