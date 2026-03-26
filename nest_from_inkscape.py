#!/usr/bin/env python3
"""
Read manual nesting layout from Inkscape SVGs and export to DXF sheets.

SVG coordinate system:
  - viewBox: 0 0 8000 4000 (in SVG mm units)
  - Scale factor: 83.02141 SVG-units per DXF-inch
  - So 96" × 48" plywood sheet → 7970 × 3985 SVG units
  - SVG Y increases downward; nesting DXF Y increases upward

Each shelf group in the SVG has:
  - A <path> in DXF-pantry inch coordinates (original, un-mirrored)
  - A transform matrix(a,b,c,d,e,f) that positions it on the canvas
  - A <text> label like "Shelf R79""

Loaded DXF shelves are X-mirrored (X → 48-X) for bottom-face machining.
The affine transform below accounts for that mirror when mapping to nesting coords.

Polish step:
  - Build adjacency graph of shelves within ADJACENCY_DIST of each other
  - BFS from the top-left anchor shelf (fixed against plywood)
  - For each shelf in traversal order: binary-search separate it from all
    already-placed shelves until minimum Shapely distance = MARGIN (0.5")
  - Final confirmation pass reports any remaining violations
"""

import sys
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import shapely.affinity as affinity

sys.path.append(str(Path(__file__).parent / 'scripts'))
from generate_nested_layouts import (
    ShelfGroup, Sheet, export_sheet_to_dxf, load_shelves,
    SHEET_HEIGHT, SHEET_WIDTH, MARGIN
)

SCALE = 83.02141        # SVG units per DXF inch
PANTRY_WIDTH = 48.0     # inches
ADJACENCY_DIST = 10.0   # inches — shelves closer than this are "adjacent"
SVG_NS = 'http://www.w3.org/2000/svg'


# ---------------------------------------------------------------------------
# SVG parsing + coordinate transform
# ---------------------------------------------------------------------------

def parse_matrix(transform_str):
    """Parse transform="matrix(a,b,c,d,e,f)" → [a,b,c,d,e,f] or None."""
    if not transform_str:
        return None
    m = re.match(r'matrix\(([^)]+)\)', transform_str.strip())
    if not m:
        return None
    vals = [float(v) for v in re.split(r'[\s,]+', m.group(1).strip()) if v]
    return vals if len(vals) == 6 else None


def build_nesting_affine(mat):
    """
    Build Shapely affine_transform coefficients to map DXF-mirrored coords
    to nesting sheet coords.

    Composition:
      1. Un-mirror X:  x → PANTRY_WIDTH - x
      2. Apply SVG matrix at scale SCALE (SVG units/inch)
      3. Divide by SCALE to get inches
      4. Flip Y (SVG Y down → nesting Y up):  y → SHEET_HEIGHT - y

    Shapely format: [a, b, d, e, xoff, yoff]
      x' = a*x + b*y + xoff
      y' = d*x + e*y + yoff
    """
    a, b, c, d, e, f = mat
    s  = SCALE
    pw = PANTRY_WIDTH
    sh = SHEET_HEIGHT
    return [
        -a / s,           # a_s
         c / s,           # b_s
         b / s,           # d_s
        -d / s,           # e_s
        (pw * a + e) / s, # xoff
        sh - (pw * b + f) / s,  # yoff
    ]


def apply_nesting_transform(sg, coeffs):
    """Transform a ShelfGroup's geometry in-place using the nesting affine coeffs."""
    sg.poly         = affinity.affine_transform(sg.poly,   coeffs)
    sg.bracket_polys = [affinity.affine_transform(b, coeffs) for b in sg.bracket_polys]
    sg.bracket_cuts  = [affinity.affine_transform(c, coeffs) for c in sg.bracket_cuts]


def parse_svg_shelves(svg_path):
    """
    Parse an Inkscape SVG, returning [(shelf_name, affine_coeffs), ...].
    shelf_name matches DXF filename stem, e.g. "shelf_R79".
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    results = []

    for g in root.iter(f'{{{SVG_NS}}}g'):
        mat = parse_matrix(g.get('transform'))
        if mat is None:
            continue

        label = None
        for text_elem in g.iter(f'{{{SVG_NS}}}text'):
            raw = ''.join(text_elem.itertext()).strip()
            if raw.startswith('Shelf '):
                label = raw.rstrip('"').strip()   # "Shelf R79"" → "Shelf R79"
                break

        if label is None:
            continue

        suffix     = label.split('Shelf ', 1)[1]   # "R79"
        shelf_name = f'shelf_{suffix}'              # "shelf_R79"
        results.append((shelf_name, build_nesting_affine(mat)))

    return results


# ---------------------------------------------------------------------------
# Polishing: adjacency graph + BFS separation
# ---------------------------------------------------------------------------

def _centroid(sg):
    c = sg.poly.centroid
    return c.x, c.y


def _sep_direction(s_move, s_fixed):
    """
    Return a unit (dx, dy) to push s_move away from s_fixed.

    If their bounding boxes overlap in both axes, use the axis with the
    SMALLER overlap (minimum penetration depth) — this separates with the
    smallest total movement.  If bounding boxes don't overlap (close but not
    touching), fall back to centroid-to-centroid direction.
    """
    b1 = s_move.bounds()    # minx, miny, maxx, maxy
    b2 = s_fixed.bounds()

    x_overlap = min(b1[2], b2[2]) - max(b1[0], b2[0])
    y_overlap = min(b1[3], b2[3]) - max(b1[1], b2[1])

    if x_overlap > 0 and y_overlap > 0:
        # Bounding boxes actually overlap: push along the shorter axis
        c1x = (b1[0] + b1[2]) / 2
        c2x = (b2[0] + b2[2]) / 2
        c1y = (b1[1] + b1[3]) / 2
        c2y = (b2[1] + b2[3]) / 2
        if x_overlap <= y_overlap:
            return (1.0, 0.0) if c1x >= c2x else (-1.0, 0.0)
        else:
            return (0.0, 1.0) if c1y >= c2y else (0.0, -1.0)
    else:
        # No bounding-box overlap: use centroid-to-centroid
        cx1, cy1 = _centroid(s_move)
        cx2, cy2 = _centroid(s_fixed)
        dx, dy = cx1 - cx2, cy1 - cy2
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < 1e-6:
            return (1.0, 0.0)
        return (dx / mag, dy / mag)


def build_adjacency(shelves):
    """Return adj dict: {idx: [neighbor_idx, ...]} for pairs within ADJACENCY_DIST."""
    n = len(shelves)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if shelves[i].distance(shelves[j]) < ADJACENCY_DIST:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_order(adj, anchor_idx, n):
    """BFS traversal starting from anchor.  Returns ordered list of indices."""
    order, seen = [], {anchor_idx}
    q = deque([anchor_idx])
    while q:
        idx = q.popleft()
        order.append(idx)
        for nbr in sorted(adj[idx]):
            if nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


def _clamp_to_sheet(s):
    """Push shelf back inside sheet bounds if it has drifted more than 0.25"."""
    TOL = 0.25
    minx, miny, maxx, maxy = s.bounds()
    cx = cy = 0.0
    if minx < -TOL:           cx =  -TOL - minx
    if maxx > SHEET_WIDTH + TOL:  cx = SHEET_WIDTH + TOL - maxx
    if miny < -TOL:           cy =  -TOL - miny
    if maxy > SHEET_HEIGHT + TOL: cy = SHEET_HEIGHT + TOL - maxy
    if cx or cy:
        s.translate(cx, cy)


def polish_layout(sheet):
    """
    Enforce MARGIN (0.5") separation between shelf pairs.

    Shelves involved in large structural overlaps (polygon area > FREEZE_THRESH)
    are frozen in their Inkscape positions — they require manual layout correction.
    Only shelves free of structural overlaps are moved.

    For each movable violation pair:
      - If both shelves are free: symmetric forces (both move)
      - If one shelf is frozen: only the free shelf moves
    """
    MAX_STEP      = 0.75   # max inches moved per shelf per pass
    MAX_PASSES    = 200
    FREEZE_THRESH = 1.0    # sq.in. — shelves in overlaps larger than this are frozen

    shelves = sheet.get_all_shelves()
    n = len(shelves)
    if n < 2:
        return

    # Mark frozen shelves (any structural overlap > threshold)
    frozen = set()
    for i in range(n):
        for j in range(i + 1, n):
            inter = shelves[i].poly.intersection(shelves[j].poly)
            if (not inter.is_empty) and inter.area > FREEZE_THRESH:
                frozen.add(i)
                frozen.add(j)

    anchor_idx = min(range(n),
                     key=lambda i: (-shelves[i].bounds()[3],
                                     shelves[i].bounds()[0]))
    frozen.add(anchor_idx)

    # Also freeze free shelves that are squeezed between opposing constraints:
    # a shelf is squeezed if the net force is < 20% of the total force magnitude.
    def is_squeezed_free(idx):
        tdx = tdy = total_mag = 0.0
        for other_idx in range(n):
            if other_idx == idx:
                continue
            d = shelves[other_idx].distance(shelves[idx])
            if d >= MARGIN - 1e-4:
                continue
            dx, dy = _sep_direction(shelves[idx], shelves[other_idx])
            needed = MARGIN - d
            tdx += dx * needed
            tdy += dy * needed
            total_mag += needed
        if total_mag < 1e-6:
            return False
        net_mag = (tdx ** 2 + tdy ** 2) ** 0.5
        return net_mag < 0.2 * total_mag

    changed = True
    while changed:
        changed = False
        for idx in list(range(n)):
            if idx in frozen:
                continue
            if is_squeezed_free(idx):
                frozen.add(idx)
                changed = True

    free = [i for i in range(n) if i not in frozen]
    frozen_names = sorted(shelves[i].name for i in frozen if i != anchor_idx)
    if frozen_names:
        print(f"  Frozen (structural overlaps, need Inkscape fix): {frozen_names}")
    print(f"  Anchor: {shelves[anchor_idx].name}")

    for pass_num in range(MAX_PASSES):
        forces = {i: [0.0, 0.0] for i in free}
        any_violation = False

        for i in range(n):
            for j in range(i + 1, n):
                # Skip pairs where BOTH are frozen
                if i in frozen and j in frozen:
                    continue
                d = shelves[i].distance(shelves[j])
                if d >= MARGIN - 1e-4:
                    continue
                any_violation = True
                needed = MARGIN - d
                dx, dy = _sep_direction(shelves[i], shelves[j])
                if i not in frozen:
                    forces[i][0] += dx * needed
                    forces[i][1] += dy * needed
                if j not in frozen:
                    forces[j][0] -= dx * needed
                    forces[j][1] -= dy * needed

        if not any_violation:
            print(f"  Converged after {pass_num + 1} pass(es).")
            return

        for i in free:
            fx, fy = forces[i]
            if abs(fx) < 1e-6 and abs(fy) < 1e-6:
                continue
            mag = (fx ** 2 + fy ** 2) ** 0.5
            if mag > MAX_STEP:
                fx = fx / mag * MAX_STEP
                fy = fy / mag * MAX_STEP
            shelves[i].translate(fx, fy)
            _clamp_to_sheet(shelves[i])

    print(f"  Reached {MAX_PASSES} passes — remaining violations may be geometrically infeasible.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_layout(sheet, label=""):
    """Report gap and bounds violations.  Returns issue count."""
    shelves = sheet.get_all_shelves()
    n = len(shelves)
    issues = 0

    for i in range(n):
        minx, miny, maxx, maxy = shelves[i].bounds()
        if minx < -0.05 or miny < -0.05 or maxx > SHEET_WIDTH + 0.05 or maxy > SHEET_HEIGHT + 0.05:
            print(f"  [bounds]  {shelves[i].name}: "
                  f"({minx:.3f},{miny:.3f})–({maxx:.3f},{maxy:.3f})")
            issues += 1

    for i in range(n):
        for j in range(i + 1, n):
            d = shelves[i].distance(shelves[j])
            if d < MARGIN - 0.01:
                # Distinguish actual polygon overlap from touching/near-touching
                inter = shelves[i].poly.intersection(shelves[j].poly)
                overlap_area = inter.area if not inter.is_empty else 0.0
                tag = f"OVERLAP area={overlap_area:.3f}" if overlap_area > 0.001 else "touching"
                print(f"  [gap={d:.4f}\"] {shelves[i].name} ↔ {shelves[j].name}  [{tag}]")
                issues += 1

    tag = f" ({label})" if label else ""
    if issues == 0:
        print(f"  OK{tag}: {n} shelves, all gaps ≥ {MARGIN}\"")
    else:
        print(f"  {issues} issue(s){tag}")
    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_dir = Path(__file__).parent
    output_dir  = project_dir / 'output'

    print("Loading DXF shelves...")
    all_shelves = load_shelves()
    shelf_dict  = {s.name: s for s in all_shelves}
    print(f"  Loaded {len(shelf_dict)} shelves.")

    svg_files = [
        project_dir / 'Inkscape' / 'Panty_Sheet_1.svg',
        project_dir / 'Inkscape' / 'Panty_Sheet_2.svg',
    ]

    for sheet_num, svg_path in enumerate(svg_files, 1):
        if not svg_path.exists():
            print(f"SVG not found: {svg_path}")
            continue

        print(f"\n--- Sheet {sheet_num}: {svg_path.name} ---")
        svg_groups = parse_svg_shelves(svg_path)
        print(f"  {len(svg_groups)} shelf groups in SVG")

        sheet = Sheet(sheet_num)

        for shelf_name, coeffs in svg_groups:
            if shelf_name not in shelf_dict:
                print(f"  WARNING: '{shelf_name}' not found in DXF shelf set")
                continue
            sg = shelf_dict.pop(shelf_name)
            apply_nesting_transform(sg, coeffs)
            sheet.add(sg)

        validate_layout(sheet, "before polish")

        print("  Polishing layout...")
        polish_layout(sheet)

        validate_layout(sheet, "after polish")
        export_sheet_to_dxf(sheet, output_dir)

    if shelf_dict:
        print(f"\nUnplaced shelves: {sorted(shelf_dict)}")


if __name__ == '__main__':
    main()
