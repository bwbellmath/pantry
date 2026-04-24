#!/usr/bin/env python3
"""
Post-process nesting layout into CNC-ready DXF files.

Reads:  nesting_layout.json  (exported by nesting_ui.html)
        nesting_geometry.json (exported by scripts/export_shelf_geometry.py)

Outputs TWO DXF files per sheet:
  sheet_<N>_contours.dxf — shelf outlines with bracket-base relief slots (through-cuts)
  sheet_<N>_pockets.dxf  — bracket tongue polygons only (partial-depth for Adaptive Clearing)

  Pocket polygons extend 0.38" past the wall edge (not starting at the armpit)
  so the 3/8" endmill clears past the bracket relief without Fusion extent tricks.

Bracket-base slot geometry (inserted into shelf outline at the wall edge):
  • 0.5"-radius arc leading in from wall edge
  • Flat line at base_height (0.2") from wall
  • 0.5"-radius arc back out to wall edge
  • The straight wall-edge segment over the slot is removed

Pocket geometry (closed curve for Adaptive Clearing):
  • Closed rectangle: armpit → tongue sides with dogbone arcs → 0.38" past tip → close

Left-shelf D-shaped pipe cutout:
  • When a bracket slot falls partially over the D-shape cutout (already empty),
    the slot is clipped — only the portion over solid shelf material is drawn.

Usage:
    python process_for_cnc.py [nesting_layout.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import ezdxf
from ezdxf import units

# ── Constants ────────────────────────────────────────────────────────────────

SHEET_W = 96.0
SHEET_H = 48.0

BASE_HEIGHT    = 0.2    # bracket base plate depth (into shelf from wall)
BASE_WIDTH     = 3.2    # bracket base plate width (along wall, centered on stud)
STEM_WIDTH     = 1.6    # bracket tongue width
BASE_R         = 0.5    # outer corner arc radius at wall entry
DOGBONE_R      = 0.25   # dogbone relief radius at tongue tip corners
POCKET_EXTEND  = 0.38   # how far past tongue tip the pocket extends to close

# chord_half: where the wall-entry arc crosses y=0 (x-offset from base edge)
_wcy       = BASE_HEIGHT - BASE_R          # arc center y (= -0.3)
CHORD_HALF = math.sqrt(BASE_R**2 - _wcy**2)   # ≈ 0.4"

WALL_TOL = 0.05   # tolerance for "is this point on the wall edge"

LAYER_CONTOUR = "CONTOUR"
LAYER_POCKET  = "POCKET"

N_ARC = 24   # points per arc segment


# ── Geometry helpers ──────────────────────────────────────────────────────────

def rot_pt(x, y, turns):
    """Rotate (x,y) by turns×90° CW in SVG Y-down coords."""
    for _ in range(turns & 3):
        x, y = y, -x
    return x, y


def transform_pts(pts, rot, cx_dxf, cy_dxf):
    """
    pts: [[x,y] ...] in local SVG-centred Y-down coords
    Returns list of (x,y) tuples in DXF sheet coords (Y-up).
    """
    out = []
    for x, y in pts:
        rx, ry = rot_pt(x, y, rot)
        out.append((rx + cx_dxf, -ry + cy_dxf))
    return out


def arc_pts(cx, cy, r, a_start_deg, a_end_deg, n=N_ARC):
    """Sample a CCW arc from a_start_deg to a_end_deg."""
    a0 = math.radians(a_start_deg)
    a1 = math.radians(a_end_deg)
    # Ensure CCW (a1 > a0)
    while a1 <= a0:
        a1 += 2 * math.pi
    pts = []
    for i in range(n + 1):
        t = a0 + (a1 - a0) * i / n
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def lerp_pt(a, b, t):
    return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))


def seg_param(a, b, p, tol=0.02):
    """
    If p lies on segment a→b (within tol), return parameter t in [0,1].
    Returns None otherwise.
    """
    dx, dy = b[0]-a[0], b[1]-a[1]
    L2 = dx*dx + dy*dy
    if L2 < 1e-12:
        return None
    t = ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / L2
    if t < -0.001 or t > 1.001:
        return None
    px = a[0] + t*dx
    py = a[1] + t*dy
    if (px-p[0])**2 + (py-p[1])**2 > tol**2:
        return None
    return t


# ── Bracket-base U-shape construction ────────────────────────────────────────

def bracket_entry_points(cut_pts_dxf, shelf_type):
    """
    Compute the two wall-edge entry/exit points of the bracket base slot,
    in DXF sheet coords.

    cut_pts_dxf: [(x1,y1),(x2,y2)] — the armpit cut line, already in DXF coords.

    The armpit midpoint is the stud center projected onto the wall face + base_height.
    The wall-edge points are base_height further out, ±(BASE_WIDTH/2 + CHORD_HALF)
    from the stud center along the wall.
    """
    (x1, y1), (x2, y2) = cut_pts_dxf
    mx, my = (x1+x2)/2, (y1+y2)/2

    # Direction along wall (from cut line tangent)
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    ux, uy = dx/L, dy/L   # unit vector along wall (stem_width direction)

    # Direction INTO shelf (perpendicular, pointing away from wall)
    # For the armpit, nx,ny points into the shelf from the armpit toward the tongue tip.
    # The wall-entry points are in the OPPOSITE direction (toward the wall).
    # We figure out the wall-outward direction from shelf_type applied in DXF coords.
    # In DXF coords (Y-up): we can infer wall direction from armpit orientation.
    # The outward-wall normal is perpendicular to ux,uy.
    # We have two choices: (-uy, ux) or (uy, -ux).
    # The wall-edge point = armpit_midpoint moved BASE_HEIGHT toward wall.

    # Determine wall outward direction by shelf_type.
    # In DXF sheet coords (Y-up), the shelf types map to:
    #   R shelf → wall on left side → normal pointing -x initially, but after
    #             nesting rotation this could change. Use cut-line orientation.
    # Simpler: the wall-outward direction is perpendicular to ux,uy such that
    # the wall point is BASE_HEIGHT from mx,my. There are two candidates;
    # we return both and let the caller pick based on shelf polygon wall edge.
    # Actually: we can determine it from the shelf bbox later. Return both candidates.

    half_span = BASE_WIDTH / 2 + CHORD_HALF   # = 2.0"

    # Wall entry/exit points (before we know which perpendicular direction is outward)
    # The two wall-edge points are ±half_span along the wall from the stud center,
    # at the wall face (BASE_HEIGHT outward from armpit).
    # We need the outward perpendicular. We'll return BOTH candidate directions.
    # The correct one will lie on the shelf wall edge (within WALL_TOL).
    candidates = []
    for sign in (1, -1):
        nx, ny = sign * (-uy), sign * ux   # perpendicular to ux,uy
        # Wall face = midpoint moved BASE_HEIGHT outward
        wx = mx + BASE_HEIGHT * nx
        wy = my + BASE_HEIGHT * ny
        # Two entry/exit points along the wall
        p1 = (wx - half_span * ux, wy - half_span * uy)
        p2 = (wx + half_span * ux, wy + half_span * uy)
        candidates.append((p1, p2, (nx, ny)))

    return candidates   # caller picks the right one


def make_slot_upath(p_entry, p_exit, n_inward, slot_width, depth, r=BASE_R, n=N_ARC//4):
    """
    Build the U-shape path for the bracket base slot in DXF coords.

    Constructs arcs in a local (u,v) bracket frame using the same angle
    ranges as generate_shelves_with_brackets.py, then transforms to DXF.

    Local frame: origin at midpoint of entry/exit on wall face,
      u-axis along wall (entry→exit), v-axis into shelf (n_inward).
    """
    dx = p_exit[0] - p_entry[0]
    dy = p_exit[1] - p_entry[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    nx, ny = n_inward

    wmx = (p_entry[0] + p_exit[0]) / 2
    wmy = (p_entry[1] + p_exit[1]) / 2

    def to_dxf(u, v):
        return (wmx + u * ux + v * nx, wmy + u * uy + v * ny)

    hw  = L / 2
    wcy = depth - r          # arc center v-coord = -0.3
    ch  = math.sqrt(r**2 - wcy**2)   # chord half ≈ 0.4

    # Arc centers in local coords
    u_lc = -hw + ch     # left  arc center u (= -BASE_WIDTH/2 when unclipped)
    u_rc =  hw - ch     # right arc center u (= +BASE_WIDTH/2 when unclipped)

    # --- Left (entry-side) arc: wall → base interior ---
    # Reference left-corner arc is CCW from 90° to arctan2(-wcy, -ch).
    # U-path traverses entry→interior = reverse direction = CW.
    # Angles: from arctan2(-wcy, -ch) ≈ 143° down to 90°.
    a_l_start = math.atan2(-wcy, -ch)
    a_l_end   = math.pi / 2
    left_arc = []
    for i in range(n + 1):
        a = a_l_start + (a_l_end - a_l_start) * i / n
        left_arc.append(to_dxf(u_lc + r * math.cos(a), wcy + r * math.sin(a)))

    # --- Right (exit-side) arc: base interior → wall ---
    # Reference right-corner arc is CCW from arctan2(-wcy, ch) to 90°.
    # U-path traverses interior→exit = reverse = CW.
    # Angles: from 90° down to arctan2(-wcy, ch) ≈ 37°.
    a_r_start = math.pi / 2
    a_r_end   = math.atan2(-wcy, ch)
    right_arc = []
    for i in range(n + 1):
        a = a_r_start + (a_r_end - a_r_start) * i / n
        right_arc.append(to_dxf(u_rc + r * math.cos(a), wcy + r * math.sin(a)))

    return left_arc + right_arc


# ── D-cutout / bracket-relief intersection helpers ──────────────────────────

def seg_seg_intersect(a1, a2, b1, b2):
    """
    Intersection of segments a1→a2 and b1→b2.
    Returns (point, t) where t is the parameter along a1→a2, or None.
    """
    d1x, d1y = a2[0]-a1[0], a2[1]-a1[1]
    d2x, d2y = b2[0]-b1[0], b2[1]-b1[1]
    cross = d1x*d2y - d1y*d2x
    if abs(cross) < 1e-10:
        return None
    dx, dy = b1[0]-a1[0], b1[1]-a1[1]
    t = (dx*d2y - dy*d2x) / cross
    u = (dx*d1y - dy*d1x) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (a1[0] + t*d1x, a1[1] + t*d1y), t
    return None


def point_in_span_gap(p, spans, axis):
    """True if p's along-wall coord falls in a gap between spans (D-cutout void)."""
    if len(spans) < 2:
        return False
    a = p[1] if axis == 'x' else p[0]
    for i in range(len(spans) - 1):
        if spans[i][1] - 0.05 < a < spans[i+1][0] + 0.05:
            return True
    return False


def truncate_upath_at_polygon(upath, poly_pts, from_start):
    """
    Walk the upath from one end and find where it first crosses a shelf
    polygon edge (the D-cutout boundary).  Truncate there.

    from_start=True  → entry side in void, search forward
    from_start=False → exit side in void, search backward

    Returns (truncated_upath, intersection_point) or None.
    """
    n_poly = len(poly_pts)
    ref = upath[0] if from_start else upath[-1]

    if from_start:
        for i in range(len(upath) - 1):
            a1, a2 = upath[i], upath[i+1]
            best_t = 2.0
            best_pt = None
            for j in range(n_poly):
                b1 = poly_pts[j]
                b2 = poly_pts[(j+1) % n_poly]
                result = seg_seg_intersect(a1, a2, b1, b2)
                if result:
                    pt, t = result
                    if dist2(pt, ref) > 0.0025 and t < best_t:
                        best_t = t
                        best_pt = pt
            if best_pt is not None:
                return [best_pt] + list(upath[i+1:]), best_pt
        return None
    else:
        for i in range(len(upath) - 2, -1, -1):
            a1, a2 = upath[i], upath[i+1]
            best_t = -1.0
            best_pt = None
            for j in range(n_poly):
                b1 = poly_pts[j]
                b2 = poly_pts[(j+1) % n_poly]
                result = seg_seg_intersect(a1, a2, b1, b2)
                if result:
                    pt, t = result
                    if dist2(pt, ref) > 0.0025 and t > best_t:
                        best_t = t
                        best_pt = pt
            if best_pt is not None:
                return list(upath[:i+1]) + [best_pt], best_pt
        return None


# ── Shelf wall-edge analysis ──────────────────────────────────────────────────

def find_wall_coord(shelf_type, bbox_local):
    """
    Return (axis, wall_val) where axis='x' or 'y' and wall_val is the
    wall-edge coordinate in local shelf space.
    """
    if shelf_type in ('R', 'R_only'):
        return 'x', bbox_local[0]    # min x in local coords
    elif shelf_type == 'L':
        return 'x', bbox_local[2]    # max x in local coords
    elif shelf_type in ('B', 'B_left', 'B_right'):
        return 'y', bbox_local[1]    # min y in local coords
    else:
        return None, None


def find_wall_edge_spans(poly_pts, axis, wall_val, tol=WALL_TOL):
    """
    Find all polygon EDGES (both endpoints on the wall) and return their spans
    as (along_min, along_max) pairs.  Multiple spans indicate gaps in the wall
    edge (e.g., D-shaped pipe cutout on left shelves).

    axis     : 'x' → wall defined by x=wall_val; 'y' → wall defined by y=wall_val
    along(p) : the coordinate perpendicular to the wall-normal (the "run" direction)
    poly_pts : list of (x,y) in DXF sheet coords.
    """
    def coord(p): return p[0] if axis == 'x' else p[1]
    def along(p): return p[1] if axis == 'x' else p[0]

    n = len(poly_pts)
    raw = []
    for i in range(n):
        a = poly_pts[i]
        b = poly_pts[(i + 1) % n]
        if abs(coord(a) - wall_val) < tol and abs(coord(b) - wall_val) < tol:
            lo = min(along(a), along(b))
            hi = max(along(a), along(b))
            raw.append((lo, hi))

    if not raw:
        return []

    # Merge overlapping / adjacent spans
    raw.sort()
    merged = [list(raw[0])]
    for lo, hi in raw[1:]:
        if lo <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def clip_bracket_to_wall_spans(p1, p2, spans, axis):
    """
    Given bracket entry/exit points p1, p2 (both on wall edge) and the list of
    wall-edge spans, clip the bracket opening to the actual wall material.

    p1, p2 are the wall-edge points of the bracket slot (sorted along wall).
    Returns list of (clipped_p1, clipped_p2) pairs — usually just one pair,
    or empty if entirely in a gap.
    """
    def along(p):
        return p[1] if axis == 'x' else p[0]

    a1 = along(p1)
    a2 = along(p2)
    if a1 > a2:
        a1, a2 = a2, a1
        p1, p2 = p2, p1

    result = []
    for lo, hi in spans:
        # Check overlap
        cl = max(a1, lo)
        ch = min(a2, hi)
        if ch <= cl + 0.001:
            continue   # no overlap with this span
        # Compute clipped wall-edge points (interpolate along wall)
        total = a2 - a1
        t1 = (cl - a1) / total if total > 0 else 0
        t2 = (ch - a1) / total if total > 0 else 1
        cp1 = lerp_pt(p1, p2, t1)
        cp2 = lerp_pt(p1, p2, t2)
        result.append((cp1, cp2))

    return result


# ── Shelf polygon wall-segment insertion ──────────────────────────────────────

def insert_slots_into_poly(shelf_pts_dxf, slots_info):
    """
    Insert bracket base U-shapes into the shelf polygon outline.

    shelf_pts_dxf: list of (x,y) in DXF sheet coords (closed polygon)
    slots_info: list of dicts with keys:
        'p1', 'p2'    : wall-edge entry/exit points (DXF coords)
        'upath'       : U-shape points from p1 into shelf and back to p2
        'axis'        : 'x' or 'y'
        'wall_val'    : wall edge coordinate

    Returns new polygon point list with slots inserted.
    """
    # Work on a copy (close if needed)
    pts = list(shelf_pts_dxf)
    if dist2(pts[0], pts[-1]) < 1e-8:
        pts = pts[:-1]   # remove duplicate closing point

    for slot in slots_info:
        p1, p2 = slot['p1'], slot['p2']
        upath  = slot['upath']

        # Find the two edges of the polygon that contain p1 and p2.
        # The slot lies on a wall-edge segment (or spans two collinear segments).
        # We walk the polygon and insert the U-shape between p1 and p2.
        n = len(pts)
        found = []
        for i in range(n):
            a = pts[i]
            b = pts[(i+1) % n]
            for p_target, label in ((p1, 'p1'), (p2, 'p2')):
                t = seg_param(a, b, p_target)
                if t is not None:
                    found.append({'edge': i, 't': t, 'pt': p_target, 'label': label})

        if len(found) < 2:
            # Couldn't find both insertion points — skip this slot
            continue

        # Deduplicate: if a point appears more than once (e.g., at a polygon vertex
        # found on both edge i-1 t=1 and edge i t=0), keep only the higher-edge entry
        # (t=1 on i-1) to avoid wrap-around skips.
        seen_labels = {}
        for entry in found:
            lbl = entry['label']
            if lbl not in seen_labels:
                seen_labels[lbl] = entry
            else:
                # Keep the one that avoids a t≈0 on a later edge causing large skips.
                # Prefer t closer to 1 on a lower edge index, or t closer to 0 on the same.
                existing = seen_labels[lbl]
                # Promote t≈0 entries to t=1 on previous edge to keep them local
                if entry['t'] < 1e-3 and entry['edge'] > existing['edge']:
                    pass  # keep existing (earlier edge)
                elif existing['t'] < 1e-3 and existing['edge'] > entry['edge']:
                    seen_labels[lbl] = entry
                else:
                    # Keep whichever has larger edge+t (further along polygon)
                    if entry['edge'] + entry['t'] > existing['edge'] + existing['t']:
                        seen_labels[lbl] = entry
        found = list(seen_labels.values())

        if len(found) < 2:
            continue

        # Normalize t≈0 on edge i to t=1 on edge i-1 (avoids wrap-around polygon skip)
        for entry in found:
            if entry['t'] < 1e-3 and entry['edge'] > 0:
                entry['edge'] -= 1
                entry['t'] = 1.0 - 1e-9

        # Sort by traversal order around polygon
        found.sort(key=lambda f: f['edge'] + f['t'])

        f0 = found[0]
        f1 = found[1]

        # Determine which insertion point is p1 (the one where U enters) vs p2 (exits).
        # The U-shape goes: p1 → dip → p2, so we insert it between f0 and f1.
        # Identify which found entry is p1 vs p2.
        if f0['label'] == 'p1' and f1['label'] == 'p2':
            insert_pts = upath
        else:
            insert_pts = list(reversed(upath))

        # Build new polygon:
        # Keep all pts up to and including f0 insertion point,
        # then insert U-shape,
        # then skip polygon vertices between f0 and f1,
        # then continue from f1 onward.
        new_pts = []
        i = 0
        while i < n:
            a = pts[i]
            b = pts[(i+1) % n]

            if i == f0['edge']:
                # Add portion of edge up to f0 insertion point
                if f0['t'] > 1e-4:
                    new_pts.append(a)
                new_pts.append(f0['pt'])
                # Insert U-shape
                new_pts.extend(insert_pts)
                new_pts.append(f1['pt'])

                # Skip edges from f0 up to and including f1
                if f1['edge'] == f0['edge']:
                    # Same edge — nothing to skip, just continue after f1
                    i += 1
                else:
                    # Skip to edge after f1
                    i = f1['edge'] + 1
                continue

            new_pts.append(a)
            i += 1

        pts = new_pts

    return pts


# ── Pocket (tongue) construction ──────────────────────────────────────────────

def make_pocket_polygon(cut_pts_dxf, bracket_pts_dxf, shelf_type, bbox_local):
    """
    Build a closed pocket polygon for the bracket tongue in DXF sheet coords.

    Constructs in a local (u,v) bracket frame using the same angle ranges
    as generate_shelves_with_brackets.py, then transforms to DXF.

    Local frame: origin at armpit midpoint, u along wall (cut line direction),
      v into shelf (toward tongue tip).

    The pocket starts at v = -(BASE_HEIGHT + POCKET_EXTEND) so the 3/8" endmill
    clears past the bracket relief slot in the shelf contour.
    """
    (x1, y1), (x2, y2) = cut_pts_dxf

    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    ux, uy = dx / L, dy / L

    # Determine inward direction (toward tongue tip)
    cand_a = (-uy,  ux)
    cand_b = ( uy, -ux)
    bx = sum(p[0] for p in bracket_pts_dxf) / len(bracket_pts_dxf)
    by = sum(p[1] for p in bracket_pts_dxf) / len(bracket_pts_dxf)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    to_tip = (bx - mx, by - my)
    if cand_a[0] * to_tip[0] + cand_a[1] * to_tip[1] >= 0:
        nx, ny = cand_a
    else:
        nx, ny = cand_b

    def to_dxf(u, v):
        return (mx + u * ux + v * nx, my + u * uy + v * ny)

    hw = STEM_WIDTH / 2   # 0.8

    # Dogbone parameters (matching reference: generate_shelves_with_brackets.py)
    R  = DOGBONE_R                  # 0.25"
    d  = R / math.sqrt(2)           # 0.1768" — offset from rectangle corner to circle center
    Rd = R * math.sqrt(2)           # 0.3536" — diagonal reach of circle

    # Find tongue_length: distance from armpit to tongue-tip rectangle corners.
    # The bracket polygon includes dogbone arcs that overshoot the rectangle corner
    # by R*(1 - 1/√2) ≈ 0.073".  The max projection gives us tip_d (arc peak),
    # so tongue_length = tip_d - overshoot.
    max_proj = -1e9
    for p in bracket_pts_dxf:
        proj = (p[0] - mx) * nx + (p[1] - my) * ny
        if proj > max_proj:
            max_proj = proj
    tongue_length = max_proj - R * (1.0 - 1.0 / math.sqrt(2))

    # Verify: rectangle corners at (±hw, tongue_length), dogbone centers at
    # (±hw ∓ d, tongue_length - d), each circle passes through its corner at
    # distance √(d²+d²) = d√2 = R.  Connecting line between arcs at v = tongue_length.
    #
    # Pocket spans from v = -0.38 (in reference frame, 0.38" past wall)
    # to v = tongue_length (the tip line), so total = tongue_length + 0.38.
    # For 4" brackets: 4.380";  6": 6.380";  10": 10.380".

    # Pocket start: 0.38" past wall in outward direction.
    # Wall is at v = -BASE_HEIGHT from armpit, so start_v = -BASE_HEIGHT - POCKET_EXTEND.
    start_v = -(BASE_HEIGHT + POCKET_EXTEND)   # -0.58 in local (= -0.38 in reference frame)

    # --- Right dogbone: center at (hw - d, tongue_length - d) ---
    # 180° CCW from -45° to 135° (matching reference exactly)
    cr_u, cr_v = hw - d, tongue_length - d
    n_db = N_ARC
    right_db = []
    for i in range(n_db + 1):
        a = math.radians(-45) + math.radians(180) * i / n_db
        right_db.append(to_dxf(cr_u + R * math.cos(a), cr_v + R * math.sin(a)))
    # right_db[0]  = (hw,      tongue_length - Rd)  on right wall
    # right_db[-1] = (hw - Rd, tongue_length)       on tip line

    # --- Left dogbone: center at (-hw + d, tongue_length - d) ---
    # 180° CCW from 45° to 225° (matching reference exactly)
    cl_u, cl_v = -hw + d, tongue_length - d
    left_db = []
    for i in range(n_db + 1):
        a = math.radians(45) + math.radians(180) * i / n_db
        left_db.append(to_dxf(cl_u + R * math.cos(a), cl_v + R * math.sin(a)))
    # left_db[0]  = (-hw + Rd, tongue_length)       on tip line
    # left_db[-1] = (-hw,      tongue_length - Rd)  on left wall

    # Build closed pocket (no extension past tip — tip bounded by dogbone arcs
    # and connecting line at v = tongue_length):
    #
    # right start → up right wall → right dogbone → connecting line →
    # left dogbone → down left wall → left start → close
    pocket = []
    pocket.append(to_dxf( hw, start_v))      # base end right  (v = -0.38 ref)
    pocket.append(right_db[0])                # right wall at dogbone entry
    pocket.extend(right_db)                   # right dogbone arc → exits on tip line
    # right_db[-1] to left_db[0]: straight connecting line at v = tongue_length
    pocket.extend(left_db)                    # left dogbone arc → exits on left wall
    pocket.append(to_dxf(-hw, start_v))       # base end left   (v = -0.38 ref)
    pocket.append(to_dxf( hw, start_v))       # close

    # Deduplicate consecutive identical points
    clean = [pocket[0]]
    for p in pocket[1:]:
        if dist2(p, clean[-1]) > 1e-8:
            clean.append(p)
    return clean


# ── DXF helpers ──────────────────────────────────────────────────────────────

def add_polyline(msp, pts, layer, closed=True):
    pts3 = [(x, y, 0) for x, y in pts]
    if closed and dist2(pts3[0][:2], pts3[-1][:2]) > 1e-8:
        pts3.append(pts3[0])
    msp.add_lwpolyline(pts3, close=closed, dxfattribs={'layer': layer})


# ── Per-shelf processing ──────────────────────────────────────────────────────

def process_shelf(item, geom_entry, msp_contour, msp_pocket):
    """
    Add CONTOUR geometry to msp_contour and POCKET geometry to msp_pocket
    for one shelf placement.
    """
    name = item['name']
    rot  = item.get('rotation', 0) & 3
    cx_svg = item['x']
    cy_svg = item['y']
    cy_dxf = SHEET_H - cy_svg
    cx_dxf = cx_svg

    g = geom_entry
    shelf_type = g.get('shelf_type', '')
    bbox_local = g['bbox']   # [minx, miny, maxx, maxy] in local Y-down coords

    # Transform all geometry to DXF sheet coords
    shelf_pts = transform_pts(g['poly'], rot, cx_dxf, cy_dxf)
    brackets_dxf = [transform_pts(b, rot, cx_dxf, cy_dxf) for b in g['brackets']]
    cuts_dxf     = [transform_pts(c, rot, cx_dxf, cy_dxf) for c in g['cuts']]

    # ── Collect bracket slots (for CONTOUR) and pockets ──────────────────────
    slots_info = []
    pockets    = []

    for i, (brk_pts, cut_pts) in enumerate(zip(brackets_dxf, cuts_dxf)):
        if len(cut_pts) < 2:
            continue

        c1, c2 = cut_pts[0], cut_pts[1]
        mx, my = (c1[0]+c2[0])/2, (c1[1]+c2[1])/2

        # Direction along wall
        dx, dy = c2[0]-c1[0], c2[1]-c1[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx/L, dy/L

        # Two perpendicular candidates for "wall outward" direction
        cands = bracket_entry_points(cut_pts, shelf_type)
        if not cands:
            continue

        # Pick the candidate whose wall-edge points actually lie on the shelf polygon
        chosen = None
        for (p1, p2, (nx, ny)) in cands:
            # Check if either wall-edge point is near any shelf edge segment
            on_poly = False
            n_pts = len(shelf_pts)
            for j in range(n_pts):
                a = shelf_pts[j]
                b = shelf_pts[(j+1) % n_pts]
                if seg_param(a, b, p1, tol=0.15) is not None:
                    on_poly = True
                    break
                if seg_param(a, b, p2, tol=0.15) is not None:
                    on_poly = True
                    break
            if on_poly:
                chosen = (p1, p2, (nx, ny))
                break

        if chosen is None:
            # Fall back: pick the candidate where points are closest to any shelf vertex
            def min_dist_to_shelf(p):
                return min(dist2(p, q) for q in shelf_pts)
            c0 = cands[0]; c1c = cands[1]
            d0 = min(min_dist_to_shelf(c0[0]), min_dist_to_shelf(c0[1]))
            d1 = min(min_dist_to_shelf(c1c[0]), min_dist_to_shelf(c1c[1]))
            chosen = c0 if d0 <= d1 else c1c

        # chosen[2] is the direction FROM armpit TOWARD wall (outward).
        # We need the inward direction (from wall into shelf), so flip it.
        p1, p2, (n_out_x, n_out_y) = chosen
        n_inward = (-n_out_x, -n_out_y)

        # Per-bracket wall axis/value detection (back shelves have brackets
        # on 3 different walls, so we must not share axis across brackets).
        if abs(p1[0] - p2[0]) < 0.3:
            brk_axis = 'x'
            brk_wall_val = (p1[0] + p2[0]) / 2
        else:
            brk_axis = 'y'
            brk_wall_val = (p1[1] + p2[1]) / 2

        # D-shape clipping: find wall-edge spans for THIS bracket's wall
        spans = find_wall_edge_spans(shelf_pts, brk_axis, brk_wall_val)

        # Clip bracket to wall material spans (handles corners)
        if spans:
            clipped = clip_bracket_to_wall_spans(p1, p2, spans, brk_axis)
        else:
            clipped = [(p1, p2)]

        # Check if original p1/p2 falls in a D-cutout gap (between spans)
        p1_in_gap = point_in_span_gap(p1, spans, brk_axis) if spans else False
        p2_in_gap = point_in_span_gap(p2, spans, brk_axis) if spans else False

        for (cp1, cp2) in clipped:
            dx2 = cp2[0]-cp1[0]
            dy2 = cp2[1]-cp1[1]
            L2 = math.hypot(dx2, dy2)
            if L2 < 0.01:
                continue

            if p1_in_gap or p2_in_gap:
                # D-cutout case: build full U-shape, truncate at void
                # boundary so the relief merges with the cutout outline.
                full_L = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                upath = make_slot_upath(p1, p2, n_inward, full_L,
                                        BASE_HEIGHT)
                slot_p1, slot_p2 = p1, p2
                if p1_in_gap:
                    result = truncate_upath_at_polygon(
                        upath, shelf_pts, from_start=True)
                    if result is None:
                        continue
                    upath, slot_p1 = result
                if p2_in_gap:
                    result = truncate_upath_at_polygon(
                        upath, shelf_pts, from_start=False)
                    if result is None:
                        continue
                    upath, slot_p2 = result
            else:
                # Normal or corner-clipped: U-shape at clipped width
                upath = make_slot_upath(cp1, cp2, n_inward, L2,
                                        BASE_HEIGHT)
                slot_p1, slot_p2 = cp1, cp2

            slots_info.append({
                'p1': slot_p1, 'p2': slot_p2,
                'upath': upath,
                'axis': brk_axis,
                'wall_val': brk_wall_val,
            })

        # Build pocket polygon
        pocket_pts = make_pocket_polygon(cut_pts, brk_pts, shelf_type, bbox_local)
        if pocket_pts:
            pockets.append(pocket_pts)

    # ── CONTOUR: insert slots into shelf polygon ──────────────────────────────
    contour_pts = insert_slots_into_poly(shelf_pts, slots_info)
    if contour_pts:
        add_polyline(msp_contour, contour_pts, LAYER_CONTOUR, closed=True)
        print(f'    {name}: contour {len(contour_pts)} pts, '
              f'{len(slots_info)} slot(s)')

    # ── POCKET: emit closed tongue polygons ──────────────────────────────────
    for j, pk in enumerate(pockets):
        add_polyline(msp_pocket, pk, LAYER_POCKET, closed=True)
    print(f'    {name}: {len(pockets)} pocket(s)')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Post-process nesting layout into CNC-ready DXF files.')
    parser.add_argument('layout', nargs='?', help='Path to nesting_layout.json (optional)')
    parser.add_argument('--project', default='pantry', help='Project name (default: pantry)')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    project_dir = repo_root / 'projects' / args.project

    layout_path = Path(args.layout) if args.layout else project_dir / 'nesting_layout.json'
    geom_path   = project_dir / 'nesting_geometry.json'
    output_dir  = project_dir / 'output'

    if not layout_path.exists():
        print(f'Layout not found: {layout_path}'); sys.exit(1)
    if not geom_path.exists():
        print(f'Geometry not found: {geom_path}'); sys.exit(1)

    with open(layout_path) as f: layout = json.load(f)
    with open(geom_path)   as f: geom   = json.load(f)

    output_dir.mkdir(exist_ok=True)

    for sh in layout.get('sheets', []):
        items = sh.get('shelves', [])
        if not items:
            continue
        sid = sh.get('id', '?')
        print(f'\nSheet {sid}: {len(items)} shelf/shelves')

        # Contour DXF (shelf outlines with bracket relief slots)
        doc_c = ezdxf.new(dxfversion='R2010')
        doc_c.units = units.IN
        msp_c = doc_c.modelspace()
        lyr_c = doc_c.layers.new(LAYER_CONTOUR)
        lyr_c.color = 3   # green

        # Pocket DXF (bracket tongue polygons only)
        doc_p = ezdxf.new(dxfversion='R2010')
        doc_p.units = units.IN
        msp_p = doc_p.modelspace()
        lyr_p = doc_p.layers.new(LAYER_POCKET)
        lyr_p.color = 5   # blue

        # Sheet boundary on both files
        sheet_rect = [(0,0,0),(SHEET_W,0,0),(SHEET_W,SHEET_H,0),(0,SHEET_H,0),(0,0,0)]
        msp_c.add_lwpolyline(sheet_rect, close=True,
                             dxfattribs={'layer': LAYER_CONTOUR})
        msp_p.add_lwpolyline(sheet_rect, close=True,
                             dxfattribs={'layer': LAYER_POCKET})

        for item in items:
            name = item['name']
            g = geom.get(name)
            if g is None:
                print(f'  WARNING: geometry not found for {name}, skipping.')
                continue
            process_shelf(item, g, msp_c, msp_p)

        out_contour = output_dir / f'sheet_{sid}_contours.dxf'
        out_pocket  = output_dir / f'sheet_{sid}_pockets.dxf'
        doc_c.saveas(out_contour)
        doc_p.saveas(out_pocket)
        print(f'  → {out_contour}')
        print(f'  → {out_pocket}')

    print('\nDone.')


if __name__ == '__main__':
    main()
