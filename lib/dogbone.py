"""
Dogbone arc helpers for axis-aligned 90° polygon corners.

A "dogbone" is a 180° arc that REPLACES a sharp corner so a finite-radius
endmill can fully clear it. The arc passes through the original corner
point and bulges past it on the diagonal opposite to the polygon interior.

Geometry (from pantry's `generate_shelves_with_brackets.create_bracket_outline`):

    For a 90° corner C with edges going to prev_pt P and next_pt N
    (back = (P-C)/|P-C|, fwd = (N-C)/|N-C|),

        center  = C + (R/sqrt(2)) * (back + fwd)        # |back+fwd| = sqrt(2)
        tangent_in  = C + R*sqrt(2) * back              # on incoming edge
        tangent_out = C + R*sqrt(2) * fwd               # on outgoing edge

    The arc sweeps 180° from tangent_in to tangent_out, passing through C.
"""

import math
import numpy as np


def axial_dogbone_arc(prev_pt, corner, next_pt, radius, n_arc_pts=24):
    """Replace `corner` with a 180° dogbone arc.

    Returns a list of (x, y) tuples STARTING at the tangent-in point on the
    incoming edge, sweeping through the corner, and ENDING at the tangent-out
    point on the outgoing edge. The list does NOT include `corner` itself —
    splice this list into the polygon in place of the corner vertex.

    Assumes the corner is 90° and edges are axis-aligned (no validation).
    """
    cx, cy = corner
    px, py = prev_pt
    nx, ny = next_pt

    back_x, back_y = px - cx, py - cy
    bl = math.hypot(back_x, back_y)
    back_x, back_y = back_x / bl, back_y / bl

    fwd_x, fwd_y = nx - cx, ny - cy
    fl = math.hypot(fwd_x, fwd_y)
    fwd_x, fwd_y = fwd_x / fl, fwd_y / fl

    R = radius
    sqrt2 = math.sqrt(2)

    cxc = cx + (R / sqrt2) * (back_x + fwd_x)
    cyc = cy + (R / sqrt2) * (back_y + fwd_y)

    t1x = cx + R * sqrt2 * back_x
    t1y = cy + R * sqrt2 * back_y
    t2x = cx + R * sqrt2 * fwd_x
    t2y = cy + R * sqrt2 * fwd_y

    a1 = math.atan2(t1y - cyc, t1x - cxc)
    a2 = math.atan2(t2y - cyc, t2x - cxc)
    a_corner = math.atan2(cy - cyc, cx - cxc)

    # Sweep direction: pick the 180° arc that passes through the corner angle.
    # CCW first.
    a2_ccw = a2 + 2 * math.pi if a2 < a1 else a2
    ac_ccw = a_corner
    while ac_ccw < a1: ac_ccw += 2 * math.pi
    if ac_ccw <= a2_ccw + 1e-9:
        return [(cxc + R * math.cos(a), cyc + R * math.sin(a))
                for a in np.linspace(a1, a2_ccw, n_arc_pts + 1)]
    # Otherwise CW.
    a2_cw = a2 - 2 * math.pi if a2 > a1 else a2
    return [(cxc + R * math.cos(a), cyc + R * math.sin(a))
            for a in np.linspace(a1, a2_cw, n_arc_pts + 1)]


def general_dogbone_arc(prev_pt, corner, next_pt, radius, n_arc_pts=18):
    """Dogbone for a non-axis-aligned corner. Same construction as the axial
    version (180° arc through the corner, bulging in (back+fwd) direction),
    but doesn't assume a 90° angle — works for any concave corner ≥ 30° or so.

    Returns a list of (x, y) tuples replacing the corner; empty list if the
    geometry is degenerate.
    """
    cx, cy = corner
    px, py = prev_pt
    nx, ny = next_pt

    bx, by = px - cx, py - cy
    bl = math.hypot(bx, by)
    if bl < 1e-12: return []
    bx, by = bx / bl, by / bl

    fx, fy = nx - cx, ny - cy
    fl = math.hypot(fx, fy)
    if fl < 1e-12: return []
    fx, fy = fx / fl, fy / fl

    # Bulge direction = unit vector along (back + fwd)
    sx, sy = bx + fx, by + fy
    sl = math.hypot(sx, sy)
    if sl < 1e-9: return []   # near-180° corner, nothing to dogbone
    sx, sy = sx / sl, sy / sl

    R = radius
    # Center placed along bulge direction at distance R from corner so the arc
    # passes through corner. Tangent points are along back and fwd directions
    # at distance R*sqrt(2) from corner only for 90° corners; for other angles
    # we just sweep a 180° arc and trim if needed.
    cxc = cx + R * sx
    cyc = cy + R * sy

    # Sweep 180° centered on the angle pointing from center to corner.
    a_corner = math.atan2(cy - cyc, cx - cxc)
    a0 = a_corner - math.pi / 2
    a1 = a_corner + math.pi / 2
    return [(cxc + R * math.cos(a), cyc + R * math.sin(a))
            for a in np.linspace(a0, a1, n_arc_pts + 1)]


def expand_polygon_with_dogbones(specs, radius, n_arc_pts=24, collinear_tol=1e-6):
    """Expand a vertex spec list into a closed polygon with dogbones inserted.

    Each entry in `specs` is one of:
      {"xy": (x, y)}                     # sharp vertex
      {"xy": (x, y), "dogbone": True}    # vertex to be replaced by a dogbone arc
      {"arc_points": [(x, y), ...]}      # already-computed arc segment to splice in

    Returns a flat list of (x, y) tuples. The polygon is implicitly closed
    by the caller.

    Vertices flagged for a dogbone but whose incoming and outgoing edges are
    actually collinear (cross product < `collinear_tol`) are passed through
    sharp — this happens when two features dedup against each other, leaving
    a "corner" that has no real angle.
    """
    n = len(specs)
    out = []
    for i, spec in enumerate(specs):
        if "arc_points" in spec:
            out.extend(spec["arc_points"])
            continue
        if not spec.get("dogbone"):
            out.append(tuple(spec["xy"]))
            continue
        prev_xy = _neighbor_xy(specs, i - 1)
        next_xy = _neighbor_xy(specs, i + 1)
        corner  = tuple(spec["xy"])
        in_dx = corner[0] - prev_xy[0]; in_dy = corner[1] - prev_xy[1]
        out_dx = next_xy[0] - corner[0]; out_dy = next_xy[1] - corner[1]
        cross = in_dx * out_dy - in_dy * out_dx
        if abs(cross) < collinear_tol:           # not actually a corner — flat
            out.append(corner)
            continue
        out.extend(axial_dogbone_arc(prev_xy, corner, next_xy,
                                     radius, n_arc_pts=n_arc_pts))
    return out


def insert_proper_dogbone(poly, corner_idx, radius, n_arc_pts=20):
    """Replace `poly[corner_idx]` and surrounding points with a dogbone arc that
    actually intersects the adjacent polygon edges.

    Construction:
      1. Place the dogbone center at corner + R * unit(back + fwd).
      2. Walk backward and forward from corner_idx along `poly` until the
         polyline first exits the circle. That gives two segment-circle
         intersection points.
      3. Build a circular arc connecting those two intersections, choosing the
         sweep direction that passes near the corner (the side facing the
         polygon material).
      4. Splice the arc into the polygon, replacing all points strictly between
         the two outside-poly endpoints.

    Returns a new list of points (or the original list unchanged if the
    construction is degenerate).
    """
    n = len(poly)
    cx, cy = poly[corner_idx]
    prev_pt = poly[(corner_idx - 1) % n]
    next_pt = poly[(corner_idx + 1) % n]

    bx = prev_pt[0] - cx; by = prev_pt[1] - cy
    bl = math.hypot(bx, by)
    fx = next_pt[0] - cx; fy = next_pt[1] - cy
    fl = math.hypot(fx, fy)
    if bl < 1e-12 or fl < 1e-12:
        return poly
    bx /= bl; by /= bl; fx /= fl; fy /= fl

    sx = bx + fx; sy = by + fy
    sl = math.hypot(sx, sy)
    if sl < 1e-9:
        return poly
    sx /= sl; sy /= sl

    R = radius
    cxc = cx + R * sx
    cyc = cy + R * sy

    def walk(step):
        """Walk poly indices by `step` until first point outside the circle.
        Returns (intersection_pt, outside_idx) or (None, None)."""
        prev = (cx, cy)
        i = corner_idx
        for _ in range(n):
            i = (i + step) % n
            pt = poly[i]
            d = math.hypot(pt[0] - cxc, pt[1] - cyc)
            if d > R + 1e-9:
                ax, ay = prev
                vx, vy = pt[0] - ax, pt[1] - ay
                fxr = ax - cxc; fyr = ay - cyc
                aa = vx * vx + vy * vy
                bb = 2 * (fxr * vx + fyr * vy)
                cc = fxr * fxr + fyr * fyr - R * R
                disc = bb * bb - 4 * aa * cc
                if disc < 0:
                    return None, None
                sq = math.sqrt(disc)
                # Pick the t in [0, 1] (the intersection on this segment)
                for t in ((-bb - sq) / (2 * aa), (-bb + sq) / (2 * aa)):
                    if -1e-9 <= t <= 1 + 1e-9:
                        return (ax + t * vx, ay + t * vy), i
                return None, None
            prev = pt
        return None, None

    back_int, back_idx = walk(-1)
    fwd_int,  fwd_idx  = walk(+1)
    if back_int is None or fwd_int is None:
        return poly

    a_back = math.atan2(back_int[1] - cyc, back_int[0] - cxc)
    a_fwd  = math.atan2(fwd_int[1]  - cyc, fwd_int[0]  - cxc)
    a_corner = math.atan2(cy - cyc, cx - cxc)

    sweep_ccw = (a_fwd - a_back) % (2 * math.pi)
    diff_corner_ccw = (a_corner - a_back) % (2 * math.pi)
    use_ccw = diff_corner_ccw < sweep_ccw + 1e-9
    sweep = sweep_ccw if use_ccw else sweep_ccw - 2 * math.pi

    arc = [(cxc + R * math.cos(a), cyc + R * math.sin(a))
           for a in np.linspace(a_back, a_back + sweep, n_arc_pts + 1)]

    # Splice — handles only the non-wrap case (corner not near polygon ends).
    # back_idx is the first outside index walking backward (so smaller than corner_idx
    # in the non-wrap case); fwd_idx is the first outside index walking forward.
    if back_idx < corner_idx < fwd_idx:
        return poly[: back_idx + 1] + arc + poly[fwd_idx:]
    return poly


def _neighbor_xy(specs, j):
    """Return a representative (x, y) for spec `j` (mod len), used as a
    direction reference when building the dogbone at an adjacent vertex.
    For arc_points specs, use the closest endpoint to the requested side."""
    n = len(specs)
    spec = specs[j % n]
    if "xy" in spec:
        return tuple(spec["xy"])
    pts = spec["arc_points"]
    return tuple(pts[-1] if (j % n) < ((j + 1) % n) else pts[0])
