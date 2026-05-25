#!/usr/bin/env python3
"""
Generate office-shelf geometry from configs/office.json.

Outputs:
  output/shelf_<idx>.dxf       — one per shelf
  output/office_layout.pdf     — page per shelf at its level + vertical-supports overview

Status:
  ✓ Shelf straight-line vertices + dogbones
  ✓ Per-shelf extra-support widths from sinusoid
  ✓ Layout PDF (shelves with dihedral overlay)
  ⚠ Outer 3-arc curve approximated by a straight chord (geometry needs clarification)
  ☐ Main vertical supports (placeholder rectangles in PDF)
  ☐ Extra vertical supports (placeholder rectangles in PDF)
  ☐ Nesting / CNC stages
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import ezdxf
from ezdxf import units
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'lib'))
from dogbone import expand_polygon_with_dogbones, insert_proper_dogbone


def axis_aligned_arc(center, radius, start_angle_deg, sweep_deg, n=24):
    """Sample a circular arc. Angles in degrees, CCW positive (standard).
    For a CW (right-turning) sweep, pass negative sweep_deg."""
    cx, cy = center
    a0 = math.radians(start_angle_deg)
    a1 = math.radians(start_angle_deg + sweep_deg)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a))
            for a in np.linspace(a0, a1, n + 1)]


def tab_z_ranges(cfg, bottoms):
    """Return {'x': [(z1,z2)…], 'y': [(z1,z2)…]} for main-support interlock tabs.

    Between consecutive shelves there are 3 tabs (each 1/3 of the gap height).
    Pattern alternates each gap: gap 0,2,4,6 = (x,y,x); gap 1,3,5 = (y,x,y).
    """
    s = cfg['stock_thickness']
    x_tabs, y_tabs = [], []
    for i in range(len(bottoms) - 1):
        z0 = bottoms[i] + s
        z3 = bottoms[i + 1]
        third = (z3 - z0) / 3.0
        z1, z2 = z0 + third, z0 + 2 * third
        if i % 2 == 0:                    # xyx
            x_tabs.append((z0, z1))
            y_tabs.append((z1, z2))
            x_tabs.append((z2, z3))
        else:                             # yxy
            y_tabs.append((z0, z1))
            x_tabs.append((z1, z2))
            y_tabs.append((z2, z3))
    return {'x': x_tabs, 'y': y_tabs}


def main_support_specs(cfg, bottoms, width, my_tab_z_ranges, z_top):
    """Vertex spec list (for expand_polygon_with_dogbones) for one main support's
    flat layout. Layout coords: X across width, Z up.

      • Bottom edge   between the bottom shelf's open notches, if shelf 0 is at z=0
      • Right edge ↑   with shelf notches at every shelf z AND tab protrusions at this
                       support's own tab z ranges
      • Top edge      (width, z_top) → (0, z_top)
      • Left edge ↓   with shelf notches at every shelf z (no tabs on this side)

    Notches: 1.75" deep into the board, height s.
    Tabs:    s deep past the right edge, height = each tab's z range.
    """
    s  = cfg['stock_thickness']
    vs = cfg['vertical_stock_thickness']
    inset = cfg['shelf']['main_inset_step']      # 1.75"
    gt = cfg.get('glue_tolerance', 0.0)
    bottom_open = any(abs(h) < 1e-9 for h in bottoms)

    if bottom_open:
        specs = [{"xy": (inset, 0)}, {"xy": (width - inset, 0)}]
    else:
        specs = [{"xy": (0, 0)}, {"xy": (width, 0)}]

    # Right edge ↑ : merge shelf notches and tab protrusions, sort by z
    feats = ([('notch', h) for h in bottoms]
             + [('tab', z1, z2) for (z1, z2) in my_tab_z_ranges])
    feats.sort(key=lambda f: f[1])
    for f in feats:
        if f[0] == 'notch':
            h = f[1]
            if bottom_open and abs(h) < 1e-9:
                specs.append({"xy": (width - inset, h + s + gt), "dogbone": True})  # concave
                specs.append({"xy": (width,         h + s + gt)})                   # convex
            else:
                specs.append({"xy": (width,         h - gt)})                    # convex
                specs.append({"xy": (width - inset, h - gt), "dogbone": True})   # concave
                specs.append({"xy": (width - inset, h + s + gt), "dogbone": True})  # concave
                specs.append({"xy": (width,         h + s + gt)})                # convex
        else:
            _, z1, z2 = f
            specs.append({"xy": (width,      z1 + gt), "dogbone": True})   # base, concave
            specs.append({"xy": (width + vs, z1 + gt)})                    # outer, convex
            specs.append({"xy": (width + vs, z2 - gt)})                    # outer, convex
            specs.append({"xy": (width,      z2 - gt), "dogbone": True})   # base, concave

    # Top edge
    specs.append({"xy": (width, z_top)})
    specs.append({"xy": (0,     z_top)})

    # Left edge ↓ : shelf notches only, in reverse z order
    for h in sorted(bottoms, reverse=True):
        if bottom_open and abs(h) < 1e-9:
            specs.append({"xy": (0,     h + s + gt)})                          # convex
            specs.append({"xy": (inset, h + s + gt), "dogbone": True})         # concave
            specs.append({"xy": (inset, h          )})                          # open-bottom notch wall
        else:
            specs.append({"xy": (0,     h + s + gt)})                          # convex
            specs.append({"xy": (inset, h + s + gt), "dogbone": True})         # concave
            specs.append({"xy": (inset, h - gt     ), "dogbone": True})        # concave
            specs.append({"xy": (0,     h - gt     )})                          # convex

    # Dedupe consecutive specs with identical xy (happens when a shelf sits
    # exactly at a board corner — e.g. shelf 0 at z=0 produces (width, 0)
    # both as the bottom-edge endpoint AND the first notch vertex).
    deduped = []
    for sp in specs:
        if deduped and 'xy' in sp and 'xy' in deduped[-1] \
                and tuple(sp['xy']) == tuple(deduped[-1]['xy']):
            if sp.get('dogbone') or deduped[-1].get('dogbone'):
                deduped[-1] = {'xy': sp['xy'], 'dogbone': True}
            continue
        deduped.append(sp)
    # Also dedupe the wrap-around (last vs first).
    if len(deduped) > 1 and 'xy' in deduped[0] and 'xy' in deduped[-1] \
            and tuple(deduped[0]['xy']) == tuple(deduped[-1]['xy']):
        deduped.pop()
    return deduped


def extra_support_points(cfg, bottoms, side, z_top, n_sin_samples=600):
    """Closed polygon (list of (x, z)) for one extra-support layout, going CCW.

    Layout: outside edge at X = 0 (against the wall); inside edge at X = width(z),
    where width(z) is the sinusoid except inside each shelf z-range [h, h+s] where
    it's flattened to width(h+s). Dog-bone reliefs are inserted at flat-to-sinusoid
    corners that are CONCAVE (where flat is stepped inward of sinusoid); convex
    corners get no relief.
    """
    es      = cfg['extra_supports']
    s       = cfg['stock_thickness']
    R       = cfg['dogbone_radius']
    nominal = es['nominal_width']
    amp     = es['sinusoid_amplitude']
    period  = es['sinusoid_period']
    phase   = es['right_phase_radians'] if side == 'right' else es['left_phase_radians']

    def w_sin(z):
        return nominal + amp * math.sin(2 * math.pi * z / period + phase)

    flats = [(h, h + s, w_sin(h + s)) for h in bottoms]   # (z_bot, z_top, w_flat)

    def in_flat(z):
        for z_b, z_t, w_f in flats:
            if z_b - 1e-9 <= z <= z_t + 1e-9:
                return w_f
        return None

    # Sample the inside edge from z=0 to z=z_top, ALWAYS hitting every shelf
    # corner exactly so we can locate them later.
    forced_zs = sorted({0.0, z_top, *(z for f in flats for z in (f[0], f[1]))})
    sample_zs = sorted(set(np.linspace(0, z_top, n_sin_samples)).union(forced_zs))

    edge = []
    for z in sample_zs:
        f = in_flat(z)
        edge.append((f if f is not None else w_sin(z), z))

    # Drop consecutive duplicates that creep in at boundary samples.
    cleaned = [edge[0]]
    for p in edge[1:]:
        if abs(p[0] - cleaned[-1][0]) > 1e-9 or abs(p[1] - cleaned[-1][1]) > 1e-9:
            cleaned.append(p)
    edge = cleaned

    # Identify flat-corner indices (exact-match z to a shelf boundary).
    flat_corner_idxs = []
    for i, (x, z) in enumerate(edge):
        for z_b, z_t, w_f in flats:
            if (abs(z - z_b) < 1e-9 or abs(z - z_t) < 1e-9) and abs(x - w_f) < 1e-9:
                flat_corner_idxs.append((i, z_b, z_t, w_f))

    # Assemble closed CCW polygon: bottom edge → inside edge ↑ → top edge → outside edge ↓.
    poly = [(0.0, 0.0), edge[0]]
    poly.extend(edge[1:])
    poly.append((0.0, z_top))

    # Polygon orientation: traversal is CCW, interior is INSIDE the board (low X).
    # Insert dog-bones at concave flat corners. Use insert_proper_dogbone, which
    # finds the actual intersections of the dogbone circle with the adjacent
    # polygon edges and splices in only the arc between them.
    inside_idxs_in_poly = [
        (i + 1, z_b, z_t, w_f) for i, z_b, z_t, w_f in flat_corner_idxs
    ]   # +1 for prepended (0, 0)

    # We re-locate corners by their (x, z) value after each insertion (indices
    # shift). This is robust to polygon-length changes from arc splicing.
    target_corners = []
    for idx, z_b, z_t, w_f in inside_idxs_in_poly:
        pp = poly[idx - 1]
        cp = poly[idx]
        np_ = poly[(idx + 1) % len(poly)]
        in_dx, in_dy = cp[0] - pp[0], cp[1] - pp[1]
        out_dx, out_dy = np_[0] - cp[0], np_[1] - cp[1]
        cross = in_dx * out_dy - in_dy * out_dx
        if abs(cp[1] - z_b) < 1e-9:
            adjacent_width = w_sin(z_b)
        else:
            adjacent_width = w_sin(z_t)
        inward_step = adjacent_width - w_f
        if cross < 0 and inward_step > 1e-6:            # concave, with real inset
            target_corners.append(tuple(cp))

    for tgt in target_corners:
        # Find current index of this corner (may have shifted after earlier inserts)
        cur_idx = next((i for i, p in enumerate(poly)
                        if abs(p[0] - tgt[0]) < 1e-9 and abs(p[1] - tgt[1]) < 1e-9), None)
        if cur_idx is None:
            continue
        poly = insert_proper_dogbone(poly, cur_idx, R, n_arc_pts=14)

    return poly


def outer_curve_arcs(cfg, right_arm_ext=0.0):
    """Three-arc 'smooth front' of the shelf, returned as a single point list.

    Topology (S-curve): arc A (convex, right-turn) → straight ↓ → arc B (concave,
    left-turn) → straight → → arc C (convex, right-turn) → straight ↓ to vertex 1.

    Centers:
      A: (cx_A,             bly + dy − r)
         cx_A = dx + 1 − 2r   ← DERIVED for axis-aligned closure with B's x.
                                (User originally wrote `mx − r + 1` = 8.625;
                                actual closure value is 4.3125 at r=4. Differs by `mx − dx + 2r − 2`.
                                If you want A at 8.625, B's x must shift to match — we'll
                                surface this if you spot it in the PDF.)
      B: (dx + r + 1,       dy + r + 1)     ← user-tuned (cx shifted right by r for style);
                                              bottom tangent (cx_B, dy + 1).
                                              r=4 → center (16.3125, 12.75)
      C: (dx + mx + rx + right_arm_ext − r, dy − r + 1)  ← shifts right with extension;
                                              r=4, no ext → (28.0625, 4.75)
    """
    s   = cfg['stock_thickness']
    dx  = cfg['dihedral']['x_extent']
    dy  = cfg['dihedral']['y_extent']
    mx  = cfg['shelf']['main_x_inset']
    rx  = cfg['shelf']['right_outer_x_offset']
    bly = cfg['shelf']['back_y_inset']
    r   = cfg['shelf']['outer_arc_radius']

    cx_B, cy_B = dx + r + 1, dy + r + 1            # bottom tangent at (cx_B, dy + 1)
    cx_C, cy_C = dx + mx + rx + right_arm_ext - r, dy - r + 1
    # Closure-derived: cx_A + r = cx_B − r, so cx_A = cx_B − 2r.
    cx_A = cx_B - 2 * r
    cy_A = bly + dy - r

    pts = []
    # Straight +x along the back: from vertex 17 (s, bly+dy) to arc A's top tangent.
    pts.append((cx_A, bly + dy))                                    # arc A entry (top)
    # Arc A: right turn (CW), 90°, from top (angle 90°) to right (angle 0°).
    pts.extend(axis_aligned_arc((cx_A, cy_A), r, 90, -90)[1:])      # exit at right
    # Straight −y from arc A right tangent to arc B left tangent
    pts.append((cx_B - r, cy_B))                                    # arc B entry (left)
    # Arc B: left turn (CCW), 90°, from left (angle 180°) to bottom (angle 270°).
    pts.extend(axis_aligned_arc((cx_B, cy_B), r, 180, 90)[1:])      # exit at bottom
    # Straight +x from arc B bottom to arc C top tangent.
    pts.append((cx_C, cy_C + r))                                    # arc C entry (top)
    # Arc C: right turn (CW), 90°, from top (angle 90°) to right (angle 0°).
    pts.extend(axis_aligned_arc((cx_C, cy_C), r, 90, -90)[1:])      # exit at right
    # Final point on the arc segment is at (cx_C + r, cy_C).
    # Polygon then closes from this point straight −y to vertex 1.
    return pts


# ── Config / derived ─────────────────────────────────────────────────────────

def load_config():
    p = Path(__file__).parent.parent / 'configs' / 'office.json'
    with open(p) as f:
        return json.load(f)


def shelf_bottoms(cfg):
    """Bottom-y for each shelf, derived from the height rule in config."""
    sh = cfg['shelves']
    s = cfg['stock_thickness']
    heights = [sh['first_height']]
    if sh['count'] >= 2:
        heights.append(sh['second_height_factor'] * sh['subsequent_separation'])
    for _ in range(sh['count'] - 2):
        heights.append(heights[-1] + s + sh['subsequent_separation'])
    return heights


def extra_support_width(cfg, y, side):
    """Width of an extra support at height y. side ∈ {'right','left'}."""
    es = cfg['extra_supports']
    phase = es['right_phase_radians'] if side == 'right' else es['left_phase_radians']
    return es['nominal_width'] + es['sinusoid_amplitude'] * math.sin(
        2 * math.pi * y / es['sinusoid_period'] + phase)


# ── Shelf construction ──────────────────────────────────────────────────────

def shelf_polygon(cfg, shelf_bottom):
    """Build closed polygon (list of (x,y)) for one shelf at the given bottom-y.

    Coordinate system: corner (back-left of room) at origin (0,0).
    The shelf's relevant geometry is in 2D plan view at height = shelf_bottom.

    If right_arm_extension is configured and shelf_top < height_limit, the right
    arm is extended past the right_extra support by `length` inches.  The slot for
    right_extra becomes a closed 3-walled slot (open only at y=0), requiring
    dogbones at all three inner corners.  Arc C shifts rightward to match the
    new arm end so the shelf terminates with the same curved profile.
    """
    s   = cfg['stock_thickness']
    vs  = cfg['vertical_stock_thickness']
    R   = cfg['dogbone_radius']
    gt  = cfg.get('glue_tolerance', 0.0)
    dx  = cfg['dihedral']['x_extent']      # 11.3125
    dy  = cfg['dihedral']['y_extent']      # 7.75
    mx  = cfg['shelf']['main_x_inset']     # 11.625
    rx  = cfg['shelf']['right_outer_x_offset']  # 9.125
    bly = cfg['shelf']['back_y_inset']     # 11.875
    mi  = cfg['shelf']['main_inset_step']  # 1.75
    r   = cfg['shelf']['outer_arc_radius']

    # Extra-support widths AT THE TOP of the shelf (per user spec).
    y_top = shelf_bottom + s
    vsr = extra_support_width(cfg, y_top, 'right')
    vsl = extra_support_width(cfg, y_top, 'left')

    right_x = dx + mx + rx   # 32.0625 — unextended far-right edge

    # Check whether to extend the right arm.
    ext_cfg    = cfg.get('right_arm_extension', {})
    ext_len    = ext_cfg.get('length', 0.0)
    height_lim = ext_cfg.get('height_limit', 27.5)
    do_extend  = ext_len > 0 and y_top < height_lim

    # Right-arm start vertices.
    # Non-extended: one open corner notch for right_extra support.
    # Extended: two open corner notches — far-end support slot at arm tip, then right_extra slot.
    # Both notches are open on the right face and wall face (y=0); same slot width vsr
    # since far_right_extra uses the same sinusoidal profile as right_extra.
    if do_extend:
        right_specs = [
            {"xy": (right_x + ext_len,                  vs)},                   # A: new arm end-top (convex)
            {"xy": (right_x + ext_len - vsr - 2*gt,     vs), "dogbone": True},  # B: far-end slot top-left (concave)
            {"xy": (right_x + ext_len - vsr - 2*gt,     0)},                    # C: far-end slot bottom-left (convex)
            {"xy": (right_x,                             0),  "dogbone": True},  # D: right_extra slot right-bottom (concave)
            {"xy": (right_x,                             vs), "dogbone": True},  # E: right_extra slot right-top (concave)
            {"xy": (right_x - vsr - 2*gt,               vs), "dogbone": True},  # F: right_extra slot left-top (concave)
            {"xy": (right_x - vsr - 2*gt,               0)},                    # G: right_extra slot left-bottom (convex)
        ]
    else:
        right_specs = [
            {"xy": (right_x,                vs)},                   # V1: arm end (convex)
            {"xy": (right_x - vsr - 2*gt,   vs), "dogbone": True},  # V2: slot top-left (concave)
            {"xy": (right_x - vsr - 2*gt,   0)},                    # V3: slot bottom-left (convex)
        ]

    # Common body: dihedral notch, back arm, left arm (unchanged), outer arc.
    body_specs = [
        {"xy": (dx,                            0)},
        {"xy": (dx,                            mi)},
        {"xy": (dx + vs + 2*gt,               mi),       "dogbone": True},   # x_main slot: widen right wall
        {"xy": (dx + vs + 2*gt,               dy - mi),  "dogbone": True},   # x_main slot: widen right wall
        {"xy": (dx,                            dy - mi)},
        {"xy": (dx,                            dy),       "dogbone": True},
        {"xy": (dx - mi,                       dy)},
        {"xy": (dx - mi,                       dy + vs + 2*gt), "dogbone": True},  # y_main slot: widen top wall
        {"xy": (mi,                            dy + vs + 2*gt), "dogbone": True},  # y_main slot: widen top wall
        {"xy": (mi,                            dy)},
        {"xy": (0,                             dy)},
        {"xy": (0,                             dy + bly - vsl - 2*gt)},                    # left_extra slot: widen
        {"xy": (vs,                            dy + bly - vsl - 2*gt), "dogbone": True},   # left_extra slot: widen
        {"xy": (vs,                            dy + bly)},   # left arm end (convex, no dogbone)
        {"arc_points": outer_curve_arcs(cfg, right_arm_ext=ext_len if do_extend else 0.0)},
    ]

    pts = expand_polygon_with_dogbones(right_specs + body_specs, R, n_arc_pts=18)
    return pts


# ── DXF output ──────────────────────────────────────────────────────────────

def write_shelf_dxf(pts, path):
    doc = ezdxf.new(dxfversion='R2010')
    doc.units = units.IN
    msp = doc.modelspace()
    closed_pts = list(pts) + [pts[0]]
    msp.add_lwpolyline([(x, y, 0) for x, y in closed_pts], close=True,
                       dxfattribs={'layer': 'SHELF'})
    doc.saveas(path)


# ── PDF layout ──────────────────────────────────────────────────────────────

def draw_dihedral_outline(ax, dx, dy):
    """The room corner with the dihedral protrusion (rectangular ‘bump’ into the room)."""
    # The two walls meet at (dx, 0)→(dx, dy)→(0, dy) (the dihedral outline as seen from above).
    # Draw the wall lines extended beyond the dihedral so the geometry context is visible.
    wall_color = '#333333'
    ax.plot([dx, dx], [0, dy], color=wall_color, lw=2)            # right face of dihedral
    ax.plot([0, dx],  [dy, dy], color=wall_color, lw=2)           # top face of dihedral
    # Extend room walls outward from the dihedral edges:
    ax.plot([dx, 60], [0, 0],   color=wall_color, lw=1, ls='--')  # rest of bottom wall
    ax.plot([0, 0],  [dy, 100], color=wall_color, lw=1, ls='--')  # rest of left wall


def y_main_polygon(cfg, bottoms, tabs, z_top):
    width = cfg['dihedral']['y_extent']
    specs = main_support_specs(cfg, bottoms, width, tabs['y'], z_top)
    return expand_polygon_with_dogbones(specs, cfg['dogbone_radius'], n_arc_pts=14)


def x_main_polygon(cfg, bottoms, tabs, z_top):
    width = cfg['dihedral']['x_extent']
    specs = main_support_specs(cfg, bottoms, width, tabs['x'], z_top)
    return expand_polygon_with_dogbones(specs, cfg['dogbone_radius'], n_arc_pts=14)


def make_layout_pdf(cfg, shelves, verticals, pdf_path):
    dx = cfg['dihedral']['x_extent']
    dy = cfg['dihedral']['y_extent']
    bottoms = shelf_bottoms(cfg)
    shelf_color = '#4A90E2'
    dogbone_color = '#E24A4A'

    with PdfPages(pdf_path) as pdf:
        # One page per shelf at its level
        for idx, (h, pts) in enumerate(zip(bottoms, shelves)):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            draw_dihedral_outline(ax, dx, dy)
            poly = mpatches.Polygon(pts, closed=True, facecolor=shelf_color,
                                    edgecolor='black', linewidth=1.0, alpha=0.55)
            ax.add_patch(poly)
            # Mark the start vertex
            ax.plot(pts[0][0], pts[0][1], marker='o', color='black', markersize=4)

            ax.set_title(f"Shelf {idx} — bottom at y(height)={h:.3f}\"  "
                         f"(top={h + cfg['stock_thickness']:.3f}\")",
                         fontsize=12, fontweight='bold')
            ax.set_xlabel("x (inches)"); ax.set_ylabel("y (inches)")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            x_max = max(p[0] for p in pts)
            ax.set_xlim(-3, x_max + 4)
            ax.set_ylim(-3, 24)

            # Annotate the per-shelf extra-support widths
            y_top = h + cfg['stock_thickness']
            vsr = extra_support_width(cfg, y_top, 'right')
            vsl = extra_support_width(cfg, y_top, 'left')
            ax.text(0.02, 0.98,
                    f"extra-support width @ shelf top:\n  right = {vsr:.4f}\"\n  left  = {vsl:.4f}\"",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85))
            ax.legend(handles=[
                mpatches.Patch(facecolor=shelf_color, edgecolor='black', alpha=0.55,
                               label=f'Shelf {idx}'),
                Line2D([0],[0], color='#333333', lw=2, label='Dihedral wall face'),
                Line2D([0],[0], color='#333333', lw=1, ls='--', label='Room wall (extends)'),
            ], loc='upper right', fontsize=9)
            pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Verticals overview page ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(17, 11))
        # Lay the four verticals side-by-side along x with a fixed gap.
        gap = 6.0
        x_cursor = 0.0
        labels = []
        colors = {'y_main': '#5B8FB9', 'x_main': '#7E9D5C',
                  'right_extra': '#C97064', 'left_extra': '#C97064',
                  'far_right_extra': '#C97064'}
        names = [n for n in ('y_main', 'x_main', 'right_extra', 'left_extra', 'far_right_extra')
                 if n in verticals]
        for name in names:
            pts = verticals[name]
            xs = [p[0] for p in pts]
            shifted = [(p[0] + x_cursor - min(xs), p[1]) for p in pts]
            poly = mpatches.Polygon(shifted, closed=True,
                                    facecolor=colors[name], edgecolor='black',
                                    linewidth=0.6, alpha=0.55)
            ax.add_patch(poly)
            xmin = min(p[0] for p in shifted); xmax = max(p[0] for p in shifted)
            labels.append((name, (xmin + xmax) / 2))
            x_cursor = xmax + gap

        for name, mid_x in labels:
            ax.text(mid_x, -3, name.replace('_', '-'),
                    ha='center', va='top', fontsize=11, fontweight='bold')

        # Annotate each shelf level with a faint dashed horizontal line across all four
        for i, h in enumerate(bottoms):
            ax.axhline(h, color='gray', linewidth=0.4, linestyle=':', alpha=0.6)
            ax.text(-2, h, f"S{i}", ha='right', va='center', fontsize=7, color='gray')

        ax.set_title("Office vertical supports — flat layouts", fontsize=14, fontweight='bold')
        ax.set_xlabel("x (inches, layout)"); ax.set_ylabel("z (inches)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlim(-4, x_cursor + 2)
        ax.set_ylim(-6, max(p[1] for p in verticals['y_main']) + 4)
        pdf.savefig(fig, dpi=150); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    out_dir = Path(__file__).parent.parent / 'output'
    out_dir.mkdir(exist_ok=True)

    bottoms = shelf_bottoms(cfg)
    s = cfg['stock_thickness']
    z_top = bottoms[-1] + s + 1.0     # 1" above the top shelf notch

    print(f"Generating {len(bottoms)} shelves at bottoms: "
          + ", ".join(f"{h:.3f}" for h in bottoms))

    # ── Shelves ───────────────────────────────────────────────────────────
    ext_cfg    = cfg.get('right_arm_extension', {})
    ext_len    = ext_cfg.get('length', 0.0)
    height_lim = ext_cfg.get('height_limit', 27.5)

    shelves = []
    for idx, h in enumerate(bottoms):
        y_top     = h + s
        do_extend = ext_len > 0 and y_top < height_lim
        pts = shelf_polygon(cfg, h)
        shelves.append(pts)
        write_shelf_dxf(pts, out_dir / f"shelf_{idx}.dxf")
        ext_note = f"  +{ext_len}\" right-arm extension" if do_extend else ""
        print(f"  shelf {idx} (bottom={h:.3f}\" top={y_top:.3f}\"): {len(pts)} pts{ext_note}")

    # ── Verticals ─────────────────────────────────────────────────────────
    tabs = tab_z_ranges(cfg, bottoms)
    print(f"\nInterlock tabs: {len(tabs['x'])} x-main, {len(tabs['y'])} y-main")
    verticals = {
        'y_main':      y_main_polygon(cfg, bottoms, tabs, z_top),
        'x_main':      x_main_polygon(cfg, bottoms, tabs, z_top),
        'right_extra': extra_support_points(cfg, bottoms, 'right', z_top),
        'left_extra':  extra_support_points(cfg, bottoms, 'left',  z_top),
    }

    # far_right_extra: sinusoidal support flush with the end of extended shelves,
    # height = top of the highest extended shelf.
    ext_bottoms = [b for b in bottoms if b + s < height_lim]
    if ext_len > 0 and ext_bottoms:
        far_z_top = ext_bottoms[-1] + s
        verticals['far_right_extra'] = extra_support_points(cfg, ext_bottoms, 'right', far_z_top)
        print(f"  far_right_extra: spans {len(ext_bottoms)} shelves, z_top={far_z_top:.3f}\"")

    for name, pts in verticals.items():
        write_shelf_dxf(pts, out_dir / f"{name}.dxf")
        print(f"  {name}: {len(pts)} pts")

    pdf_path = out_dir / "office_layout.pdf"
    make_layout_pdf(cfg, shelves, verticals, pdf_path)
    print(f"\n✓ Layout PDF: {pdf_path}")
    print(f"  ({len(bottoms)} shelf pages + 1 verticals overview)")


if __name__ == '__main__':
    main()
