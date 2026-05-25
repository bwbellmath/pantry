#!/usr/bin/env python3
"""
Render the office shelf assembly in 3D.

Reads DXF files from output/, maps each piece into world space, then
invokes Blender (via lib/blender_render_furniture.py) to produce PNG renders.

Piece coordinate conventions
─────────────────────────────
All coords are in inches.  The room corner is at world origin (0,0,0).
The dihedral bump sits at x=[0,dx], y=[0,dy] in plan view.

  shelves      DXF (px,py)  → world (px,       py,       shelf_bottom)
               extruded +Z by stock_thickness

  x_main       DXF (px,py)  → world (px,       dy,       py)
               extruded +Y by vertical_stock_thickness
               (board spans x=0..x_extent along the top face of the dihedral)

  y_main       DXF (px,py)  → world (dx,       px,       py)
               extruded +X by vertical_stock_thickness
               (board spans y=0..y_extent along the right face of the dihedral)

  right_extra  DXF (px,py)  → world (right_x-px, 0,      py)
               extruded +Y by vertical_stock_thickness
               (sinusoidal brace at far-right end of the shelf unit,
                outside edge at x=right_x against the room right corner)

  left_extra   DXF (px,py)  → world (0,  back_y-px,      py)
               extruded +X by vertical_stock_thickness
               (sinusoidal brace at far-back end of the left arm,
                outside edge at y=back_y against the room back corner)
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'lib'))

from blender_renderer import read_dxf_polygon, build_furniture_geometry, run_blender_render

PROJECT   = Path(__file__).parent.parent
CFG_PATH  = PROJECT / 'configs' / 'office.json'
OUT_DIR   = PROJECT / 'output'
DXF_DIR   = OUT_DIR


def load_config():
    with open(CFG_PATH) as f:
        return json.load(f)


def shelf_bottoms(cfg):
    sh = cfg['shelves']
    s  = cfg['stock_thickness']
    heights = [sh['first_height']]
    if sh['count'] >= 2:
        heights.append(sh['second_height_factor'] * sh['subsequent_separation'])
    for _ in range(sh['count'] - 2):
        heights.append(heights[-1] + s + sh['subsequent_separation'])
    return heights


def build_pieces(cfg):
    s   = cfg['stock_thickness']
    vs  = cfg['vertical_stock_thickness']
    dx  = cfg['dihedral']['x_extent']      # 11.3125
    dy  = cfg['dihedral']['y_extent']      # 7.75
    mx  = cfg['shelf']['main_x_inset']     # 11.625
    rx  = cfg['shelf']['right_outer_x_offset']  # 9.125
    bly = cfg['shelf']['back_y_inset']     # 11.875

    ext_len = cfg.get('right_arm_extension', {}).get('length', 0.0)

    right_x    = dx + mx + rx         # 32.0625 — far-right edge of unit
    far_right_x = right_x + ext_len   # 50.0625 — end of extended shelves
    back_y     = dy + bly             # 19.625  — far-back edge of left arm

    bottoms = shelf_bottoms(cfg)
    pieces  = []

    # ── Shelves ──────────────────────────────────────────────────────────
    for idx, bottom in enumerate(bottoms):
        poly = read_dxf_polygon(DXF_DIR / f'shelf_{idx}.dxf')
        pieces.append({
            'name':      f'shelf_{idx}',
            'polygon':   poly,
            'origin':    [0.0, 0.0, bottom],
            'u_axis':    [1.0, 0.0, 0.0],
            'v_axis':    [0.0, 1.0, 0.0],
            'normal':    [0.0, 0.0, 1.0],
            'thickness': s,
            'material':  'plywood',
        })

    # ── x_main vertical support ───────────────────────────────────────────
    # Spans x=0..x_extent along the top (y=dy) face of the dihedral.
    poly = read_dxf_polygon(DXF_DIR / 'x_main.dxf')
    pieces.append({
        'name':      'x_main',
        'polygon':   poly,
        'origin':    [0.0, dy, 0.0],
        'u_axis':    [1.0, 0.0, 0.0],   # DXF x → world X
        'v_axis':    [0.0, 0.0, 1.0],   # DXF y → world Z
        'normal':    [0.0, 1.0, 0.0],   # extrude in +Y
        'thickness': vs,
        'material':  'walnut',
    })

    # ── y_main vertical support ───────────────────────────────────────────
    # Spans y=0..y_extent along the right (x=dx) face of the dihedral.
    poly = read_dxf_polygon(DXF_DIR / 'y_main.dxf')
    pieces.append({
        'name':      'y_main',
        'polygon':   poly,
        'origin':    [dx, 0.0, 0.0],
        'u_axis':    [0.0, 1.0, 0.0],   # DXF x → world Y
        'v_axis':    [0.0, 0.0, 1.0],   # DXF y → world Z
        'normal':    [1.0, 0.0, 0.0],   # extrude in +X
        'thickness': vs,
        'material':  'walnut',
    })

    # ── right_extra sinusoidal brace ─────────────────────────────────────
    # Outside edge at x=right_x (DXF local_x=0); width grows leftward.
    poly = read_dxf_polygon(DXF_DIR / 'right_extra.dxf')
    pieces.append({
        'name':      'right_extra',
        'polygon':   poly,
        'origin':    [right_x, 0.0, 0.0],
        'u_axis':    [-1.0, 0.0, 0.0],  # DXF x → -world X (grows toward center)
        'v_axis':    [0.0, 0.0, 1.0],   # DXF y → world Z
        'normal':    [0.0, 1.0, 0.0],   # extrude in +Y
        'thickness': vs,
        'material':  'walnut',
    })

    # ── left_extra sinusoidal brace ───────────────────────────────────────
    # Outside edge at y=back_y (DXF local_x=0); width grows forward (-Y).
    poly = read_dxf_polygon(DXF_DIR / 'left_extra.dxf')
    pieces.append({
        'name':      'left_extra',
        'polygon':   poly,
        'origin':    [0.0, back_y, 0.0],
        'u_axis':    [0.0, -1.0, 0.0],  # DXF x → -world Y (grows toward center)
        'v_axis':    [0.0, 0.0, 1.0],   # DXF y → world Z
        'normal':    [1.0, 0.0, 0.0],   # extrude in +X
        'thickness': vs,
        'material':  'walnut',
    })

    # ── far_right_extra sinusoidal brace ──────────────────────────────────
    # Flush with end of extended shelves; outside edge at x=far_right_x.
    far_dxf = DXF_DIR / 'far_right_extra.dxf'
    if ext_len > 0 and far_dxf.exists():
        poly = read_dxf_polygon(far_dxf)
        pieces.append({
            'name':      'far_right_extra',
            'polygon':   poly,
            'origin':    [far_right_x, 0.0, 0.0],
            'u_axis':    [-1.0, 0.0, 0.0],  # DXF x → -world X (grows toward center)
            'v_axis':    [0.0, 0.0, 1.0],   # DXF y → world Z
            'normal':    [0.0, 1.0, 0.0],   # extrude in +Y
            'thickness': vs,
            'material':  'walnut',
        })

    return pieces


def _flip_camera(cam, target, scale=1.5):
    """
    Rotate camera 180° around the vertical axis through `target`, then
    scale the camera-to-target distance by `scale`.

    This flips from the wall-facing side to the room-facing side of the
    assembly and pulls the camera back so the full unit fits in frame.
    """
    ox, oy, oz = cam[0]-target[0], cam[1]-target[1], cam[2]-target[2]
    # 180° horizontal flip: negate XY offsets, keep Z
    ox, oy = -ox, -oy
    # Scale full 3D offset
    ox, oy, oz = ox*scale, oy*scale, oz*scale
    return [target[0]+ox, target[1]+oy, target[2]+oz]


def build_renders(cfg):
    """
    Four camera views from the room-facing side of the L-shaped assembly.

    Cameras are 180°-flipped from original wall-side positions then pulled
    back 2.25× total (1.5 × 1.5) so the full unit fits in frame.
    right_arm and left_arm both target the scene centre so they show the
    whole unit from a different angle rather than fixating on one arm.
    """
    cx, cy, cz = 16.0, 10.0, 39.0

    views = [
        # (name, original_wall-side_location, target, fov, total_scale)
        ('corner_iso', [-35.0, -30.0, 80.0], [cx,  cy,   cz  ], 55.0, 2.25),
        ('right_arm',  [ 55.0, -20.0, 65.0], [cx,  cy,   cz  ], 55.0, 2.25),
        ('left_arm',   [-20.0,  38.0, 70.0], [cx,  cy,   cz  ], 55.0, 2.25),
        ('elevated',   [-10.0, -45.0,105.0], [cx,  cy,   45.0], 60.0, 2.25),
    ]

    return [
        {
            'name':            name,
            'camera_location': _flip_camera(cam, tgt, scale=scale),
            'camera_target':   tgt,
            'fov_deg':         fov,
            'resolution':      [1920, 1080],
            'samples':         64,
        }
        for name, cam, tgt, fov, scale in views
    ]


def main():
    cfg = load_config()

    print("Office shelf renderer")
    print(f"  Config:  {CFG_PATH}")
    print(f"  DXF dir: {DXF_DIR}")
    print(f"  Output:  {OUT_DIR}\n")

    pieces = build_pieces(cfg)
    renders = build_renders(cfg)

    print(f"Pieces ({len(pieces)}):")
    for p in pieces:
        print(f"  {p['name']:15s}  {len(p['polygon'])} verts")

    geometry = build_furniture_geometry(pieces, renders)
    run_blender_render(geometry, OUT_DIR)


if __name__ == '__main__':
    main()
