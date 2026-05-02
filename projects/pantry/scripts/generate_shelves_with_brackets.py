#!/usr/bin/env python3
"""
Generate shelf DXF and PDF with mounting brackets at stud positions.

Edit configs/stud_positions.json with exact measurements, then run this script.
"""

import ezdxf
from ezdxf import units
from pathlib import Path
import sys
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, Patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'lib'))
from bracket_geometry import (
    bracket_dims, stud_to_bracket_offset, corner_bracket_offset,
)

# =============================================================================
# BRACKET GEOMETRY FUNCTIONS
# =============================================================================

def create_bracket_outline(base_width, base_height, stem_width, tongue_length,
                            dogbone_radius=None, base_corner_radius=None, n_arc_pts=32):
    """
    Create bracket outline points (the negative pocket shape cut into the shelf).

    Wall-side outer corners at [0, base_height] and [base_width, base_height]:
      These are convex exterior corners of the pocket. We add an intentional 0.5"
      radius so the geometry is well-defined rather than relying on tool runout.
      Circle centers sit below y=0 at (0, base_height - r) and (base_width, base_height - r).
      Each arc is CCW ~53° connecting the outer wall to the bottom edge.

        LEFT  corner center: (0,          base_height - r_wc)
          arc entry:  (0,                base_height)    coming from base top
          arc exit:   (-chord_half,      0)              joining bottom edge

        RIGHT corner center: (base_width, base_height - r_wc)
          arc entry:  (base_width+chord_half, 0)         coming from bottom edge
          arc exit:   (base_width,        base_height)   joining base top

      where chord_half = sqrt(r_wc^2 - (r_wc - base_height)^2)

    Armpit corners at [stem_right, base_height] and [stem_left, base_height]:
      These are inside corners of the pocket — the end mill leaves material behind
      here naturally. The bracket has convex corners there, so they fit fine. SHARP.

    Tongue tip corners at [stem_right, total_height] and [stem_left, total_height]:
      Inside corners of the pocket; bracket has sharp outside corners that must seat
      fully. Requires dogbone relief (180° CCW arc, minimum excursion).

    # OLD (incorrect): armpit fillets (wrong corners were rounded).
    # OLD simple: 9-point T-shape, no arcs anywhere.
    """
    if dogbone_radius is None:
        dogbone_radius = DOGBONE_RADIUS
    if base_corner_radius is None:
        base_corner_radius = BASE_CORNER_RADIUS

    stem_left  = (base_width - stem_width) / 2
    stem_right = (base_width + stem_width) / 2
    total_height = base_height + tongue_length

    # --- Tongue tip dogbones ---
    R  = dogbone_radius
    d  = R / np.sqrt(2)
    Rd = R * np.sqrt(2)

    cr = np.array([stem_right - d, total_height - d])
    arc_r = np.array([
        [cr[0] + R * np.cos(a), cr[1] + R * np.sin(a)]
        for a in np.linspace(np.radians(-45), np.radians(135), n_arc_pts + 1)
    ])
    # arc_r[0]  = (stem_right,      total_height - Rd)  [right wall]
    # arc_r[-1] = (stem_right - Rd, total_height)       [back wall]

    cl = np.array([stem_left + d, total_height - d])
    arc_l = np.array([
        [cl[0] + R * np.cos(a), cl[1] + R * np.sin(a)]
        for a in np.linspace(np.radians(45), np.radians(225), n_arc_pts + 1)
    ])
    # arc_l[0]  = (stem_left + Rd, total_height)        [back wall]
    # arc_l[-1] = (stem_left,      total_height - Rd)   [left wall]

    # --- Wall-side outer corner arcs ---
    r_wc = base_corner_radius                     # 0.5"
    wcy  = base_height - r_wc                     # center y = -0.3" (below base)
    chord_half = np.sqrt(r_wc**2 - wcy**2)        # x-distance from center to y=0 crossing

    # RIGHT wall corner: CCW from (base_width+chord_half, 0) to (base_width, base_height)
    cwc_r = np.array([base_width, wcy])
    arc_wc_r = np.array([
        [cwc_r[0] + r_wc * np.cos(a), cwc_r[1] + r_wc * np.sin(a)]
        for a in np.linspace(np.arctan2(-wcy, chord_half), np.pi / 2,
                             n_arc_pts // 4 + 1)
    ])
    # arc_wc_r[0]  = (base_width + chord_half, 0)
    # arc_wc_r[-1] = (base_width, base_height)

    # LEFT wall corner: CCW from (0, base_height) to (-chord_half, 0)
    cwc_l = np.array([0.0, wcy])
    arc_wc_l = np.array([
        [cwc_l[0] + r_wc * np.cos(a), cwc_l[1] + r_wc * np.sin(a)]
        for a in np.linspace(np.pi / 2, np.arctan2(-wcy, -chord_half),
                             n_arc_pts // 4 + 1)
    ])
    # arc_wc_l[0]  = (0, base_height)
    # arc_wc_l[-1] = (-chord_half, 0)

    pts = []
    pts.extend(arc_wc_r.tolist())  # bottom edge → right outer corner → base top
    # arc_wc_r[-1] = (base_width, base_height); implicit line to right armpit:
    pts.append([stem_right, base_height])   # RIGHT ARMPIT — sharp
    # implicit straight up right wall to arc_r[0]:
    pts.extend(arc_r.tolist())              # right tip dogbone
    # implicit straight along back wall: arc_r[-1] to arc_l[0]
    pts.extend(arc_l.tolist())             # left tip dogbone
    # implicit straight down left wall to arc_la[0]:
    pts.append([stem_left, base_height])    # LEFT ARMPIT — sharp
    # implicit straight along base top to arc_wc_l[0]:
    pts.extend(arc_wc_l.tolist())          # base top → left outer corner → bottom edge
    # arc_wc_l[-1] = (-chord_half, 0); close with bottom edge back to arc_wc_r[0]:
    pts.append([base_width + chord_half, 0.0])
    return np.array(pts)

def create_bracket_cut_line(base_width, base_height, stem_width):
    """
    Create the cut boundary line between full-depth and partial-depth zones.
    """
    stem_left = (base_width - stem_width) / 2
    stem_right = (base_width + stem_width) / 2
    return np.array([
        [stem_left, base_height],
        [stem_right, base_height],
    ])

# Fallback defaults — the canonical values live in configs/stud_positions.json
# (bracket_dogbone_radius, bracket_base_corner_radius). place_bracket() loads
# them from config and passes them through; these defaults only apply if
# create_bracket_outline is called without explicit radii.
DOGBONE_RADIUS = 0.25       # 0.5" diameter for a 3/8" end mill
BASE_CORNER_RADIUS = 0.5    # must be ≥ tool radius (3/16")

def mirror_x_pts(pts, width):
    """Mirror points about X = width/2 (reflect left↔right).

    Applied to bracket pockets in the combined DXF so that when the shelf is
    placed face-down on the CNC table (to machine the bottom), the pockets land
    in the correct pantry position after the board is flipped back face-up.
    The individual shelf outline DXFs are already mirrored by generate_from_patterns.py.
    """
    m = pts.copy()
    m[:, 0] = width - m[:, 0]
    return m


def rotate_points(points, angle_deg):
    """Rotate points around origin by angle in degrees."""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(points, rotation_matrix.T)

def translate_points(points, dx, dy):
    """Translate points by (dx, dy)."""
    return points + np.array([dx, dy])

def place_bracket(stud_center_x, stud_center_y, wall_side, config):
    """
    Place a bracket centered on a stud, oriented for the given wall.

    wall_side: 'left', 'right', 'back', 'back_left_support', 'back_right_support',
              'corner_left', 'corner_right'
    Returns: (outline_points, cut_line_points)
      outline_points includes dogbone relief geometry at the tongue tip corners.
    """
    # Get tongue length based on wall side
    if wall_side == 'right':
        tongue_length = config['right_wall']['tongue_length']
    elif wall_side == 'left':
        tongue_length = config['left_wall']['tongue_length']
    elif wall_side == 'back':
        tongue_length = config['back_wall']['tongue_length']
    elif wall_side in ['back_left_support', 'back_right_support']:
        tongue_length = config['back_shelf_side_brackets']['tongue_length']
    elif wall_side in ['corner_left', 'corner_right']:
        tongue_length = config['back_shelf_corner_brackets']['tongue_length']
    else:
        tongue_length = 6.0  # default

    d = bracket_dims(config, tongue_length)
    base_width = d['base_width']

    outline = create_bracket_outline(base_width, d['base_height'], d['stem_width'], tongue_length,
                                     dogbone_radius=d['dogbone_radius'],
                                     base_corner_radius=d['base_corner_radius'])
    cut_line = create_bracket_cut_line(base_width, d['base_height'], d['stem_width'])

    # Center bracket on origin
    outline[:, 0] -= base_width / 2
    cut_line[:, 0] -= base_width / 2

    if wall_side == 'right':
        # Rotate 90° CCW so stem points left (into pantry)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)

    elif wall_side == 'left':
        # Rotate 90° CW so stem points right (into pantry)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)

    elif wall_side == 'back':
        # Rotate 180° so stem points down (into pantry)
        outline = rotate_points(outline, 180)
        cut_line = rotate_points(cut_line, 180)

    elif wall_side == 'back_left_support':
        # Mounted to left wall, stem points right (into back shelf)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)

    elif wall_side == 'back_right_support':
        # Mounted to right wall, stem points left (into back shelf)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)

    elif wall_side == 'corner_left':
        # Back-left corner, stem points right (into back shelf)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)

    elif wall_side == 'corner_right':
        # Back-right corner, stem points left (into back shelf)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)

    # Translate to position
    outline = translate_points(outline, stud_center_x, stud_center_y)
    cut_line = translate_points(cut_line, stud_center_x, stud_center_y)

    return outline, cut_line


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / 'output'
    config_path = project_dir / 'configs' / 'stud_positions.json'

    # Load stud positions
    print(f"Loading stud positions from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"  Right wall: {len(config['right_wall']['stud_centers_y'])} studs, {config['right_wall']['tongue_length']}\" tongue")
    print(f"  Left wall:  {len(config['left_wall']['stud_centers_y'])} studs, {config['left_wall']['tongue_length']}\" tongue")
    print(f"  Back wall:  {len(config['back_wall']['stud_centers_x'])} studs, {config['back_wall']['tongue_length']}\" tongue")
    if config['back_shelf_side_brackets']['enabled']:
        print(f"  Back shelf side brackets: enabled, {config['back_shelf_side_brackets']['tongue_length']}\" tongue at stud_y={config['back_shelf_side_brackets']['stud_y']}")

    # Layout parameters
    PANTRY_WIDTH = 48.0
    PANTRY_DEPTH = 49.0
    LEVEL_SPACING = 60.0
    BACK_SHELF_Y_OFFSET = 12.0

    # Find all shelf DXF files
    shelf_files = list(output_dir.glob('shelf_*.dxf'))

    # Parse shelf names
    levels = {}
    for f in shelf_files:
        if '_exact' in f.stem or 'all_' in f.stem:
            continue
        match = re.match(r'shelf_([LRB])(\d+)', f.stem)
        if match:
            shelf_type = match.group(1)
            height = int(match.group(2))
            if height not in levels:
                levels[height] = []
            levels[height].append({
                'file': f,
                'type': shelf_type,
                'name': f.stem
            })

    sorted_heights = sorted(levels.keys())
    print(f"\nGenerating outputs for {len(sorted_heights)} levels")

    # =============================================================================
    # CREATE DXF
    # =============================================================================

    doc = ezdxf.new('R2010')
    doc.units = units.IN
    msp = doc.modelspace()

    bracket_count = 0

    # Store bracket data for PDF generation
    all_brackets = {}

    # Raw stud positions for PDF markers (same for every level — the studs don't move).
    # axis: 'x' or 'y' — which coordinate is the meaningful stud position along its wall.
    # label_pos: 'left'|'right'|'above'|'below' — where to anchor the text relative to the marker.
    stud_markers = []
    for sx in config['back_wall']['stud_centers_x']:
        stud_markers.append({'x': sx, 'y': config['back_wall']['wall_y'],
                             'axis': 'x', 'label_pos': 'above'})
    for sy in config['left_wall']['stud_centers_y']:
        stud_markers.append({'x': config['left_wall']['wall_x'], 'y': sy,
                             'axis': 'y', 'label_pos': 'left'})
    for sy in config['right_wall']['stud_centers_y']:
        stud_markers.append({'x': config['right_wall']['wall_x'], 'y': sy,
                             'axis': 'y', 'label_pos': 'right'})
    if config.get('back_shelf_side_brackets', {}).get('enabled', False):
        bs = config['back_shelf_side_brackets']
        stud_markers.append({'x': config['left_wall']['wall_x'],  'y': bs['stud_y'],
                             'axis': 'y', 'label_pos': 'left'})
        stud_markers.append({'x': config['right_wall']['wall_x'], 'y': bs['stud_y'],
                             'axis': 'y', 'label_pos': 'right'})
    if config.get('back_shelf_corner_brackets', {}).get('enabled', False):
        cc = config['back_shelf_corner_brackets']
        stud_markers.append({'x': cc['left_corner_stud_x'],  'y': cc['back_wall_y'],
                             'axis': 'x', 'label_pos': 'above'})
        stud_markers.append({'x': cc['right_corner_stud_x'], 'y': cc['back_wall_y'],
                             'axis': 'x', 'label_pos': 'above'})

    for level_idx, height in enumerate(sorted_heights):
        x_offset = level_idx * LEVEL_SPACING
        all_brackets[height] = []

        # Add level label
        msp.add_text(
            f'{height}"',
            dxfattribs={'height': 2.0, 'layer': 'LABELS', 'color': 8}
        ).set_placement((x_offset + PANTRY_WIDTH/2, -3), align=ezdxf.enums.TextEntityAlignment.CENTER)

        for shelf_info in levels[height]:
            shelf_file = shelf_info['file']
            shelf_type = shelf_info['type']
            shelf_name = shelf_info['name']

            shelf_doc = ezdxf.readfile(shelf_file)

            if shelf_type == 'L':
                wall_side = 'left'
                y_offset = 0
                layer = 'SHELF_LEFT'
                color = 5
            elif shelf_type == 'R':
                wall_side = 'right'
                y_offset = 0
                layer = 'SHELF_RIGHT'
                color = 1
            else:
                wall_side = 'back'
                y_offset = BACK_SHELF_Y_OFFSET
                layer = 'SHELF_BACK'
                color = 3

            # Add shelf outline
            for entity in shelf_doc.modelspace():
                if entity.dxftype() == 'LWPOLYLINE':
                    points = list(entity.get_points())
                    offset_points = [(p[0] + x_offset, p[1] + y_offset) for p in points]
                    msp.add_lwpolyline(offset_points, close=True,
                                       dxfattribs={'layer': layer, 'color': color})

            # Add brackets
            shelf_brackets = 0

            if wall_side == 'left':
                wall_x = config['left_wall']['wall_x']
                lw_off = stud_to_bracket_offset(config, config['left_wall']['tongue_length'],
                                                config['left_wall'].get('stud_offset_sign', 0))
                for stud_y in config['left_wall']['stud_centers_y']:
                    bracket_y = stud_y + lw_off
                    outline, cut_line = place_bracket(wall_x, bracket_y, 'left', config)
                    # Store for PDF (without DXF offset)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'left'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

            elif wall_side == 'right':
                wall_x = config['right_wall']['wall_x']
                rw_off = stud_to_bracket_offset(config, config['right_wall']['tongue_length'],
                                                config['right_wall'].get('stud_offset_sign', 0))
                for stud_y in config['right_wall']['stud_centers_y']:
                    bracket_y = stud_y + rw_off
                    outline, cut_line = place_bracket(wall_x, bracket_y, 'right', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'right'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

            elif wall_side == 'back':
                wall_y = config['back_wall']['wall_y']
                bw_off = stud_to_bracket_offset(config, config['back_wall']['tongue_length'],
                                                config['back_wall'].get('stud_offset_sign', 0))
                for stud_x in config['back_wall']['stud_centers_x']:
                    bracket_x = stud_x + bw_off
                    outline, cut_line = place_bracket(bracket_x, wall_y, 'back', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

                # Add side support brackets for back shelves
                if config['back_shelf_side_brackets']['enabled']:
                    bs = config['back_shelf_side_brackets']
                    bs_off = stud_to_bracket_offset(config, bs['tongue_length'],
                                                    bs.get('stud_offset_sign', 0))
                    side_y = bs['stud_y'] + bs_off

                    # Left side support bracket
                    left_x = config['left_wall']['wall_x']
                    outline, cut_line = place_bracket(left_x, side_y, 'back_left_support', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back_left_support'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

                    # Right side support bracket
                    right_x = config['right_wall']['wall_x']
                    outline, cut_line = place_bracket(right_x, side_y, 'back_right_support', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back_right_support'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

                # Add corner brackets for back shelves
                if config.get('back_shelf_corner_brackets', {}).get('enabled', False):
                    corner_cfg = config['back_shelf_corner_brackets']
                    cb_y = corner_cfg['back_wall_y'] - corner_bracket_offset(config, corner_cfg['tongue_length'])

                    # Left corner bracket (facing right)
                    outline, cut_line = place_bracket(corner_cfg['left_corner_stud_x'], cb_y, 'corner_left', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'corner_left'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

                    # Right corner bracket (facing left)
                    outline, cut_line = place_bracket(corner_cfg['right_corner_stud_x'], cb_y, 'corner_right', config)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'corner_right'
                    })
                    # Mirror bracket X for bottom-face machining, then apply layout offset
                    outline = mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = mirror_x_pts(cut_line, PANTRY_WIDTH)
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    shelf_brackets += 1

            bracket_count += shelf_brackets
            print(f"  Level {height}\", {shelf_name}: {shelf_brackets} brackets")

    # Save DXF
    dxf_path = output_dir / 'all_shelves_with_brackets.dxf'
    doc.saveas(dxf_path)
    print(f"\n✓ Saved DXF: {dxf_path}")
    print(f"  Total brackets: {bracket_count}")

    # =============================================================================
    # CREATE PDF LAYOUT
    # =============================================================================

    print(f"\nGenerating PDF layout...")

    # Load shelf polygons for PDF
    shelf_polygons = {}
    for height in sorted_heights:
        shelf_polygons[height] = []
        for shelf_info in levels[height]:
            shelf_doc = ezdxf.readfile(shelf_info['file'])
            for entity in shelf_doc.modelspace():
                if entity.dxftype() == 'LWPOLYLINE':
                    # Un-mirror X: DXF files are mirrored for bottom-face machining;
                    # apply X → PANTRY_WIDTH - X again to restore pantry coordinates for PDF.
                    points = np.array([(PANTRY_WIDTH - p[0], p[1]) for p in entity.get_points()])
                    shelf_polygons[height].append({
                        'points': points,
                        'type': shelf_info['type'],
                        'name': shelf_info['name']
                    })

    # Color mapping
    shelf_colors = {
        'L': '#4A90E2',  # Blue for left
        'R': '#E24A4A',  # Red for right
        'B': '#50C878',  # Green for back
    }
    bracket_color = '#FF00FF'  # Magenta

    pdf_path = output_dir / 'pantry_layout.pdf'
    with PdfPages(pdf_path) as pdf:
        for height in sorted_heights:
            fig, ax = plt.subplots(figsize=(11, 8.5))

            # Draw pantry outline
            pantry_outline = mpatches.Rectangle(
                (0, 0), PANTRY_WIDTH, PANTRY_DEPTH,
                fill=False, edgecolor='black', linewidth=2
            )
            ax.add_patch(pantry_outline)

            # Door label
            ax.text(PANTRY_WIDTH/2, -2, 'DOOR',
                   ha='center', va='top', fontsize=12, fontweight='bold')
            ax.plot([0, PANTRY_WIDTH], [0, 0], 'k--', linewidth=1, alpha=0.5)

            # Wall labels
            ax.text(-2, PANTRY_DEPTH/2, 'EAST\n(Left)',
                   ha='right', va='center', fontsize=10, rotation=90)
            ax.text(PANTRY_WIDTH + 2, PANTRY_DEPTH/2, 'WEST\n(Right)',
                   ha='left', va='center', fontsize=10, rotation=90)
            ax.text(PANTRY_WIDTH/2, PANTRY_DEPTH + 2, 'SOUTH\n(Back)',
                   ha='center', va='bottom', fontsize=10)

            # Draw shelves
            for shelf_data in shelf_polygons[height]:
                color = shelf_colors.get(shelf_data['type'], '#CCCCCC')
                poly_patch = mpatches.Polygon(
                    shelf_data['points'], closed=True,
                    facecolor=color, edgecolor='black',
                    linewidth=1.5, alpha=0.7
                )
                ax.add_patch(poly_patch)

                # Shelf label
                centroid_x = np.mean(shelf_data['points'][:, 0])
                centroid_y = np.mean(shelf_data['points'][:, 1])
                ax.text(centroid_x, centroid_y, shelf_data['name'],
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Draw brackets
            for bracket_data in all_brackets[height]:
                outline = bracket_data['outline']
                bracket_patch = mpatches.Polygon(
                    outline[:-1], closed=True,  # Remove duplicate closing point
                    facecolor=bracket_color, edgecolor='black',
                    linewidth=1, alpha=0.6
                )
                ax.add_patch(bracket_patch)

            # Draw stud-center markers (raw pantry coords — shelves & brackets are
            # already in pantry coords here, no mirror needed).
            label_offsets = {
                'above': (0,  0.6, 'center', 'bottom'),
                'below': (0, -0.6, 'center', 'top'),
                'left':  (-0.6, 0, 'right',  'center'),
                'right': ( 0.6, 0, 'left',   'center'),
            }
            for s in stud_markers:
                sx, sy = s['x'], s['y']
                ax.plot(sx, sy, marker='+', color='black', markersize=10,
                        markeredgewidth=1.5, zorder=5)
                ax.plot(sx, sy, marker='o', markerfacecolor='none',
                        markeredgecolor='black', markersize=6, markeredgewidth=1.0, zorder=5)
                # Label with the meaningful coordinate
                value = sx if s['axis'] == 'x' else sy
                dx, dy, ha, va = label_offsets[s['label_pos']]
                ax.text(sx + dx, sy + dy, f"{s['axis']}={value:.3f}",
                        ha=ha, va=va, fontsize=7, color='black',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='gray', linewidth=0.5, alpha=0.85),
                        zorder=6)

            # Title
            ax.set_title(f'Level: {height}" from floor\n({len(shelf_polygons[height])} shelves, {len(all_brackets[height])} brackets)',
                        fontsize=14, fontweight='bold', pad=20)

            # Axis settings
            ax.set_xlim(-5, PANTRY_WIDTH + 5)
            ax.set_ylim(-5, PANTRY_DEPTH + 5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('Width (inches)', fontsize=10)
            ax.set_ylabel('Depth (inches)', fontsize=10)

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Patch(facecolor=shelf_colors['L'], edgecolor='black', label='Left (East)'),
                Patch(facecolor=shelf_colors['R'], edgecolor='black', label='Right (West)'),
                Patch(facecolor=shelf_colors['B'], edgecolor='black', label='Back (South)'),
                Patch(facecolor=bracket_color, edgecolor='black', alpha=0.6, label='Brackets'),
                Line2D([0], [0], marker='+', color='black', markersize=10, markeredgewidth=1.5,
                       linestyle='', label='Stud center'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

            # Dimension annotations
            ax.annotate('', xy=(PANTRY_WIDTH, -3), xytext=(0, -3),
                       arrowprops=dict(arrowstyle='<->', lw=1.5))
            ax.text(PANTRY_WIDTH/2, -3.5, f'{PANTRY_WIDTH}"',
                   ha='center', va='top', fontsize=9)

            ax.annotate('', xy=(PANTRY_WIDTH + 3, PANTRY_DEPTH), xytext=(PANTRY_WIDTH + 3, 0),
                       arrowprops=dict(arrowstyle='<->', lw=1.5))
            ax.text(PANTRY_WIDTH + 3.5, PANTRY_DEPTH/2, f'{PANTRY_DEPTH}"',
                   ha='left', va='center', fontsize=9, rotation=90)

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"✓ Saved PDF: {pdf_path}")
    print(f"  {len(sorted_heights)} pages")


if __name__ == '__main__':
    main()
