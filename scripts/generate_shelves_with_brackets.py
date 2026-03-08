#!/usr/bin/env python3
"""
Generate shelf DXF and PDF with mounting brackets at stud positions.

Edit configs/stud_positions.json with exact measurements, then run this script.
"""

import ezdxf
from ezdxf import units
from pathlib import Path
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon, Patch

# =============================================================================
# BRACKET GEOMETRY FUNCTIONS
# =============================================================================

def create_bracket_outline(base_width, base_height, stem_width, tongue_length):
    """
    Create bracket outline points for given dimensions.

    Returns numpy array of points forming the T-shaped bracket.
    """
    # Stem is centered on the base
    stem_left = (base_width - stem_width) / 2
    stem_right = (base_width + stem_width) / 2
    total_height = base_height + tongue_length

    return np.array([
        [0.0, 0.0],
        [base_width, 0.0],
        [base_width, base_height],
        [stem_right, base_height],
        [stem_right, total_height],
        [stem_left, total_height],
        [stem_left, base_height],
        [0.0, base_height],
        [0.0, 0.0],
    ])

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

# Dogbone relief circles at the two tip corners of the tongue.
# Diameter 0.5" gives clearance for a 3/8" end mill (tool radius 3/16").
DOGBONE_RADIUS = 0.25  # inches (1/2" diameter)

def create_bracket_dogbone_centers(base_width, base_height, stem_width, tongue_length):
    """
    Return the two dogbone relief circle centers at the tip corners of the bracket tongue.

    In local (pre-rotation) coordinates:
      - [stem_left,  base_height + tongue_length]  (left tip corner)
      - [stem_right, base_height + tongue_length]  (right tip corner)

    These are the inside corners at the FAR END of the tongue where a square-cornered
    bracket cannot seat without relief. The base/armpit corners are NOT included here.
    """
    stem_left = (base_width - stem_width) / 2
    stem_right = (base_width + stem_width) / 2
    total_height = base_height + tongue_length
    return np.array([
        [stem_left,  total_height],
        [stem_right, total_height],
    ])

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
    Returns: (outline_points, cut_line_points, dogbone_centers)
      dogbone_centers: 2x2 array of [x, y] circle centers for 0.5"-diameter relief
                       holes at the two inside corners of the tongue tip.
    # OLD return was: (outline_points, cut_line_points)
    """
    base_width = config['bracket_base_width']
    base_height = config['bracket_base_height']
    stem_width = config['bracket_stem_width']

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

    outline = create_bracket_outline(base_width, base_height, stem_width, tongue_length)
    cut_line = create_bracket_cut_line(base_width, base_height, stem_width)
    dogbone_centers = create_bracket_dogbone_centers(base_width, base_height, stem_width, tongue_length)

    # Center bracket on origin
    outline[:, 0] -= base_width / 2
    cut_line[:, 0] -= base_width / 2
    dogbone_centers[:, 0] -= base_width / 2

    if wall_side == 'right':
        # Rotate 90° CCW so stem points left (into pantry)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)
        dogbone_centers = rotate_points(dogbone_centers, 90)

    elif wall_side == 'left':
        # Rotate 90° CW so stem points right (into pantry)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)
        dogbone_centers = rotate_points(dogbone_centers, -90)

    elif wall_side == 'back':
        # Rotate 180° so stem points down (into pantry)
        outline = rotate_points(outline, 180)
        cut_line = rotate_points(cut_line, 180)
        dogbone_centers = rotate_points(dogbone_centers, 180)

    elif wall_side == 'back_left_support':
        # Mounted to left wall, stem points right (into back shelf)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)
        dogbone_centers = rotate_points(dogbone_centers, -90)

    elif wall_side == 'back_right_support':
        # Mounted to right wall, stem points left (into back shelf)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)
        dogbone_centers = rotate_points(dogbone_centers, 90)

    elif wall_side == 'corner_left':
        # Back-left corner, stem points right (into back shelf)
        outline = rotate_points(outline, -90)
        cut_line = rotate_points(cut_line, -90)
        dogbone_centers = rotate_points(dogbone_centers, -90)

    elif wall_side == 'corner_right':
        # Back-right corner, stem points left (into back shelf)
        outline = rotate_points(outline, 90)
        cut_line = rotate_points(cut_line, 90)
        dogbone_centers = rotate_points(dogbone_centers, 90)

    # Translate to position
    outline = translate_points(outline, stud_center_x, stud_center_y)
    cut_line = translate_points(cut_line, stud_center_x, stud_center_y)
    dogbone_centers = translate_points(dogbone_centers, stud_center_x, stud_center_y)

    return outline, cut_line, dogbone_centers


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
        print(f"  Back shelf side brackets: enabled, {config['back_shelf_side_brackets']['tongue_length']}\" tongue at y={config['back_shelf_side_brackets']['y_position']}")

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
                for stud_y in config['left_wall']['stud_centers_y']:
                    outline, cut_line, dogbone_centers = place_bracket(wall_x, stud_y, 'left', config)  # was: outline, cut_line
                    # Store for PDF (without DXF offset)
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'left'
                    })
                    # Apply DXF offset
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

            elif wall_side == 'right':
                wall_x = config['right_wall']['wall_x']
                for stud_y in config['right_wall']['stud_centers_y']:
                    outline, cut_line, dogbone_centers = place_bracket(wall_x, stud_y, 'right', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'right'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

            elif wall_side == 'back':
                wall_y = config['back_wall']['wall_y']
                for stud_x in config['back_wall']['stud_centers_x']:
                    outline, cut_line, dogbone_centers = place_bracket(stud_x, wall_y, 'back', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

                # Add side support brackets for back shelves
                if config['back_shelf_side_brackets']['enabled']:
                    side_y = config['back_shelf_side_brackets']['y_position']

                    # Left side support bracket
                    left_x = config['left_wall']['wall_x']
                    outline, cut_line, dogbone_centers = place_bracket(left_x, side_y, 'back_left_support', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back_left_support'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

                    # Right side support bracket
                    right_x = config['right_wall']['wall_x']
                    outline, cut_line, dogbone_centers = place_bracket(right_x, side_y, 'back_right_support', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'back_right_support'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

                # Add corner brackets for back shelves
                if config.get('back_shelf_corner_brackets', {}).get('enabled', False):
                    corner_cfg = config['back_shelf_corner_brackets']

                    # Left corner bracket (facing right)
                    lc = corner_cfg['left_corner']
                    outline, cut_line, dogbone_centers = place_bracket(lc['x'], lc['y'], 'corner_left', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'corner_left'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
                    shelf_brackets += 1

                    # Right corner bracket (facing left)
                    rc = corner_cfg['right_corner']
                    outline, cut_line, dogbone_centers = place_bracket(rc['x'], rc['y'], 'corner_right', config)  # was: outline, cut_line
                    all_brackets[height].append({
                        'outline': outline.copy(),
                        'wall': 'corner_right'
                    })
                    outline = translate_points(outline, x_offset, y_offset)
                    cut_line = translate_points(cut_line, x_offset, y_offset)
                    dogbone_centers_dxf = translate_points(dogbone_centers.copy(), x_offset, y_offset)
                    msp.add_lwpolyline([(p[0], p[1]) for p in outline], close=True,
                                       dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
                    msp.add_line((cut_line[0][0], cut_line[0][1]), (cut_line[1][0], cut_line[1][1]),
                                dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
                    for dc in dogbone_centers_dxf:
                        msp.add_circle((dc[0], dc[1]), DOGBONE_RADIUS,
                                       dxfattribs={'layer': 'BRACKET_DOGBONE', 'color': 2})
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
                    points = np.array([(p[0], p[1]) for p in entity.get_points()])
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
            legend_elements = [
                Patch(facecolor=shelf_colors['L'], edgecolor='black', label='Left (East)'),
                Patch(facecolor=shelf_colors['R'], edgecolor='black', label='Right (West)'),
                Patch(facecolor=shelf_colors['B'], edgecolor='black', label='Back (South)'),
                Patch(facecolor=bracket_color, edgecolor='black', alpha=0.6, label='Brackets'),
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
