#!/usr/bin/env python3
"""
Generate PDF cutting templates with:
1. Dimensioned overview of complete level
2. Separated individual shelves with cut lines
3. Optimized plywood sheet layout (8' x 4')
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    generate_interior_mask
)
from config import ShelfConfig


def compute_corner_data(config, level):
    """
    Compute corner arcs and intersection points for a level.
    Returns data needed for cutting shelves apart.
    """
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        return None

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    radius = config.design_params['interior_corner_radius']

    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    left_params = (left_depth, amplitude, period, left_offset)
    right_params = (right_depth, amplitude, period, right_offset)
    back_params = (back_depth, amplitude, period, back_offset)

    # Solve for left-back corner
    try:
        lb_center, lb_point1, lb_point2, lb_pos1, lb_pos2 = solve_tangent_circle_two_sinusoids(
            pos1_init=pantry_depth - 10,
            base_depth1=left_depth,
            amplitude1=amplitude,
            period1=period,
            offset1=left_offset,
            pos2_init=10,
            base_depth2=back_depth,
            amplitude2=amplitude,
            period2=period,
            offset2=back_offset,
            radius=radius,
            wall1_type='L',
            wall2_type='B',
            pantry_width=pantry_width,
            pantry_depth=pantry_depth,
            corner_type='left-back',
            left_params=left_params,
            right_params=right_params,
            back_params=back_params
        )
        lb_arc = generate_circle_arc(lb_center, lb_point1, lb_point2, num_points=30, interior_arc=True)
        lb_cut_y = (lb_point1[1] + lb_point2[1]) / 2
    except Exception as e:
        print(f"Warning: Failed to solve left-back corner for level {level}: {e}")
        lb_arc = None
        lb_cut_y = None

    # Solve for right-back corner
    try:
        rb_center, rb_point1, rb_point2, rb_pos1, rb_pos2 = solve_tangent_circle_two_sinusoids(
            pos1_init=pantry_depth - 10,
            base_depth1=right_depth,
            amplitude1=amplitude,
            period1=period,
            offset1=right_offset,
            pos2_init=pantry_width - 10,
            base_depth2=back_depth,
            amplitude2=amplitude,
            period2=period,
            offset2=back_offset,
            radius=radius,
            wall1_type='R',
            wall2_type='B',
            pantry_width=pantry_width,
            pantry_depth=pantry_depth,
            corner_type='right-back',
            left_params=left_params,
            right_params=right_params,
            back_params=back_params
        )
        rb_arc = generate_circle_arc(rb_center, rb_point1, rb_point2, num_points=30, interior_arc=True)
        rb_cut_y = (rb_point1[1] + rb_point2[1]) / 2
    except Exception as e:
        print(f"Warning: Failed to solve right-back corner for level {level}: {e}")
        rb_arc = None
        rb_cut_y = None

    return {
        'lb_arc': lb_arc,
        'rb_arc': rb_arc,
        'lb_cut_y': lb_cut_y,
        'rb_cut_y': rb_cut_y
    }


def generate_shelf_piece(config, level, wall, cut_data):
    """
    Generate the outline points for a single shelf piece after cutting.

    Args:
        config: ShelfConfig instance
        level: Level number
        wall: 'E' (left), 'S' (back), or 'W' (right)
        cut_data: Dictionary with cut line y-coordinates

    Returns:
        np.ndarray of [x, y] points forming the cut shelf outline
    """
    shelf = config.get_shelf(level, wall)
    if not shelf:
        return None

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']

    base_depth = config.get_base_depth(wall)
    offset = shelf['sinusoid_offset']

    # Generate main sinusoid curve
    sinusoid_wall = generate_sinusoid_points(
        0, pantry_depth if wall in ['E', 'W'] else pantry_width,
        base_depth, amplitude, period, offset, num_points=200
    )

    sinusoid_pantry = np.array([
        wall_to_pantry_coords(pos, depth, wall, pantry_width, pantry_depth)
        for pos, depth in sinusoid_wall
    ])

    if wall == 'S':
        # Back shelf: runs along x-axis, cut at both ends
        lb_cut_y = cut_data.get('lb_cut_y', pantry_depth)
        rb_cut_y = cut_data.get('rb_cut_y', pantry_depth)

        # Back shelf sinusoid goes into the pantry (y decreases from pantry_depth)
        # We want to keep the full sinusoid and just cut the corners

        # Build outline: left cut corner -> sinusoid -> right cut corner -> close
        outline = []
        outline.append([0, lb_cut_y])  # Left edge at cut line
        outline.append([0, pantry_depth])  # Left corner on wall
        outline.extend(sinusoid_pantry)  # Full sinusoid curve
        outline.append([pantry_width, pantry_depth])  # Right corner on wall
        outline.append([pantry_width, rb_cut_y])  # Right edge at cut line
        # Close back along cut line (approximately)
        outline.append([pantry_width, (lb_cut_y + rb_cut_y) / 2])
        outline.append([0, (lb_cut_y + rb_cut_y) / 2])

        return np.array(outline)

    elif wall == 'E':
        # Left shelf: cut at the back (south) end
        cut_y = cut_data.get('lb_cut_y', pantry_depth)

        # Filter sinusoid points to only those with y <= cut_y
        sinusoid_filtered = sinusoid_pantry[sinusoid_pantry[:, 1] <= cut_y]

        # Build outline: wall edge -> sinusoid -> cut line -> back
        outline = [[0, 0]]  # NW corner
        outline.extend(sinusoid_filtered)  # Sinusoid from front to cut
        outline.append([sinusoid_filtered[-1][0], cut_y])  # To cut line
        outline.append([0, cut_y])  # Along cut line to wall

        return np.array(outline)

    elif wall == 'W':
        # Right shelf: cut at the back (south) end
        cut_y = cut_data.get('rb_cut_y', pantry_depth)

        # Filter sinusoid points to only those with y <= cut_y
        sinusoid_filtered = sinusoid_pantry[sinusoid_pantry[:, 1] <= cut_y]

        # Build outline: wall edge -> sinusoid -> cut line -> back
        outline = [[pantry_width, 0]]  # NE corner
        outline.extend(sinusoid_filtered)  # Sinusoid from front to cut
        outline.append([sinusoid_filtered[-1][0], cut_y])  # To cut line
        outline.append([pantry_width, cut_y])  # Along cut line to wall

        return np.array(outline)

    return None


def create_overview_page(ax, config, level):
    """Create dimensioned overview page similar to shelf_level_*.png"""
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']

    # Draw pantry outline
    pantry_outline = np.array([
        [0, 0], [pantry_width, 0], [pantry_width, pantry_depth],
        [0, pantry_depth], [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=3)

    # Get shelves
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        return

    # Generate interior mask
    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    X, Y, interior_mask = generate_interior_mask(
        left_depth, amplitude, period, left_offset,
        right_depth, amplitude, period, right_offset,
        back_depth, amplitude, period, back_offset,
        pantry_width, pantry_depth,
        resolution=150
    )

    ax.contourf(X, Y, ~interior_mask, levels=[0.5, 1.5], colors=['#FFE4B5'], alpha=0.4)
    ax.contourf(X, Y, interior_mask, levels=[0.5, 1.5], colors=['#E0F7FA'], alpha=0.3)

    # Generate and plot curves
    for wall, color, label in [('E', 'blue', 'Left'), ('S', 'red', 'Back'), ('W', 'green', 'Right')]:
        shelf = config.get_shelf(level, wall)
        depth = config.get_base_depth(wall)
        offset = shelf['sinusoid_offset']

        extent_end = pantry_depth if wall in ['E', 'W'] else pantry_width
        curve_wall = generate_sinusoid_points(0, extent_end, depth, amplitude, period, offset, num_points=200)
        curve = np.array([wall_to_pantry_coords(pos, depth, wall, pantry_width, pantry_depth)
                         for pos, depth in curve_wall])

        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3, label=f'{label} shelf', alpha=0.8)

    # Plot corner arcs and cut lines
    corner_data = compute_corner_data(config, level)
    if corner_data:
        if corner_data['lb_arc'] is not None:
            ax.plot(corner_data['lb_arc'][:, 0], corner_data['lb_arc'][:, 1],
                   'cyan', linewidth=4, label='Corner arcs', alpha=0.9)
        if corner_data['rb_arc'] is not None:
            ax.plot(corner_data['rb_arc'][:, 0], corner_data['rb_arc'][:, 1],
                   'magenta', linewidth=4, alpha=0.9)

        if corner_data['lb_cut_y']:
            ax.plot([0, pantry_width], [corner_data['lb_cut_y'], corner_data['lb_cut_y']],
                   'r--', linewidth=2, alpha=0.7, label='Cut lines')
        if corner_data['rb_cut_y']:
            ax.plot([0, pantry_width], [corner_data['rb_cut_y'], corner_data['rb_cut_y']],
                   'r--', linewidth=2, alpha=0.7)

    height = left_shelf['height']
    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=12)
    ax.set_ylabel('Y (inches)', fontsize=12)
    ax.set_title(f'Level {level} - Height: {height:.1f}" - Complete Assembly',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)


def create_separated_shelves_page(ax, config, level):
    """Create page showing three separated shelf pieces"""
    corner_data = compute_corner_data(config, level)
    if not corner_data:
        return

    # Generate the three shelf pieces
    left_piece = generate_shelf_piece(config, level, 'E', corner_data)
    back_piece = generate_shelf_piece(config, level, 'S', corner_data)
    right_piece = generate_shelf_piece(config, level, 'W', corner_data)

    # Layout: stack vertically with spacing
    spacing = 10

    # Draw each piece with offset
    pieces = [
        (left_piece, 'blue', 'Left (East)'),
        (back_piece, 'red', 'Back (South)'),
        (right_piece, 'green', 'Right (West)')
    ]

    y_offset = 0
    for piece, color, label in pieces:
        if piece is None:
            continue

        # Offset piece
        piece_offset = piece.copy()
        piece_offset[:, 1] += y_offset

        # Draw
        ax.plot(piece_offset[:, 0], piece_offset[:, 1], color=color, linewidth=2)
        ax.fill(piece_offset[:, 0], piece_offset[:, 1], color=color, alpha=0.2)

        # Add label
        center_x = np.mean(piece_offset[:, 0])
        center_y = np.mean(piece_offset[:, 1])
        ax.text(center_x, center_y, label, fontsize=12, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Update offset for next piece
        max_y = np.max(piece_offset[:, 1])
        y_offset = max_y + spacing

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=11)
    ax.set_ylabel('Y (inches)', fontsize=11)
    ax.set_title(f'Level {level} - Individual Shelf Pieces (Cut Apart)',
                fontsize=13, fontweight='bold')


def optimize_plywood_layout(shelf_pieces, sheet_width=96, sheet_height=48):
    """
    Optimize layout of shelf pieces on plywood sheets.
    Uses sinusoid alignment to minimize waste.

    Args:
        shelf_pieces: List of (piece_outline, metadata) tuples
        sheet_width: 8' = 96"
        sheet_height: 4' = 48"

    Returns:
        List of (sheet_number, piece_outline, position, rotation) tuples
    """
    layouts = []
    current_sheet = 0
    current_x = 0
    current_row = 0
    row_height = 0
    row_pieces = []

    for i, (piece, meta) in enumerate(shelf_pieces):
        # Get bounding box
        min_x, max_x = np.min(piece[:, 0]), np.max(piece[:, 0])
        min_y, max_y = np.min(piece[:, 1]), np.max(piece[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        # Simple row-based packing for now
        if current_x + width > sheet_width:
            # Move to next row
            current_x = 0
            current_row += row_height + 2  # 2" spacing
            row_height = 0

            if current_row + height > sheet_height:
                # Move to next sheet
                current_sheet += 1
                current_row = 0
                current_x = 0

        # Place piece
        position = np.array([current_x - min_x, current_row - min_y])
        layouts.append((current_sheet, piece, position, 0, meta))

        # Update trackers
        current_x += width + 2  # 2" spacing
        row_height = max(row_height, height)

    return layouts


def create_plywood_layout_pages(pdf, config, level):
    """Create plywood sheet layout pages for a level"""
    corner_data = compute_corner_data(config, level)
    if not corner_data:
        return

    # Generate all shelf pieces with metadata
    shelf_pieces = []
    for wall in ['E', 'S', 'W']:
        piece = generate_shelf_piece(config, level, wall, corner_data)
        if piece is not None:
            shelf_pieces.append((piece, {'level': level, 'wall': wall}))

    # Optimize layout
    layouts = optimize_plywood_layout(shelf_pieces)

    # Group by sheet
    sheets = {}
    for sheet_num, piece, position, rotation, meta in layouts:
        if sheet_num not in sheets:
            sheets[sheet_num] = []
        sheets[sheet_num].append((piece, position, rotation, meta))

    # Draw each sheet
    for sheet_num in sorted(sheets.keys()):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)

        # Draw sheet outline (8' x 4' = 96" x 48")
        sheet_rect = patches.Rectangle(
            (0, 0), 96, 48,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(sheet_rect)

        # Draw each piece
        colors = {'E': 'blue', 'S': 'red', 'W': 'green'}
        for piece, position, rotation, meta in sheets[sheet_num]:
            piece_placed = piece + position
            color = colors.get(meta['wall'], 'gray')

            ax.plot(piece_placed[:, 0], piece_placed[:, 1], color=color, linewidth=2)
            ax.fill(piece_placed[:, 0], piece_placed[:, 1], color=color, alpha=0.3)

            # Add label
            center_x = np.mean(piece_placed[:, 0])
            center_y = np.mean(piece_placed[:, 1])
            wall_names = {'E': 'L', 'S': 'B', 'W': 'R'}
            ax.text(center_x, center_y, f"L{meta['level']}-{wall_names[meta['wall']]}",
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(-2, 98)
        ax.set_ylim(-2, 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (inches)', fontsize=11)
        ax.set_ylabel('Y (inches)', fontsize=11)
        ax.set_title(f'Plywood Sheet #{sheet_num + 1} (8\' × 4\' = 96" × 48") - Level {level}',
                    fontsize=13, fontweight='bold')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def generate_cutting_pdf(config, output_path='output/cutting_templates.pdf'):
    """Generate complete cutting template PDF"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    levels = sorted(set(s['level'] for s in config.shelves))

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = (
            f'Pantry Shelf Cutting Templates\n\n'
            f'Configuration: {config.version}\n'
            f'Pantry: {config.pantry["width"]}" × {config.pantry["depth"]}" × {config.pantry["height"]}"\n'
            f'Levels: {len(levels)}\n\n'
            f'Each level includes:\n'
            f'  1. Complete assembly view with dimensions\n'
            f'  2. Individual shelf pieces (cut apart at red lines)\n'
            f'  3. Plywood sheet layout (8\' × 4\' sheets)\n\n'
            f'Material: 1" Baltic Birch Plywood'
        )

        ax.text(0.5, 0.5, title_text,
               transform=ax.transAxes, fontsize=14, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Generate pages for each level
        for level in levels:
            print(f"Generating pages for level {level}...")

            # Page 1: Overview
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_overview_page(ax, config, level)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: Separated shelves
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_separated_shelves_page(ax, config, level)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 3+: Plywood layouts
            create_plywood_layout_pages(pdf, config, level)

    print(f"\nPDF generated: {output_path}")


if __name__ == '__main__':
    print("="*60)
    print("Generating Cutting Template PDF")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    print(f"Loading config: {config_path}\n")
    config = ShelfConfig.from_file(config_path)

    generate_cutting_pdf(config)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
