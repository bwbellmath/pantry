#!/usr/bin/env python3
"""
Generate final cutting templates with:
- Proper closed polygon shapes for each shelf
- SVG export for each piece
- Optimized 2D bin packing on 8'x4' plywood sheets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
import svgwrite
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    sinusoid_depth,
    generate_interior_mask
)
from config import ShelfConfig


class ShelfPiece:
    """Represents a single shelf piece with its geometry."""

    def __init__(self, polygon, level, wall, metadata):
        """
        Args:
            polygon: np.ndarray of [x, y] points forming closed polygon
            level: Shelf level number
            wall: Wall identifier ('E', 'S', 'W')
            metadata: Dictionary with additional info
        """
        self.polygon = polygon
        self.level = level
        self.wall = wall
        self.metadata = metadata

        # Compute bounding box
        self.min_x = np.min(polygon[:, 0])
        self.max_x = np.max(polygon[:, 0])
        self.min_y = np.min(polygon[:, 1])
        self.max_y = np.max(polygon[:, 1])
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y

    def get_centered_polygon(self):
        """Return polygon centered at origin."""
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2
        return self.polygon - np.array([center_x, center_y])

    def get_normalized_polygon(self):
        """Return polygon with min corner at origin."""
        return self.polygon - np.array([self.min_x, self.min_y])

    def export_svg(self, filepath):
        """Export shelf piece as SVG file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add margin
        margin = 5
        width = self.width + 2 * margin
        height = self.height + 2 * margin

        dwg = svgwrite.Drawing(str(filepath), size=(f'{width}in', f'{height}in'),
                              viewBox=f'{self.min_x - margin} {self.min_y - margin} {width} {height}')

        # Convert polygon to SVG path
        points = [(p[0], p[1]) for p in self.polygon]
        dwg.add(dwg.polygon(points, fill='lightblue', stroke='black', stroke_width=0.1))

        # Add dimensions text
        dwg.add(dwg.text(f'Level {self.level} - Wall {self.wall}',
                        insert=(self.min_x, self.min_y - 2),
                        font_size='2', font_family='Arial'))
        dwg.add(dwg.text(f'{self.width:.2f}" × {self.height:.2f}"',
                        insert=(self.min_x, self.min_y - 0.5),
                        font_size='1.5', font_family='Arial'))

        dwg.save()


def compute_shelf_geometries(config, level):
    """
    Compute the complete closed polygon geometries for all three shelf pieces.

    Returns:
        dict with keys 'E', 'S', 'W' containing ShelfPiece objects
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

    # Solve for corner arcs and intersection points
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
        print(f"Warning: Failed to solve left-back corner: {e}")
        return None

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
        print(f"Warning: Failed to solve right-back corner: {e}")
        return None

    # Generate sinusoid curves
    left_curve_wall = generate_sinusoid_points(0, pantry_depth, left_depth,
                                               amplitude, period, left_offset, num_points=200)
    back_curve_wall = generate_sinusoid_points(0, pantry_width, back_depth,
                                               amplitude, period, back_offset, num_points=200)
    right_curve_wall = generate_sinusoid_points(0, pantry_depth, right_depth,
                                                amplitude, period, right_offset, num_points=200)

    left_curve = np.array([wall_to_pantry_coords(pos, depth, 'E', pantry_width, pantry_depth)
                          for pos, depth in left_curve_wall])
    back_curve = np.array([wall_to_pantry_coords(pos, depth, 'S', pantry_width, pantry_depth)
                          for pos, depth in back_curve_wall])
    right_curve = np.array([wall_to_pantry_coords(pos, depth, 'W', pantry_width, pantry_depth)
                           for pos, depth in right_curve_wall])

    # Build LEFT (East) shelf polygon
    # Start at NW corner (0, 0), go along sinusoid to cut point, then back along wall
    left_polygon = []
    left_polygon.append([0, 0])  # NW corner (door)

    # Add sinusoid points up to cut line
    for point in left_curve:
        if point[1] <= lb_cut_y:
            left_polygon.append(point)
        else:
            break

    # Find arc points that are on the left side (before cut)
    for point in lb_arc:
        if point[1] <= lb_cut_y:
            left_polygon.append(point)

    # Add horizontal cut line back to wall
    # Find the rightmost x at the cut line
    last_x = left_polygon[-1][0] if left_polygon else 0
    left_polygon.append([last_x, lb_cut_y])
    left_polygon.append([0, lb_cut_y])  # Back to wall

    left_piece = ShelfPiece(
        np.array(left_polygon),
        level, 'E',
        {'height': left_shelf['height'], 'cut_y': lb_cut_y}
    )

    # Build RIGHT (West) shelf polygon
    right_polygon = []
    right_polygon.append([pantry_width, 0])  # NE corner (door)

    # Add sinusoid points up to cut line
    for point in right_curve:
        if point[1] <= rb_cut_y:
            right_polygon.append(point)
        else:
            break

    # Find arc points that are on the right side (before cut)
    for point in rb_arc:
        if point[1] <= rb_cut_y:
            right_polygon.append(point)

    # Add horizontal cut line back to wall
    last_x = right_polygon[-1][0] if right_polygon else pantry_width
    right_polygon.append([last_x, rb_cut_y])
    right_polygon.append([pantry_width, rb_cut_y])  # Back to wall

    right_piece = ShelfPiece(
        np.array(right_polygon),
        level, 'W',
        {'height': right_shelf['height'], 'cut_y': rb_cut_y}
    )

    # Build BACK (South) shelf polygon
    back_polygon = []

    # Start at left edge on cut line
    back_polygon.append([0, lb_cut_y])

    # Add left corner arc points after the cut
    for point in lb_arc:
        if point[1] >= lb_cut_y:
            back_polygon.append(point)

    # Add back sinusoid curve
    back_polygon.extend(back_curve)

    # Add right corner arc points after the cut
    for point in rb_arc:
        if point[1] >= rb_cut_y:
            back_polygon.append(point)

    # Close along right edge back to cut line
    back_polygon.append([pantry_width, rb_cut_y])

    # Close along cut line (approximately - use straight line for simplicity)
    back_polygon.append([0, lb_cut_y])

    back_piece = ShelfPiece(
        np.array(back_polygon),
        level, 'S',
        {'height': back_shelf['height'], 'cut_y_left': lb_cut_y, 'cut_y_right': rb_cut_y}
    )

    return {
        'E': left_piece,
        'S': back_piece,
        'W': right_piece,
        'lb_cut_y': lb_cut_y,
        'rb_cut_y': rb_cut_y,
        'lb_arc': lb_arc,
        'rb_arc': rb_arc
    }


def simple_2d_bin_pack(pieces, bin_width=96, bin_height=48):
    """
    Simple 2D bin packing using First Fit Decreasing Height (FFDH).

    Args:
        pieces: List of ShelfPiece objects
        bin_width: Width of bin (8' = 96")
        bin_height: Height of bin (4' = 48")

    Returns:
        List of (bin_num, piece, x, y, rotated) tuples
    """
    # Sort pieces by height (descending)
    sorted_pieces = sorted(pieces, key=lambda p: p.height, reverse=True)

    bins = []  # List of bins, each bin is a list of (piece, x, y, rotated)
    current_bin = []
    current_bin_num = 0

    # Track rows in current bin
    rows = []  # List of (y_position, height, remaining_width)

    for piece in sorted_pieces:
        placed = False

        # Try to place in existing rows
        for row_idx, (row_y, row_height, row_width_used) in enumerate(rows):
            remaining_width = bin_width - row_width_used

            # Try normal orientation
            if piece.width <= remaining_width and piece.height <= row_height:
                x = row_width_used
                y = row_y
                current_bin.append((piece, x, y, False))
                rows[row_idx] = (row_y, row_height, row_width_used + piece.width + 1)  # 1" spacing
                placed = True
                break

            # Try rotated orientation (90 degrees)
            if piece.height <= remaining_width and piece.width <= row_height:
                x = row_width_used
                y = row_y
                current_bin.append((piece, x, y, True))
                rows[row_idx] = (row_y, row_height, row_width_used + piece.height + 1)  # 1" spacing
                placed = True
                break

        if not placed:
            # Try to create new row in current bin
            if rows:
                # Find top of current rows
                next_y = max(r[0] + r[1] for r in rows) + 1  # 1" spacing
            else:
                next_y = 0

            # Try normal orientation in new row
            if piece.width <= bin_width and next_y + piece.height <= bin_height:
                current_bin.append((piece, 0, next_y, False))
                rows.append((next_y, piece.height, piece.width + 1))
                placed = True

            # Try rotated in new row
            elif piece.height <= bin_width and next_y + piece.width <= bin_height:
                current_bin.append((piece, 0, next_y, True))
                rows.append((next_y, piece.width, piece.height + 1))
                placed = True

        if not placed:
            # Start new bin
            bins.append(current_bin)
            current_bin = [(piece, 0, 0, False)]
            rows = [(0, piece.height, piece.width + 1)]
            current_bin_num += 1

    # Add last bin
    if current_bin:
        bins.append(current_bin)

    # Format output
    result = []
    for bin_num, bin_contents in enumerate(bins):
        for piece, x, y, rotated in bin_contents:
            result.append((bin_num, piece, x, y, rotated))

    return result


def create_overview_page_with_cuts(ax, config, level, shelf_data):
    """Create overview page with proper cut lines."""
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

    # Generate interior mask
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

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

    # Draw shelf curves
    for wall, color, label in [('E', 'blue', 'Left'), ('S', 'red', 'Back'), ('W', 'green', 'Right')]:
        shelf = config.get_shelf(level, wall)
        depth = config.get_base_depth(wall)
        offset = shelf['sinusoid_offset']

        extent_end = pantry_depth if wall in ['E', 'W'] else pantry_width
        curve_wall = generate_sinusoid_points(0, extent_end, depth, amplitude, period, offset, num_points=200)
        curve = np.array([wall_to_pantry_coords(pos, depth, wall, pantry_width, pantry_depth)
                         for pos, depth in curve_wall])

        ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=3, label=f'{label} shelf', alpha=0.8)

    # Draw corner arcs
    if shelf_data['lb_arc'] is not None:
        ax.plot(shelf_data['lb_arc'][:, 0], shelf_data['lb_arc'][:, 1],
               'cyan', linewidth=4, label='Corner arcs', alpha=0.9)
    if shelf_data['rb_arc'] is not None:
        ax.plot(shelf_data['rb_arc'][:, 0], shelf_data['rb_arc'][:, 1],
               'magenta', linewidth=4, alpha=0.9)

    # Draw CORRECT cut lines - only from arc to wall
    lb_cut_y = shelf_data['lb_cut_y']
    rb_cut_y = shelf_data['rb_cut_y']

    # Find x-coordinate where left arc intersects cut line
    lb_arc_x = None
    for point in shelf_data['lb_arc']:
        if abs(point[1] - lb_cut_y) < 0.5:  # Within tolerance
            lb_arc_x = point[0]
            break
    if lb_arc_x:
        ax.plot([0, lb_arc_x], [lb_cut_y, lb_cut_y], 'r--', linewidth=2, alpha=0.8, label='Cut lines')

    # Find x-coordinate where right arc intersects cut line
    rb_arc_x = None
    for point in shelf_data['rb_arc']:
        if abs(point[1] - rb_cut_y) < 0.5:  # Within tolerance
            rb_arc_x = point[0]
            break
    if rb_arc_x:
        ax.plot([rb_arc_x, pantry_width], [rb_cut_y, rb_cut_y], 'r--', linewidth=2, alpha=0.8)

    height = left_shelf['height']
    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=12)
    ax.set_ylabel('Y (inches)', fontsize=12)
    ax.set_title(f'Level {level} - Height: {height:.1f}" - Assembly View',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)


def create_plywood_layout_page(ax, bin_num, placements, bin_width=96, bin_height=48):
    """Create plywood sheet layout page."""
    # Draw sheet outline
    sheet_rect = mpatches.Rectangle(
        (0, 0), bin_width, bin_height,
        linewidth=3, edgecolor='black', facecolor='lightgray', alpha=0.2
    )
    ax.add_patch(sheet_rect)

    colors = {'E': 'blue', 'S': 'red', 'W': 'green'}

    for bin_n, piece, x, y, rotated in placements:
        if bin_n != bin_num:
            continue

        # Get polygon
        poly = piece.get_normalized_polygon()

        if rotated:
            # Rotate 90 degrees counterclockwise
            rot_matrix = np.array([[0, -1], [1, 0]])
            poly = poly @ rot_matrix.T
            # Re-normalize
            poly = poly - np.array([np.min(poly[:, 0]), np.min(poly[:, 1])])

        # Translate to position
        poly_placed = poly + np.array([x, y])

        # Draw
        color = colors.get(piece.wall, 'gray')
        ax.plot(poly_placed[:, 0], poly_placed[:, 1], color=color, linewidth=2)
        ax.fill(poly_placed[:, 0], poly_placed[:, 1], color=color, alpha=0.3)

        # Add label
        center_x = np.mean(poly_placed[:, 0])
        center_y = np.mean(poly_placed[:, 1])
        wall_names = {'E': 'L', 'S': 'B', 'W': 'R'}
        label = f"L{piece.level}-{wall_names[piece.wall]}"
        if rotated:
            label += "↻"

        ax.text(center_x, center_y, label,
               fontsize=10, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlim(-2, bin_width + 2)
    ax.set_ylim(-2, bin_height + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=11)
    ax.set_ylabel('Y (inches)', fontsize=11)
    ax.set_title(f'Plywood Sheet #{bin_num + 1} (8\' × 4\' = 96" × 48")',
                fontsize=13, fontweight='bold')


def generate_final_templates(config, output_dir='output'):
    """Generate all final templates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = sorted(set(s['level'] for s in config.shelves))

    # Collect all shelf pieces
    all_pieces = []
    all_shelf_data = {}

    for level in levels:
        print(f"Processing level {level}...")
        shelf_data = compute_shelf_geometries(config, level)

        if not shelf_data:
            print(f"  Warning: Could not compute geometries for level {level}")
            continue

        all_shelf_data[level] = shelf_data

        # Export SVG for each piece
        for wall in ['E', 'S', 'W']:
            piece = shelf_data[wall]
            svg_path = output_dir / f'shelf_L{level}_{wall}.svg'
            piece.export_svg(svg_path)
            print(f"  Exported SVG: {svg_path}")
            all_pieces.append(piece)

    # Optimize packing
    print("\nOptimizing plywood layout...")
    placements = simple_2d_bin_pack(all_pieces)

    num_sheets = max(p[0] for p in placements) + 1
    print(f"  Packed {len(all_pieces)} pieces onto {num_sheets} sheet(s)")

    # Generate PDF
    pdf_path = output_dir / 'final_cutting_templates.pdf'
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = (
            f'Pantry Shelf Cutting Templates - FINAL\n\n'
            f'Configuration: {config.version}\n'
            f'Pantry: {config.pantry["width"]}" × {config.pantry["depth"]}" × {config.pantry["height"]}"\n'
            f'Levels: {len(levels)}\n'
            f'Total pieces: {len(all_pieces)}\n'
            f'Plywood sheets needed: {num_sheets}\n\n'
            f'Material: 1" Baltic Birch Plywood\n'
            f'Corner radius: 3"\n\n'
            f'SVG files exported for each piece\n'
            f'See plywood layouts for cutting arrangement'
        )

        ax.text(0.5, 0.5, title_text,
               transform=ax.transAxes, fontsize=13, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Overview pages for each level
        for level in levels:
            if level not in all_shelf_data:
                continue

            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_overview_page_with_cuts(ax, config, level, all_shelf_data[level])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Plywood layout pages
        for sheet_num in range(num_sheets):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_plywood_layout_page(ax, sheet_num, placements)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"\nPDF generated: {pdf_path}")
    print(f"\nAll done!")


if __name__ == '__main__':
    print("="*60)
    print("Generating Final Cutting Templates")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    print(f"Loading config: {config_path}\n")
    config = ShelfConfig.from_file(config_path)

    generate_final_templates(config)

    print("\n" + "="*60)
