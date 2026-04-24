#!/usr/bin/env python3
"""
Generate PDF cutting templates for pantry shelves using correct tangent circle geometry.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    sinusoid_depth
)
from config import ShelfConfig


def compute_shelf_outline(config, level, wall):
    """
    Compute the complete outline for a single shelf including corner arcs.

    Args:
        config: ShelfConfig instance
        level: Shelf level number
        wall: Wall identifier ('E', 'S', or 'W')

    Returns:
        np.ndarray of [x, y] points forming complete closed outline
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

    # Generate the main sinusoid curve
    extent_start = shelf['extent_start']
    extent_end = shelf['extent_end']

    sinusoid_wall = generate_sinusoid_points(
        extent_start, extent_end, base_depth,
        amplitude, period, offset, num_points=200
    )

    # Convert to pantry coordinates
    sinusoid_pantry = np.array([
        wall_to_pantry_coords(pos, depth, wall, pantry_width, pantry_depth)
        for pos, depth in sinusoid_wall
    ])

    # For South wall, no corner modifications needed
    if wall == 'S':
        # Close the outline along the wall
        if wall == 'S':
            # South wall runs along x-axis at y=pantry_depth
            # Close by going back along y=pantry_depth
            outline = sinusoid_pantry
            # Add closing edge along the wall
            start_point = np.array([extent_start, pantry_depth])
            end_point = np.array([extent_end, pantry_depth])
            # The sinusoid already includes these endpoints, just close it
            return outline

    # For East and West walls, add corner arcs
    if wall not in ['E', 'W']:
        return sinusoid_pantry

    # Get neighboring shelves for corner solving
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        # Can't solve corners without all three shelves
        return sinusoid_pantry

    # Get all parameters for interior checking
    left_depth = config.get_base_depth('E')
    back_depth = config.get_base_depth('S')
    right_depth = config.get_base_depth('W')

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    left_params = (left_depth, amplitude, period, left_offset)
    right_params = (right_depth, amplitude, period, right_offset)
    back_params = (back_depth, amplitude, period, back_offset)

    radius_interior = config.design_params['interior_corner_radius']
    radius_door = config.design_params['door_corner_radius']

    # Compute depths at the two corners
    depth_at_north = sinusoid_depth(0, base_depth, amplitude, period, offset)
    depth_at_south = sinusoid_depth(pantry_depth, base_depth, amplitude, period, offset)

    # Door corner (North) - REMOVE material
    if wall == 'E':
        door_point_wall = np.array([0, 0])  # Wall at door
        door_point_shelf = np.array([depth_at_north, 0])  # Shelf edge at door
    else:  # 'W'
        door_point_wall = np.array([pantry_width, 0])
        door_point_shelf = np.array([pantry_width - depth_at_north, 0])

    # For door corner, use simple circle at corner
    # Center is at distance=radius from both edges
    if wall == 'E':
        door_center = np.array([radius_door, radius_door])
    else:  # 'W'
        door_center = np.array([pantry_width - radius_door, radius_door])

    door_arc = generate_circle_arc(
        door_center,
        door_point_wall,
        door_point_shelf,
        num_points=20,
        interior_arc=True
    )

    # Interior corner (South) - ADD material
    # Solve for tangent circle between this shelf and back shelf
    if wall == 'E':
        corner_type = 'left-back'
        wall1_type = 'L'
        wall2_type = 'B'
        pos1_init = pantry_depth - 10
        pos2_init = 10
    else:  # 'W'
        corner_type = 'right-back'
        wall1_type = 'R'
        wall2_type = 'B'
        pos1_init = pantry_depth - 10
        pos2_init = pantry_width - 10

    try:
        if wall == 'E':
            lb_center, lb_point1, lb_point2, lb_pos1, lb_pos2 = solve_tangent_circle_two_sinusoids(
                pos1_init=pos1_init,
                base_depth1=left_depth,
                amplitude1=amplitude,
                period1=period,
                offset1=left_offset,
                pos2_init=pos2_init,
                base_depth2=back_depth,
                amplitude2=amplitude,
                period2=period,
                offset2=back_offset,
                radius=radius_interior,
                wall1_type=wall1_type,
                wall2_type=wall2_type,
                pantry_width=pantry_width,
                pantry_depth=pantry_depth,
                corner_type=corner_type,
                left_params=left_params,
                right_params=right_params,
                back_params=back_params
            )
            interior_arc = generate_circle_arc(lb_center, lb_point1, lb_point2, num_points=20, interior_arc=True)
        else:  # 'W'
            rb_center, rb_point1, rb_point2, rb_pos1, rb_pos2 = solve_tangent_circle_two_sinusoids(
                pos1_init=pos1_init,
                base_depth1=right_depth,
                amplitude1=amplitude,
                period1=period,
                offset1=right_offset,
                pos2_init=pos2_init,
                base_depth2=back_depth,
                amplitude2=amplitude,
                period2=period,
                offset2=back_offset,
                radius=radius_interior,
                wall1_type=wall1_type,
                wall2_type=wall2_type,
                pantry_width=pantry_width,
                pantry_depth=pantry_depth,
                corner_type=corner_type,
                left_params=left_params,
                right_params=right_params,
                back_params=back_params
            )
            interior_arc = generate_circle_arc(rb_center, rb_point1, rb_point2, num_points=20, interior_arc=True)
    except Exception as e:
        print(f"Warning: Failed to solve interior corner for level {level}, wall {wall}: {e}")
        interior_arc = None

    # Assemble complete outline
    # Structure: door_arc -> sinusoid (trimmed) -> interior_arc -> wall_edge -> back to start

    # Trim sinusoid to avoid overlap with arcs
    trim_count = 5
    sinusoid_trimmed = sinusoid_pantry[trim_count:-trim_count] if len(sinusoid_pantry) > 2*trim_count else sinusoid_pantry

    # Build outline
    outline_points = []

    # Add door arc
    for point in door_arc:
        outline_points.append(point)

    # Add sinusoid
    for point in sinusoid_trimmed:
        outline_points.append(point)

    # Add interior arc if it exists
    if interior_arc is not None:
        for point in interior_arc:
            outline_points.append(point)

    # Close back to start along the wall
    if wall == 'E':
        outline_points.append(np.array([0, pantry_depth]))  # SW corner along wall
        outline_points.append(np.array([0, 0]))  # Back to NW corner
    else:  # 'W'
        outline_points.append(np.array([pantry_width, pantry_depth]))  # SE corner
        outline_points.append(np.array([pantry_width, 0]))  # Back to NE corner

    return np.array(outline_points)


def create_cutting_template(ax, config, level, wall):
    """
    Create a cutting template for a single shelf.

    Args:
        ax: Matplotlib axes
        config: ShelfConfig instance
        level: Shelf level number
        wall: Wall identifier
    """
    shelf = config.get_shelf(level, wall)
    if not shelf:
        ax.text(0.5, 0.5, f'No {wall} shelf at level {level}',
               transform=ax.transAxes, ha='center', va='center')
        return

    # Compute outline
    outline = compute_shelf_outline(config, level, wall)

    if outline is None:
        ax.text(0.5, 0.5, f'Could not generate outline',
               transform=ax.transAxes, ha='center', va='center')
        return

    # Draw the outline
    ax.plot(outline[:, 0], outline[:, 1], 'b-', linewidth=2, label='Cut line')
    ax.fill(outline[:, 0], outline[:, 1], alpha=0.3, color='lightblue', label='Shelf surface')

    # Add dimensions
    min_x, max_x = np.min(outline[:, 0]), np.max(outline[:, 0])
    min_y, max_y = np.min(outline[:, 1]), np.max(outline[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    # Width dimension line
    y_offset = min_y - 3
    ax.plot([min_x, max_x], [y_offset, y_offset], 'k-', linewidth=1)
    ax.plot([min_x, min_x], [y_offset - 0.5, y_offset + 0.5], 'k-', linewidth=1)
    ax.plot([max_x, max_x], [y_offset - 0.5, y_offset + 0.5], 'k-', linewidth=1)
    ax.text((min_x + max_x) / 2, y_offset - 1.5, f'{width:.2f}"',
           ha='center', va='top', fontsize=10, fontweight='bold')

    # Height dimension line
    x_offset = max_x + 3
    ax.plot([x_offset, x_offset], [min_y, max_y], 'k-', linewidth=1)
    ax.plot([x_offset - 0.5, x_offset + 0.5], [min_y, min_y], 'k-', linewidth=1)
    ax.plot([x_offset - 0.5, x_offset + 0.5], [max_y, max_y], 'k-', linewidth=1)
    ax.text(x_offset + 1.5, (min_y + max_y) / 2, f'{height:.2f}"',
           ha='left', va='center', fontsize=10, fontweight='bold', rotation=90)

    # Labels
    wall_names = {'E': 'East (Left)', 'S': 'South (Back)', 'W': 'West (Right)'}
    ax.set_xlabel('X (inches)', fontsize=11)
    ax.set_ylabel('Y (inches)', fontsize=11)
    ax.set_title(
        f'Level {level} - {wall_names[wall]} Wall Shelf\n'
        f'Height from floor: {shelf["height"]:.1f}"',
        fontsize=13, fontweight='bold'
    )

    # Add parameters
    base_depth = config.get_base_depth(wall)
    params_text = (
        f'Cut from 1" Baltic Birch Plywood\n\n'
        f'Base depth: {base_depth:.1f}"\n'
        f'Amplitude: {config.design_params["sinusoid_amplitude"]:.1f}"\n'
        f'Period: {config.design_params["sinusoid_period"]:.1f}"\n'
        f'Phase offset: {shelf["sinusoid_offset"]:.4f} rad\n'
        f'Interior corner radius: {config.design_params["interior_corner_radius"]:.1f}"\n'
        f'Door corner radius: {config.design_params["door_corner_radius"]:.1f}"'
    )

    ax.text(0.98, 0.98, params_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Set aspect and grid
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    # Set limits with margin
    margin = 5
    ax.set_xlim(min_x - margin, x_offset + 5)
    ax.set_ylim(y_offset - 3, max_y + margin)


def generate_pdf(config, output_path='cutting_templates.pdf'):
    """
    Generate PDF with cutting templates for all shelves.

    Args:
        config: ShelfConfig instance
        output_path: Path to output PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all levels
    levels = sorted(set(s['level'] for s in config.shelves))

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = (
            f'Pantry Shelf Cutting Templates\n\n'
            f'Configuration: {config.version}\n'
            f'Pantry dimensions: {config.pantry["width"]}" W × '
            f'{config.pantry["depth"]}" D × {config.pantry["height"]}" H\n\n'
            f'Number of levels: {len(levels)}\n'
            f'Total shelf sections: {len(config.shelves)}\n\n'
            f'Design Parameters:\n'
            f'  Shelf depths: East {config.design_params["shelf_base_depth_east"]}", '
            f'South {config.design_params["shelf_base_depth_south"]}", '
            f'West {config.design_params["shelf_base_depth_west"]}"\n'
            f'  Sinusoid: {config.design_params["sinusoid_amplitude"]}" amplitude, '
            f'{config.design_params["sinusoid_period"]}" period\n'
            f'  Material: 1" Baltic Birch Plywood\n'
            f'  Corner radii: {config.design_params["interior_corner_radius"]}" interior, '
            f'{config.design_params["door_corner_radius"]}" door\n\n'
            f'Cut each template from 1" thick Baltic birch plywood.\n'
            f'Follow the blue cut lines precisely.\n'
            f'Label each piece with level and wall position.'
        )

        ax.text(0.5, 0.5, title_text,
               transform=ax.transAxes,
               fontsize=12,
               ha='center',
               va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Generate template for each shelf
        for level in levels:
            for wall in ['E', 'S', 'W']:
                shelf = config.get_shelf(level, wall)
                if not shelf:
                    continue

                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)

                create_cutting_template(ax, config, level, wall)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

                print(f"  Added template: Level {level}, {wall} wall")

    print(f"\nPDF generated: {output_path}")
    print(f"  Total pages: {len(config.shelves) + 1}")


if __name__ == '__main__':
    print("="*60)
    print("Generating PDF Cutting Templates")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    print(f"Loading config: {config_path}")
    config = ShelfConfig.from_file(config_path)
    print()

    print("Creating cutting templates...")
    generate_pdf(config, 'output/cutting_templates.pdf')

    print()
    print("="*60)
    print("Done!")
    print("="*60)
