#!/usr/bin/env python3
"""
Generate complete pantry shelf layout with tangent circle corners.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    generate_interior_mask
)
from config import ShelfConfig


def generate_shelf_config(num_levels=4):
    """Generate a shelf configuration with multiple levels."""
    config = ShelfConfig()
    config.generate_shelf_entries(num_levels=num_levels, randomize=True, seed=42)

    # Save to file
    config_dir = Path('configs')
    version = config.get_next_version_number(config_dir)
    config.version = version

    config_path = config_dir / f'pantry_{version}.json'
    config.to_file(config_path)

    print(f"Generated config: {config_path}")
    print(f"  Levels: {num_levels}")
    print(f"  Total shelves: {len(config.shelves)}")

    return config


def compute_corner_arcs(config, level):
    """Compute tangent circle arcs for a specific shelf level."""
    # Get the three shelves at this level
    left_shelf = config.get_shelf(level, 'E')  # Still using old keys internally
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        print(f"Warning: Missing shelves for level {level}")
        return None, None

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    radius = config.design_params['door_corner_radius']

    # Get depths and offsets
    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Prepare shelf parameters for interior checking
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
    except Exception as e:
        print(f"Warning: Failed to solve left-back corner for level {level}: {e}")
        lb_arc = None

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
    except Exception as e:
        print(f"Warning: Failed to solve right-back corner for level {level}: {e}")
        rb_arc = None

    # Also return the intersection points for cut line calculation
    lb_points = (lb_point1, lb_point2) if 'lb_point1' in locals() else None
    rb_points = (rb_point1, rb_point2) if 'rb_point1' in locals() else None

    return lb_arc, rb_arc, lb_points, rb_points


def plot_shelf_level(ax, config, level, show_arcs=True):
    """Plot a single shelf level."""
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        return

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']

    # Get parameters
    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Generate sinusoid curves
    left_curve_wall = generate_sinusoid_points(0, pantry_depth, left_depth,
                                               amplitude, period, left_offset, num_points=200)
    back_curve_wall = generate_sinusoid_points(0, pantry_width, back_depth,
                                               amplitude, period, back_offset, num_points=200)
    right_curve_wall = generate_sinusoid_points(0, pantry_depth, right_depth,
                                                amplitude, period, right_offset, num_points=200)

    # Convert to pantry coordinates
    left_curve = np.array([wall_to_pantry_coords(pos, depth, 'E', pantry_width, pantry_depth)
                           for pos, depth in left_curve_wall])
    back_curve = np.array([wall_to_pantry_coords(pos, depth, 'S', pantry_width, pantry_depth)
                           for pos, depth in back_curve_wall])
    right_curve = np.array([wall_to_pantry_coords(pos, depth, 'W', pantry_width, pantry_depth)
                            for pos, depth in right_curve_wall])

    # Plot curves
    color = plt.cm.viridis(level / 4)  # Color by level
    ax.plot(left_curve[:, 0], left_curve[:, 1], color=color, linewidth=2, alpha=0.8)
    ax.plot(back_curve[:, 0], back_curve[:, 1], color=color, linewidth=2, alpha=0.8)
    ax.plot(right_curve[:, 0], right_curve[:, 1], color=color, linewidth=2, alpha=0.8)

    # Plot corner arcs if requested
    if show_arcs:
        lb_arc, rb_arc, lb_points, rb_points = compute_corner_arcs(config, level)
        if lb_arc is not None:
            ax.plot(lb_arc[:, 0], lb_arc[:, 1], color=color, linewidth=3, alpha=0.9)
        if rb_arc is not None:
            ax.plot(rb_arc[:, 0], rb_arc[:, 1], color=color, linewidth=3, alpha=0.9)

        # Draw horizontal cut lines at average y-coordinate of arc endpoints
        if lb_points is not None:
            lb_y_avg = (lb_points[0][1] + lb_points[1][1]) / 2
            ax.plot([0, pantry_width], [lb_y_avg, lb_y_avg], 'r--', linewidth=2, alpha=0.7, label='Cut line (left-back)')
        if rb_points is not None:
            rb_y_avg = (rb_points[0][1] + rb_points[1][1]) / 2
            ax.plot([0, pantry_width], [rb_y_avg, rb_y_avg], 'r--', linewidth=2, alpha=0.7, label='Cut line (right-back)')


def visualize_all_shelves(config, output_file='all_shelves.png'):
    """Visualize all shelf levels."""
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw pantry outline
    pantry_outline = np.array([
        [0, 0],
        [pantry_width, 0],
        [pantry_width, pantry_depth],
        [0, pantry_depth],
        [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=3, label='Pantry walls')

    # Get number of levels
    levels = sorted(set(s['level'] for s in config.shelves))

    # Plot each level
    for level in levels:
        plot_shelf_level(ax, config, level, show_arcs=True)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', linewidth=3, label='Pantry walls')]
    for level in levels:
        color = plt.cm.viridis(level / len(levels))
        height = config.get_shelf(level, 'E')['height']
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2,
                                     label=f'Level {level} (h={height:.1f}")'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches) - Left to Right', fontsize=12)
    ax.set_ylabel('Y (inches) - Front (door) to Back', fontsize=12)
    ax.set_title('Complete Pantry Shelf Layout with Tangent Circle Corners', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()


def visualize_single_level(config, level, output_file=None):
    """Visualize a single shelf level in detail."""
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']

    if output_file is None:
        output_file = f'shelf_level_{level}.png'

    fig, ax = plt.subplots(figsize=(14, 12))

    # Get sinusoid parameters
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    # Draw pantry outline
    pantry_outline = np.array([
        [0, 0],
        [pantry_width, 0],
        [pantry_width, pantry_depth],
        [0, pantry_depth],
        [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=3, label='Pantry walls')

    # Plot the shelf level
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    height = left_shelf['height']

    # Get sinusoid offsets
    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Generate and plot interior/exterior shading
    print(f"  Generating interior/exterior mask for level {level}...")
    X, Y, interior_mask = generate_interior_mask(
        left_depth, amplitude, period, left_offset,
        right_depth, amplitude, period, right_offset,
        back_depth, amplitude, period, back_offset,
        pantry_width, pantry_depth,
        resolution=150
    )

    # Shade the shelf regions (exterior = False in mask)
    ax.contourf(X, Y, ~interior_mask, levels=[0.5, 1.5], colors=['#FFE4B5'], alpha=0.4)
    # Shade the interior region
    ax.contourf(X, Y, interior_mask, levels=[0.5, 1.5], colors=['#E0F7FA'], alpha=0.3)

    # Generate and plot curves
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

    ax.plot(left_curve[:, 0], left_curve[:, 1], 'b-', linewidth=3, label='Left shelf', alpha=0.8)
    ax.plot(back_curve[:, 0], back_curve[:, 1], 'r-', linewidth=3, label='Back shelf', alpha=0.8)
    ax.plot(right_curve[:, 0], right_curve[:, 1], 'g-', linewidth=3, label='Right shelf', alpha=0.8)

    # Compute and plot corner arcs
    lb_arc, rb_arc, lb_points, rb_points = compute_corner_arcs(config, level)
    if lb_arc is not None:
        ax.plot(lb_arc[:, 0], lb_arc[:, 1], 'c-', linewidth=4, label='Left-back arc', alpha=0.9)
    if rb_arc is not None:
        ax.plot(rb_arc[:, 0], rb_arc[:, 1], 'm-', linewidth=4, label='Right-back arc', alpha=0.9)

    # Draw horizontal cut lines at average y-coordinate of arc endpoints
    if lb_points is not None:
        lb_y_avg = (lb_points[0][1] + lb_points[1][1]) / 2
        ax.plot([0, pantry_width], [lb_y_avg, lb_y_avg], 'r--', linewidth=2, alpha=0.7, label='Cut line (left-back)')
    if rb_points is not None:
        rb_y_avg = (rb_points[0][1] + rb_points[1][1]) / 2
        ax.plot([0, pantry_width], [rb_y_avg, rb_y_avg], 'r--', linewidth=2, alpha=0.7, label='Cut line (right-back)')

    # Create legend with shading info
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E0F7FA', alpha=0.3, label='Interior (usable space)'),
        Patch(facecolor='#FFE4B5', alpha=0.4, label='Shelf regions'),
        plt.Line2D([0], [0], color='b', linewidth=3, label='Left shelf'),
        plt.Line2D([0], [0], color='r', linewidth=3, label='Back shelf'),
        plt.Line2D([0], [0], color='g', linewidth=3, label='Right shelf'),
        plt.Line2D([0], [0], color='c', linewidth=4, label='Left-back arc'),
        plt.Line2D([0], [0], color='m', linewidth=4, label='Right-back arc'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches) - Left to Right', fontsize=12)
    ax.set_ylabel('Y (inches) - Front (door) to Back', fontsize=12)
    ax.set_title(f'Shelf Level {level} (Height: {height:.1f}") - Detailed View',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved level {level} visualization to {output_file}")
    plt.close()


if __name__ == '__main__':
    # Load existing configuration instead of generating new one
    print("="*60)
    print("Generating Pantry Shelves")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    if config_path.exists():
        print(f"Loading existing config: {config_path}")
        config = ShelfConfig.from_file(config_path)
    else:
        print("Generating new configuration...")
        config = generate_shelf_config(num_levels=4)
    print()

    # Visualize all shelves
    print("Creating visualization of all shelves...")
    visualize_all_shelves(config, 'all_shelves.png')
    print()

    # Visualize individual levels
    print("Creating detailed visualizations for each level...")
    for level in range(4):
        visualize_single_level(config, level)

    print()
    print("="*60)
    print("Done! Generated:")
    print("  - configs/pantry_XXXX.json (configuration file)")
    print("  - all_shelves.png (overview)")
    print("  - shelf_level_0.png through shelf_level_3.png (details)")
    print("="*60)
