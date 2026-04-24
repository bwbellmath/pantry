#!/usr/bin/env python3
"""
Verify that extracted polygon geometry matches the visualization exactly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
from extract_and_export_geometry import extract_exact_shelf_geometries
from config import ShelfConfig
from geometry import generate_interior_mask


def verify_level(config, level):
    """Create overlay visualization to verify polygon extraction."""
    geom_data = extract_exact_shelf_geometries(config, level)

    if not geom_data:
        print(f"Failed to extract geometry for level {level}")
        return

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # LEFT: Original visualization
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Draw pantry on both
    for ax in [ax1, ax2]:
        pantry_outline = np.array([
            [0, 0], [pantry_width, 0], [pantry_width, pantry_depth],
            [0, pantry_depth], [0, 0]
        ])
        ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=2)

        X, Y, interior_mask = generate_interior_mask(
            left_depth, amplitude, period, left_offset,
            right_depth, amplitude, period, right_offset,
            back_depth, amplitude, period, back_offset,
            pantry_width, pantry_depth,
            resolution=150
        )
        ax.contourf(X, Y, ~interior_mask, levels=[0.5, 1.5], colors=['#FFE4B5'], alpha=0.3)

    # LEFT PANEL: Draw curves and arcs
    ax1.plot(geom_data['left_curve'][:, 0], geom_data['left_curve'][:, 1],
            'b-', linewidth=2, alpha=0.8, label='Sinusoids')
    ax1.plot(geom_data['back_curve'][:, 0], geom_data['back_curve'][:, 1],
            'r-', linewidth=2, alpha=0.8)
    ax1.plot(geom_data['right_curve'][:, 0], geom_data['right_curve'][:, 1],
            'g-', linewidth=2, alpha=0.8)
    ax1.plot(geom_data['lb_arc'][:, 0], geom_data['lb_arc'][:, 1],
            'c-', linewidth=3, alpha=0.9, label='Arcs')
    ax1.plot(geom_data['rb_arc'][:, 0], geom_data['rb_arc'][:, 1],
            'm-', linewidth=3, alpha=0.9)

    # Draw cut lines
    lb_cut_y = geom_data['lb_cut_y']
    rb_cut_y = geom_data['rb_cut_y']
    lb_arc_x = geom_data['lb_arc'][-1][0]
    rb_arc_x = geom_data['rb_arc'][-1][0]
    ax1.plot([0, lb_arc_x], [lb_cut_y, lb_cut_y], 'r--', linewidth=2, label='Cut lines')
    ax1.plot([rb_arc_x, pantry_width], [rb_cut_y, rb_cut_y], 'r--', linewidth=2)

    ax1.set_title('Original Visualization', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    # RIGHT PANEL: Draw extracted polygons with vertex labels
    colors = {'E': 'blue', 'S': 'red', 'W': 'green'}
    labels = {'E': 'Left shelf polygon', 'S': 'Back shelf polygon', 'W': 'Right shelf polygon'}

    for wall in ['E', 'S', 'W']:
        poly = geom_data[wall]
        ax2.plot(poly[:, 0], poly[:, 1], color=colors[wall], linewidth=2,
                label=labels[wall], marker='o', markersize=4)
        ax2.fill(poly[:, 0], poly[:, 1], color=colors[wall], alpha=0.2)

        # Add vertex labels with arrows - label every 10th point plus first/last few
        for i in range(len(poly)):
            if i < 3 or i >= len(poly) - 3 or i % 10 == 0:
                # Use different offsets based on position to avoid overlaps
                # Offset radially outward from polygon center
                center = np.mean(poly, axis=0)
                direction = poly[i] - center
                direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

                # Offset distance varies by wall to prevent overlaps
                offset_dist = 15 if wall == 'S' else 12
                offset = direction_norm * offset_dist

                ax2.annotate(f'{wall}{i}',
                           xy=poly[i],
                           xytext=offset,
                           textcoords='offset points',
                           fontsize=7,
                           color=colors[wall],
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   edgecolor=colors[wall],
                                   alpha=0.9,
                                   linewidth=1.5),
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3,rad=0',
                                         color=colors[wall],
                                         linewidth=1.5,
                                         alpha=0.7))

    ax2.set_title('Extracted Polygons', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')

    # Set same limits for both
    for ax in [ax1, ax2]:
        ax.set_xlim(-2, pantry_width + 2)
        ax.set_ylim(-2, pantry_depth + 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (inches)')
        ax.set_ylabel('Y (inches)')

    plt.suptitle(f'Level {level} Geometry Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(f'output/verify_level_{level}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved verification: {output_path}")
    plt.close()


if __name__ == '__main__':
    config_path = Path('configs/pantry_0002.json')
    config = ShelfConfig.from_file(config_path)

    levels = sorted(set(s['level'] for s in config.shelves))

    print("Generating verification images...")
    for level in levels:
        verify_level(config, level)

    print("\nVerification images saved to output/verify_level_*.png")
    print("Compare left (original) with right (extracted polygons)")
