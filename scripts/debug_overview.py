#!/usr/bin/env python3
"""
Debug script to visualize shelf geometry.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
import shelf_generator

def visualize_level(config_path: Path, level: int):
    """Visualize a single level with all shelves."""

    # Load config
    cfg = config.ShelfConfig.from_file(config_path)
    gen = shelf_generator.ShelfGenerator(cfg)

    # Get footprints for this level
    footprints = gen.get_footprints_by_level(level, num_points=100)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw pantry outline
    pantry_width = cfg.pantry['width']
    pantry_depth = cfg.pantry['depth']

    pantry_rect = patches.Rectangle(
        (0, 0), pantry_width, pantry_depth,
        linewidth=2, edgecolor='black', facecolor='none',
        linestyle='--', label='Pantry walls'
    )
    ax.add_patch(pantry_rect)

    # Draw door
    door_clearance_east = cfg.pantry.get('door_clearance_east', 6.0)
    door_clearance_west = cfg.pantry.get('door_clearance_west', 4.0)
    door_width = pantry_width - door_clearance_east - door_clearance_west
    door_rect = patches.Rectangle(
        (door_clearance_east, -2), door_width, 2,
        linewidth=1, edgecolor='brown', facecolor='lightgray',
        label='Door'
    )
    ax.add_patch(door_rect)

    # Draw each shelf
    colors = {'E': 'red', 'S': 'green', 'W': 'blue'}
    for footprint in footprints:
        outline = footprint.outline_points
        color = colors[footprint.wall]
        wall_name = {'E': 'East (Left)', 'S': 'South (Back)', 'W': 'West (Right)'}[footprint.wall]

        # Plot outline
        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=2,
               label=wall_name, marker='o', markersize=2)
        # Close the polygon
        ax.plot([outline[-1, 0], outline[0, 0]],
               [outline[-1, 1], outline[0, 1]], color=color, linewidth=2,
               marker='o', markersize=2)
        ax.fill(outline[:, 0], outline[:, 1], alpha=0.2, color=color)

        # Mark first and last points
        ax.plot(outline[0, 0], outline[0, 1], 'ko', markersize=8, label=f'{footprint.wall} start')
        ax.plot(outline[-1, 0], outline[-1, 1], 'kx', markersize=8, label=f'{footprint.wall} end')

    # Set axis properties
    ax.set_xlim(-5, pantry_width + 5)
    ax.set_ylim(-5, pantry_depth + 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=12)
    ax.set_ylabel('Y (inches)', fontsize=12)
    ax.set_title(f'Level {level} - Height: {footprints[0].shelf_data["height"]:.1f}"',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)

    # Add coordinate system annotation
    ax.annotate('Origin (0,0)\nNW Corner', xy=(0, 0), xytext=(10, 10),
               textcoords='offset points', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    output_path = Path('output/renders') / f'debug_level_{level}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    config_path = Path('configs/pantry_0002.json')

    # Visualize all 4 levels
    for level in range(4):
        visualize_level(config_path, level)

    print("\nDone! Check output/renders/ for debug_level_*.png files")
