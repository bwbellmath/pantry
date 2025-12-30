#!/usr/bin/env python3
"""
Test script for visualizing tangent circles between two sinusoid curves.
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
    wall_to_pantry_coords
)
from config import ShelfConfig


def test_left_back_corner():
    """Test tangent circle for left-back corner."""
    # Load or create config
    config = ShelfConfig()

    # Get parameters
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    radius = config.design_params['door_corner_radius']

    # Left wall parameters
    base_depth_left = config.design_params['shelf_base_depth_east']
    offset_left = 0.0  # No phase offset for testing

    # Back wall parameters
    base_depth_back = config.design_params['shelf_base_depth_south']
    offset_back = np.pi / 4  # Some phase offset

    print(f"Pantry dimensions: {pantry_width}\" × {pantry_depth}\"")
    print(f"Left wall base depth: {base_depth_left}\"")
    print(f"Back wall base depth: {base_depth_back}\"")
    print(f"Corner radius: {radius}\"")
    print(f"Sinusoid: amplitude={amplitude}\", period={period}\"")
    print()

    # Initial guesses for positions where circle might be tangent
    # For left-back corner, we expect it near the back (high y for left, low x for back)
    pos1_init = pantry_depth - 10  # Position along left wall (y-axis), near back
    pos2_init = 10  # Position along back wall (x-axis), near left side

    print("Solving for tangent circle...")
    center, point1, point2, pos1, pos2 = solve_tangent_circle_two_sinusoids(
        pos1_init=pos1_init,
        base_depth1=base_depth_left,
        amplitude1=amplitude,
        period1=period,
        offset1=offset_left,
        pos2_init=pos2_init,
        base_depth2=base_depth_back,
        amplitude2=amplitude,
        period2=period,
        offset2=offset_back,
        radius=radius,
        wall1_type='L',
        wall2_type='B',
        pantry_width=pantry_width,
        pantry_depth=pantry_depth,
        corner_type='left-back'
    )

    print(f"Circle center: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"Tangent point on left wall: ({point1[0]:.3f}, {point1[1]:.3f}) at position {pos1:.3f}")
    print(f"Tangent point on back wall: ({point2[0]:.3f}, {point2[1]:.3f}) at position {pos2:.3f}")

    # Verify distances
    dist1 = np.linalg.norm(point1 - center)
    dist2 = np.linalg.norm(point2 - center)
    print(f"Distance from center to point1: {dist1:.6f}\" (should be {radius}\")")
    print(f"Distance from center to point2: {dist2:.6f}\" (should be {radius}\")")
    print()

    # Generate the arc
    arc_points = generate_circle_arc(center, point1, point2, num_points=50, interior_arc=True)

    # Generate the full sinusoid curves for visualization
    left_curve_wall = generate_sinusoid_points(0, pantry_depth, base_depth_left,
                                               amplitude, period, offset_left, num_points=200)
    back_curve_wall = generate_sinusoid_points(0, pantry_width, base_depth_back,
                                                amplitude, period, offset_back, num_points=200)

    # Convert to pantry coordinates
    left_curve_pantry = np.array([
        wall_to_pantry_coords(pos, depth, 'E', pantry_width, pantry_depth)
        for pos, depth in left_curve_wall
    ])
    back_curve_pantry = np.array([
        wall_to_pantry_coords(pos, depth, 'S', pantry_width, pantry_depth)
        for pos, depth in back_curve_wall
    ])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw pantry outline
    pantry_outline = np.array([
        [0, 0],
        [pantry_width, 0],
        [pantry_width, pantry_depth],
        [0, pantry_depth],
        [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=2, label='Pantry walls')

    # Draw sinusoid curves
    ax.plot(left_curve_pantry[:, 0], left_curve_pantry[:, 1], 'b-', linewidth=2, label='Left wall shelf')
    ax.plot(back_curve_pantry[:, 0], back_curve_pantry[:, 1], 'r-', linewidth=2, label='Back wall shelf')

    # Draw tangent circle arc
    ax.plot(arc_points[:, 0], arc_points[:, 1], 'g-', linewidth=3, label=f'Tangent arc (R={radius}")')

    # Draw circle center
    ax.plot(center[0], center[1], 'go', markersize=8, label='Circle center')

    # Draw tangent points
    ax.plot(point1[0], point1[1], 'bs', markersize=10, label='Tangent pt (Left)')
    ax.plot(point2[0], point2[1], 'rs', markersize=10, label='Tangent pt (Back)')

    # Draw radii to tangent points
    ax.plot([center[0], point1[0]], [center[1], point1[1]], 'g--', alpha=0.5)
    ax.plot([center[0], point2[0]], [center[1], point2[1]], 'g--', alpha=0.5)

    # Draw full circle for reference (dotted)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'g:', alpha=0.3, linewidth=1)

    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlabel('X (inches) - Left to Right')
    ax.set_ylabel('Y (inches) - Front (door) to Back')
    ax.set_title('Tangent Circle Solution: Left-Back Corner')

    plt.tight_layout()
    plt.savefig('tangent_circle_left_back.png', dpi=150)
    print("Saved visualization to tangent_circle_left_back.png")
    plt.close()


def test_right_back_corner():
    """Test tangent circle for right-back corner."""
    # Load or create config
    config = ShelfConfig()

    # Get parameters
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    radius = config.design_params['door_corner_radius']

    # Right wall parameters
    base_depth_right = config.design_params['shelf_base_depth_west']
    offset_right = np.pi / 2

    # Back wall parameters
    base_depth_back = config.design_params['shelf_base_depth_south']
    offset_back = np.pi / 4

    print("\n" + "="*60)
    print("Testing Right-Back corner")
    print("="*60)
    print(f"Right wall base depth: {base_depth_right}\"")
    print(f"Back wall base depth: {base_depth_back}\"")

    # Initial guesses
    pos1_init = pantry_depth - 10  # Position along right wall, near back
    pos2_init = pantry_width - 10  # Position along back wall, near right side

    print("Solving for tangent circle...")
    center, point1, point2, pos1, pos2 = solve_tangent_circle_two_sinusoids(
        pos1_init=pos1_init,
        base_depth1=base_depth_right,
        amplitude1=amplitude,
        period1=period,
        offset1=offset_right,
        pos2_init=pos2_init,
        base_depth2=base_depth_back,
        amplitude2=amplitude,
        period2=period,
        offset2=offset_back,
        radius=radius,
        wall1_type='R',
        wall2_type='B',
        pantry_width=pantry_width,
        pantry_depth=pantry_depth,
        corner_type='right-back'
    )

    print(f"Circle center: ({center[0]:.3f}, {center[1]:.3f})")
    print(f"Tangent point on right wall: ({point1[0]:.3f}, {point1[1]:.3f}) at position {pos1:.3f}")
    print(f"Tangent point on back wall: ({point2[0]:.3f}, {point2[1]:.3f}) at position {pos2:.3f}")

    # Generate the arc
    arc_points = generate_circle_arc(center, point1, point2, num_points=50, interior_arc=True)

    # Generate curves
    right_curve_wall = generate_sinusoid_points(0, pantry_depth, base_depth_right,
                                               amplitude, period, offset_right, num_points=200)
    back_curve_wall = generate_sinusoid_points(0, pantry_width, base_depth_back,
                                                amplitude, period, offset_back, num_points=200)

    right_curve_pantry = np.array([
        wall_to_pantry_coords(pos, depth, 'W', pantry_width, pantry_depth)
        for pos, depth in right_curve_wall
    ])
    back_curve_pantry = np.array([
        wall_to_pantry_coords(pos, depth, 'S', pantry_width, pantry_depth)
        for pos, depth in back_curve_wall
    ])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    pantry_outline = np.array([
        [0, 0], [pantry_width, 0], [pantry_width, pantry_depth],
        [0, pantry_depth], [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=2, label='Pantry walls')

    ax.plot(right_curve_pantry[:, 0], right_curve_pantry[:, 1], 'b-', linewidth=2, label='Right wall shelf')
    ax.plot(back_curve_pantry[:, 0], back_curve_pantry[:, 1], 'r-', linewidth=2, label='Back wall shelf')
    ax.plot(arc_points[:, 0], arc_points[:, 1], 'g-', linewidth=3, label=f'Tangent arc (R={radius}")')
    ax.plot(center[0], center[1], 'go', markersize=8, label='Circle center')
    ax.plot(point1[0], point1[1], 'bs', markersize=10, label='Tangent pt (Right)')
    ax.plot(point2[0], point2[1], 'rs', markersize=10, label='Tangent pt (Back)')

    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel('X (inches) - Left to Right')
    ax.set_ylabel('Y (inches) - Front (door) to Back')
    ax.set_title('Tangent Circle Solution: Right-Back Corner')

    plt.tight_layout()
    plt.savefig('tangent_circle_right_back.png', dpi=150)
    print("Saved visualization to tangent_circle_right_back.png")
    plt.close()


if __name__ == '__main__':
    test_left_back_corner()
    test_right_back_corner()
