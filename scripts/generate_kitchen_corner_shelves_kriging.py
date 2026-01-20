#!/usr/bin/env python3
"""
Generate kitchen corner shelf geometry using Gaussian process (kriging) curves.

This creates truly random, mathematically beautiful shelf edges using:
- Fixed circular arc at center (8", 33") with 4" radius
- Random angle ±45° from orthogonal to determine arc endpoints
- Gaussian process curves from cabinet/wall to arc with:
  * Hard observations (zero variance) at endpoints
  * Slope matching at arc tangent points
  * Smoothness regularization
  * Standard deviation ≤ 1" from straight line
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon, Arc as MPLArc
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import cdist

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


class GaussianProcessCurve:
    """
    Gaussian Process for generating smooth random curves with constraints.

    Uses kriging to generate curves conditioned on:
    - Hard observations (fixed points)
    - Derivative constraints (slope matching)
    - Smoothness via covariance function
    """

    def __init__(self, length_scale=10.0, variance=1.0, nugget=1e-6):
        """
        Initialize GP with covariance parameters.

        Args:
            length_scale: Correlation length for squared exponential kernel
            variance: Overall variance of the process
            nugget: Small value for numerical stability
        """
        self.length_scale = length_scale
        self.variance = variance
        self.nugget = nugget

    def kernel(self, x1, x2):
        """
        Squared exponential (Gaussian) covariance kernel.

        k(x1, x2) = σ² * exp(-||x1 - x2||² / (2 * l²))
        """
        # Ensure inputs are 2D arrays
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)

        # Compute squared distances
        dist_sq = cdist(x1, x2, metric='sqeuclidean')

        # Squared exponential kernel
        K = self.variance * np.exp(-dist_sq / (2 * self.length_scale**2))

        return K

    def kernel_derivative_y(self, x1, x2):
        """
        Derivative of kernel with respect to y-coordinate of x2.

        ∂k/∂y₂ = k(x1, x2) * (y₁ - y₂) / l²
        """
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)

        K = self.kernel(x1, x2)

        # y-difference
        y_diff = x1[:, 1:2] - x2[:, 1:2].T

        # Derivative
        dK_dy = K * y_diff / self.length_scale**2

        return dK_dy

    def condition_on_observations(self, obs_points, obs_values, obs_derivatives=None):
        """
        Condition GP on observations and optionally derivative constraints.

        Args:
            obs_points: Array of shape (n_obs, 2) - observation locations [y, x]
            obs_values: Array of shape (n_obs,) - observed x values
            obs_derivatives: Optional array of shape (n_deriv,) - dx/dy at obs_points

        Returns:
            Conditioned GP ready for prediction
        """
        self.obs_points = np.atleast_2d(obs_points)
        self.obs_values = np.atleast_1d(obs_values)

        n_obs = len(self.obs_values)

        # Build covariance matrix for observations
        K_obs = self.kernel(self.obs_points, self.obs_points)

        if obs_derivatives is not None:
            obs_derivatives = np.atleast_1d(obs_derivatives)
            n_deriv = len(obs_derivatives)

            # Extended covariance matrix including derivatives
            # [K_obs,     K_obs_deriv]
            # [K_deriv_obs, K_deriv_deriv]

            K_obs_deriv = self.kernel_derivative_y(self.obs_points[:n_deriv], self.obs_points)
            K_deriv_deriv = -self.kernel(self.obs_points[:n_deriv], self.obs_points[:n_deriv]) / self.length_scale**2

            # Build full covariance matrix
            K_full = np.block([
                [K_obs, K_obs_deriv.T],
                [K_obs_deriv, K_deriv_deriv]
            ])

            # Add nugget for numerical stability
            K_full += self.nugget * np.eye(n_obs + n_deriv)

            # Combined observation vector
            y_full = np.concatenate([self.obs_values, obs_derivatives])

            # Precompute inverse for predictions
            self.K_inv = np.linalg.solve(K_full, np.eye(n_obs + n_deriv))
            self.alpha = np.linalg.solve(K_full, y_full)

            self.has_derivatives = True
            self.n_deriv = n_deriv
        else:
            # No derivatives, standard GP
            K_obs += self.nugget * np.eye(n_obs)

            self.K_inv = np.linalg.solve(K_obs, np.eye(n_obs))
            self.alpha = np.linalg.solve(K_obs, self.obs_values)

            self.has_derivatives = False

    def predict(self, test_points, return_std=False):
        """
        Predict at test points using conditioned GP.

        Args:
            test_points: Array of shape (n_test, 2) - test locations [y, x]
            return_std: If True, also return standard deviation

        Returns:
            mean: Predicted x values at test points
            std: (optional) Standard deviation of predictions
        """
        test_points = np.atleast_2d(test_points)
        n_test = test_points.shape[0]

        # Covariance between test and observation points
        K_test_obs = self.kernel(test_points, self.obs_points)

        if self.has_derivatives:
            # Include derivative observations
            K_test_deriv = self.kernel_derivative_y(test_points, self.obs_points[:self.n_deriv])
            K_test_all = np.hstack([K_test_obs, K_test_deriv])

            mean = K_test_all @ self.alpha
        else:
            mean = K_test_obs @ self.alpha

        if return_std:
            # Posterior variance
            K_test = self.kernel(test_points, test_points)

            if self.has_derivatives:
                v = K_test_all @ self.K_inv @ K_test_all.T
            else:
                v = K_test_obs @ self.K_inv @ K_test_obs.T

            variance = np.diag(K_test - v)
            std = np.sqrt(np.maximum(variance, 0))  # Avoid negative due to numerics

            return mean, std

        return mean


def generate_kriging_corner_shelf(depth, length, corner_radius,
                                   circle_center_x, circle_center_y,
                                   random_angle_deg, gp_length_scale, gp_variance,
                                   n_points=100):
    """
    Generate kitchen corner shelf with GP (kriging) random curves.

    Args:
        depth: Base depth (12" for kitchen shelves)
        length: Length along wall (37" for kitchen shelves)
        corner_radius: Circle radius (4")
        circle_center_x: Circle center x-coordinate (8")
        circle_center_y: Circle center y-coordinate (33")
        random_angle_deg: Random angle ±45° from orthogonal
        gp_length_scale: GP correlation length
        gp_variance: GP variance
        n_points: Number of points for curve discretization

    Returns:
        Polygon as numpy array, debug info dict
    """
    c_x = circle_center_x
    c_y = circle_center_y
    r = corner_radius

    # Convert random angle to radians
    # Orthogonal to wall (pointing left) is at 180° (π)
    # Random angle is ± 45° from that
    angle_rad = np.deg2rad(180 + random_angle_deg)

    # Two arc endpoints from the random angle
    # First endpoint: at random_angle
    # Second endpoint: symmetric about the orthogonal (or could be another random angle)

    # For the right edge: arc point connects to cabinet corner (0, length)
    # For the bottom edge: arc point connects to wall point

    # Let's define two angles:
    # angle1: for right edge connection (toward cabinet corner at top-left)
    # angle2: for bottom edge connection (toward wall)

    # The angle from circle center to cabinet corner (0, length=37)
    # atan2(y - c_y, x - c_x) = atan2(37 - 33, 0 - 8) = atan2(4, -8)
    angle_to_cabinet = np.arctan2(length - c_y, 0 - c_x)

    # The angle from circle center to a point on the left wall (x=0)
    # We want the arc to span some range
    # Let's use the random_angle to determine the arc span

    # Arc point 1 (for right edge):
    angle1 = angle_rad
    x_arc1 = c_x + r * np.cos(angle1)
    y_arc1 = c_y + r * np.sin(angle1)

    # Arc point 2 (for bottom edge):
    # This should be roughly toward bottom-left
    # Let's make it symmetric or use another random angle
    # For simplicity, let's use angle1 + 90° or angle1 - 90°
    angle2 = angle1 - np.pi/2  # 90° clockwise
    x_arc2 = c_x + r * np.cos(angle2)
    y_arc2 = c_y + r * np.sin(angle2)

    # Derivative (slope) at arc points
    # For a circle, tangent direction at angle θ is perpendicular to radius
    # Tangent angle = θ + π/2
    # dx/dy = dx/dt / (dy/dt) where t is angle parameter
    # x = c_x + r*cos(θ), y = c_y + r*sin(θ)
    # dx/dθ = -r*sin(θ), dy/dθ = r*cos(θ)
    # dx/dy = -sin(θ)/cos(θ) = -tan(θ)

    slope_arc1 = -np.tan(angle1)  # dx/dy at arc point 1
    slope_arc2 = -np.tan(angle2)  # dx/dy at arc point 2

    # Generate right edge curve (cabinet corner to arc point 1)
    # Hard observations:
    # - At y = length (cabinet), x = 0 (cabinet corner)
    # - At y = y_arc1, x = x_arc1 (arc point)
    # Derivative constraint at arc point: dx/dy = slope_arc1

    gp_right = GaussianProcessCurve(length_scale=gp_length_scale, variance=gp_variance)

    obs_points_right = np.array([
        [length, 0],        # Cabinet corner at (y=37, x=0)
        [y_arc1, x_arc1]    # Arc point 1
    ])
    obs_values_right = np.array([0, x_arc1])
    obs_derivatives_right = np.array([slope_arc1])  # Only at second point (arc tangent)

    # Condition GP
    gp_right.condition_on_observations(obs_points_right, obs_values_right, obs_derivatives_right)

    # Generate right edge points
    y_right = np.linspace(y_arc1, length, n_points)
    test_points_right = np.column_stack([y_right, np.zeros(n_points)])  # x doesn't matter for prediction
    x_right, std_right = gp_right.predict(test_points_right, return_std=True)

    # Clip to ensure within ±1" of straight line
    x_straight_right = np.linspace(x_arc1, 0, n_points)
    x_right = np.clip(x_right, x_straight_right - 1.0, x_straight_right + 1.0)

    # Generate bottom edge curve (wall point to arc point 2)
    # We need to determine where on the wall (y=0) the bottom edge starts
    # Let's say it starts at x = x_arc2 (same x as arc point 2) on the wall (y=0)
    # Actually, let's make it start at x=0, y=0 (bottom-left corner)

    # Wait, let me reconsider the geometry:
    # - Cabinet corner is at (0, 37) - top-left
    # - Bottom-left is at (0, 0)
    # - Right edge goes from cabinet corner down to arc point 1
    # - Arc connects arc point 1 to arc point 2
    # - Bottom edge goes from arc point 2 to bottom-left corner

    gp_bottom = GaussianProcessCurve(length_scale=gp_length_scale, variance=gp_variance)

    obs_points_bottom = np.array([
        [y_arc2, x_arc2],   # Arc point 2
        [0, 0]              # Bottom-left corner at (y=0, x=0)
    ])
    obs_values_bottom = np.array([x_arc2, 0])
    obs_derivatives_bottom = np.array([slope_arc2])  # Only at first point (arc tangent)

    # Condition GP
    gp_bottom.condition_on_observations(obs_points_bottom, obs_values_bottom, obs_derivatives_bottom)

    # Generate bottom edge points
    y_bottom = np.linspace(0, y_arc2, n_points)
    test_points_bottom = np.column_stack([y_bottom, np.zeros(n_points)])
    x_bottom, std_bottom = gp_bottom.predict(test_points_bottom, return_std=True)

    # Clip to ensure within ±1" of straight line
    x_straight_bottom = np.linspace(0, x_arc2, n_points)
    x_bottom = np.clip(x_bottom, x_straight_bottom - 1.0, x_straight_bottom + 1.0)

    # Generate arc points between arc point 2 and arc point 1
    # Arc goes counter-clockwise from angle2 to angle1
    if angle1 < angle2:
        arc_angles = np.linspace(angle2, angle1 + 2*np.pi, 20)
    else:
        arc_angles = np.linspace(angle2, angle1, 20)

    x_arc = c_x + r * np.cos(arc_angles)
    y_arc = c_y + r * np.sin(arc_angles)

    # Build complete polygon counter-clockwise
    polygon = []

    # Start at bottom-left corner
    polygon.append([0, 0])

    # Bottom edge (from bottom-left to arc point 2)
    for i in range(len(y_bottom)):
        polygon.append([x_bottom[i], y_bottom[i]])

    # Arc (from arc point 2 to arc point 1)
    for i in range(len(arc_angles)):
        polygon.append([x_arc[i], y_arc[i]])

    # Right edge (from arc point 1 to cabinet corner)
    for i in range(len(y_right)):
        polygon.append([x_right[i], y_right[i]])

    # Top edge (from cabinet corner along wall to top-left)
    # Actually, the cabinet corner IS the top-left at (0, length)
    # We just close back to (0, 0) via the left edge

    # Left edge is implicit - it closes back to start

    debug_info = {
        'circle_center': (c_x, c_y),
        'circle_radius': r,
        'random_angle_deg': random_angle_deg,
        'arc_point1': (x_arc1, y_arc1),
        'arc_point2': (x_arc2, y_arc2),
        'slope_arc1': slope_arc1,
        'slope_arc2': slope_arc2,
        'gp_params': {
            'length_scale': gp_length_scale,
            'variance': gp_variance
        },
        'std_right': std_right.max(),
        'std_bottom': std_bottom.max()
    }

    return np.array(polygon), debug_info


def export_polygon_to_svg(polygon, filepath, shelf_height, width_in, height_in):
    """Export polygon to SVG."""
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    dpi = 96
    svg_width = width * dpi
    svg_height = height * dpi

    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">')
    svg_lines.append(f'  <title>Kitchen Corner Shelf (Kriging) {shelf_height}" - {width:.2f}" x {height:.2f}"</title>')

    path_commands = []
    for i, (x, y) in enumerate(polygon):
        x_svg = (x - min_x) * dpi
        y_svg = (max_y - y) * dpi

        if i == 0:
            path_commands.append(f'M {x_svg:.3f},{y_svg:.3f}')
        else:
            path_commands.append(f'L {x_svg:.3f},{y_svg:.3f}')

    path_commands.append('Z')
    path_data = ' '.join(path_commands)

    svg_lines.append(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="2"/>')
    svg_lines.append('</svg>')

    with open(filepath, 'w') as f:
        f.write('\n'.join(svg_lines))

    print(f"  Exported: {filepath}")


def create_visualization(polygon, shelf_height, debug_info, output_path):
    """Create visualization with circle and GP curves."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot polygon
    patch = MPLPolygon(polygon, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(patch)

    # Plot circle center
    c_x, c_y = debug_info['circle_center']
    r = debug_info['circle_radius']
    ax.plot(c_x, c_y, 'ro', markersize=8, label=f'Circle center ({c_x:.1f}", {c_y:.1f}")')

    # Plot circle (for reference)
    circle = plt.Circle((c_x, c_y), r, fill=False, edgecolor='red',
                        linestyle='--', alpha=0.3, linewidth=1)
    ax.add_patch(circle)

    # Plot arc points
    x1, y1 = debug_info['arc_point1']
    x2, y2 = debug_info['arc_point2']
    ax.plot([x1, x2], [y1, y2], 'go', markersize=6, label='Arc tangent points')

    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])

    margin = 2
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax.set_xlabel('Depth from wall (inches)')
    ax.set_ylabel('Length along wall (inches)')
    ax.set_title(f'Kitchen Corner Shelf (Kriging) - {shelf_height}" from ceiling\n' +
                 f'Random angle: {debug_info["random_angle_deg"]:.1f}°, ' +
                 f'GP length scale: {debug_info["gp_params"]["length_scale"]:.1f}"')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {output_path}")


def main():
    print("=" * 60)
    print("Kitchen Corner Shelf Generator (Kriging/GP)")
    print("=" * 60)

    # Kitchen corner specifications
    depth = 12.0
    length = 37.0
    corner_radius = 4.0
    thickness = 1.0

    # Fixed circle center
    circle_center_x = 8.0
    circle_center_y = 33.0

    # Shelf heights
    shelf_heights = [12, 24, 36]

    print(f"\nShelf specifications:")
    print(f"  Space: {length}\" along wall × {depth}\" depth")
    print(f"  Fixed circle: center ({circle_center_x}\", {circle_center_y}\"), radius {corner_radius}\"")
    print(f"  Edge method: Gaussian Process (kriging) with smoothness regularization")
    print(f"  Thickness: {thickness}\"")
    print(f"  Number of shelves: {len(shelf_heights)}")

    # Output directory
    output_dir = Path('output/kitchen_corner_kriging')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    print(f"\nGenerating shelves with random GP curves:")

    all_polygons = []

    for i, height in enumerate(shelf_heights):
        # Random angle ±45° from orthogonal
        random_angle = np.random.uniform(-45, 45)

        # Random GP parameters
        gp_length_scale = np.random.uniform(5.0, 15.0)
        gp_variance = np.random.uniform(0.5, 2.0)

        # Generate polygon
        polygon, debug_info = generate_kriging_corner_shelf(
            depth, length, corner_radius,
            circle_center_x, circle_center_y,
            random_angle, gp_length_scale, gp_variance
        )

        # Get dimensions
        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        print(f"\n  Shelf at {height}\" from ceiling:")
        print(f"    Dimensions: {width:.2f}\" × {poly_height:.2f}\"")
        print(f"    Random angle: {random_angle:.1f}°")
        print(f"    GP length scale: {gp_length_scale:.2f}\"")
        print(f"    GP variance: {gp_variance:.2f}")
        print(f"    Max std (right): {debug_info['std_right']:.3f}\"")
        print(f"    Max std (bottom): {debug_info['std_bottom']:.3f}\"")

        # Export SVG
        svg_path = output_dir / f'kitchen_corner_kriging_shelf_{height}in.svg'
        export_polygon_to_svg(polygon, svg_path, height, width, poly_height)

        # Create visualization
        viz_path = output_dir / f'kitchen_corner_kriging_shelf_{height}in.png'
        create_visualization(polygon, height, debug_info, viz_path)

        all_polygons.append((polygon, height, debug_info))

    # Create combined PDF
    print(f"\nGenerating combined PDF...")
    pdf_path = output_dir / 'kitchen_corner_kriging_cutting_templates.pdf'

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Kitchen Corner Shelf', ha='center', fontsize=24, weight='bold')
        fig.text(0.5, 0.65, 'Kriging/GP Random Curves', ha='center', fontsize=20)
        fig.text(0.5, 0.55, f'Space: {length}" × {depth}"', ha='center', fontsize=12)
        fig.text(0.5, 0.52, f'Fixed circle: ({circle_center_x}", {circle_center_y}"), r={corner_radius}"', ha='center', fontsize=12)
        fig.text(0.5, 0.49, 'Gaussian Process edges with smoothness regularization', ha='center', fontsize=12)
        heights_str = ", ".join(str(h) + '"' for h in shelf_heights)
        fig.text(0.5, 0.46, f'{len(shelf_heights)} shelves: {heights_str} from ceiling', ha='center', fontsize=12)
        fig.text(0.5, 0.35, 'Mathematically beautiful random curves', ha='center', fontsize=10, style='italic')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Individual shelf pages
        for polygon, height, debug_info in all_polygons:
            fig, ax = plt.subplots(figsize=(11, 8.5))

            patch = MPLPolygon(polygon, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(patch)

            # Circle and arc points
            c_x, c_y = debug_info['circle_center']
            r = debug_info['circle_radius']
            circle = plt.Circle((c_x, c_y), r, fill=False, edgecolor='red',
                               linestyle='--', alpha=0.3)
            ax.add_patch(circle)

            min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
            min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
            width = max_x - min_x
            poly_height = max_y - min_y

            margin = 2
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            ax.set_xlabel('Depth from wall (inches)', fontsize=12)
            ax.set_ylabel('Length along wall (inches)', fontsize=12)
            ax.set_title(f'Kitchen Corner Shelf (Kriging) - {height}" from ceiling\n' +
                        f'{width:.2f}" × {poly_height:.2f}" | Angle: {debug_info["random_angle_deg"]:.1f}°',
                        fontsize=14, weight='bold')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"  Saved: {pdf_path}")

    print("\n" + "=" * 60)
    print(f"SUCCESS! Generated {len(shelf_heights)} kriging shelves")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - {len(shelf_heights)} SVG cutting templates")
    print(f"  - {len(shelf_heights)} visualization images")
    print(f"  - 1 combined PDF")


if __name__ == '__main__':
    main()
