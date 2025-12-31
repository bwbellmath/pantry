#!/usr/bin/env python3
"""
Generate kitchen corner shelf geometry and SVG files.

Kitchen corner specifications:
- Space: 37" along wall, 12" depth from wall
- 3 shelves at heights: 12", 24", 36" (from ceiling to bottom of shelf)
- Shelf thickness: 1"
- Corner radius: 4"
- Sinusoidal edges: 2" amplitude, 24" period
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.backends.backend_pdf import PdfPages

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


def simple_minimize_scalar(func, bounds, tol=1e-6, max_iter=100):
    """Simple golden section search for scalar minimization."""
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    a, b = bounds
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = func(x1)
    f2 = func(x2)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = func(x2)

    class Result:
        def __init__(self, x):
            self.x = x

    return Result((a + b) / 2)


def generate_kitchen_corner_shelf(depth, length, amplitude, period, offset, corner_radius):
    """
    Generate kitchen corner shelf polygon.

    Creates a corner shelf with:
    - Top-left: Cabinet corner (90 degree, no radius)
    - Top edge: Straight along one wall
    - Left edge: Straight along other wall
    - Right edge: Sinusoidal
    - Bottom edge: Straight (far from cabinet)
    - Bottom-right corner: Rounded where sinusoid meets straight edge

    Coordinate system:
    - Origin (0,0) is bottom-left
    - X-axis: depth (0 to ~12")
    - Y-axis: length (0 to 37")
    - Cabinet corner at (0, length) = top-left
    - Outer corner at (~12, 0) = bottom-right

    Args:
        depth: Base depth (12" for kitchen shelves)
        length: Length along wall (37" for kitchen shelves)
        amplitude: Sinusoid amplitude (2")
        period: Sinusoid period (24")
        offset: Random phase offset
        corner_radius: Bottom-right corner radius (4" for kitchen shelves)

    Returns:
        Polygon as numpy array of (x, y) coordinates
    """
    # Sinusoid for right edge: x = depth + amplitude * sin(2*pi*y/period + offset)
    # We need to find where the circle is tangent to:
    # 1. The sinusoid (right edge)
    # 2. The straight line y = 0 (bottom edge)

    # For a circle tangent to y = 0:
    # Center must be at (c_x, corner_radius)

    # For a circle tangent to the sinusoid x = f(y):
    # At tangent point (x_t, y_t):
    # 1. Point is on circle: (x_t - c_x)^2 + (y_t - c_y)^2 = r^2
    # 2. Normal to curve points toward center
    # For x = f(y), tangent vector is (dx/dy, 1), normal is (-dx/dy, 1) (inward)
    # Vector from point to center (c_x - x_t, c_y - y_t) || normal
    # So: (c_x - x_t) / (-dx/dy) = (c_y - y_t) / 1

    # Since circle is tangent to y=0, c_y = corner_radius

    def find_tangent_point(y_guess):
        """Find tangent point on sinusoid that satisfies both tangency and circle constraints."""
        y_t = y_guess

        # Check bounds
        if y_t < 0 or y_t > length/4:
            return 1e10

        x_t = depth + amplitude * np.sin(2 * np.pi * y_t / period + offset)

        # Derivative dx/dy at this point
        dx_dy = amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y_t / period + offset)

        # For valid solution, need dx_dy > 0 (curve sloping upward to the right)
        # Threshold scales with amplitude: minimum slope should be ~0.15 * amplitude
        min_slope = 0.15 * amplitude
        if dx_dy <= 0 or abs(dx_dy) < min_slope:
            return 1e10

        # Combined constraint from tangency + circle equation:
        # From tangency: x_t - c_x = (r - y_t) / dx_dy
        # From circle: (x_t - c_x)² = y_t(2*r - y_t)
        # These must both be satisfied:
        # [(r - y_t) / dx_dy]² = y_t(2*r - y_t)

        r = corner_radius
        lhs = (r - y_t)**2 / (dx_dy**2)
        rhs = y_t * (2*r - y_t)

        # The error is the difference between these two expressions
        error = abs(lhs - rhs)

        # Also compute c_x and verify it's reasonable
        c_x = x_t - (r - y_t) / dx_dy

        # Sanity checks: circle must be interior and reasonable
        if c_x <= 0 or c_x >= x_t or c_x > depth:
            return 1e10

        return error

    # Calculate where dx/dy goes to zero
    # dx/dy = amplitude * (2π/period) * cos(2πy/period + offset) = 0
    # when: 2πy/period + offset = π/2
    # y = (π/2 - offset) * period / (2π)
    y_derivative_zero = (np.pi/2 - offset) * period / (2*np.pi)

    # Search for tangent point on the sinusoid
    # Look in the range where we expect the tangent, but stop before derivative goes to zero
    # Also leave some margin (90% of the zero crossing) to ensure dx/dy > 0
    upper_bound = min(2*corner_radius, length/4, 0.9 * y_derivative_zero if y_derivative_zero > 0 else length)
    result = simple_minimize_scalar(find_tangent_point, bounds=(0, upper_bound))
    y_tangent = result.x

    # Calculate the tangent point and circle center
    x_tangent = depth + amplitude * np.sin(2 * np.pi * y_tangent / period + offset)
    dx_dy = amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y_tangent / period + offset)
    # Corrected formula: divide instead of multiply
    c_x = x_tangent - (corner_radius - y_tangent) / dx_dy
    c_y = corner_radius

    # DEBUG: Return debug info
    debug_info = {
        'offset': offset,
        'y_derivative_zero': y_derivative_zero,
        'upper_bound': upper_bound,
        'y_tangent': y_tangent,
        'x_tangent': x_tangent,
        'c_x': c_x,
        'c_y': c_y,
        'dx_dy': dx_dy,
        'error': find_tangent_point(y_tangent)
    }

    # Angles for the arc
    # From tangent on bottom edge to tangent on sinusoid
    angle_bottom = -np.pi / 2  # Pointing downward to bottom edge tangent point at (c_x, 0)
    angle_sinusoid = np.arctan2(y_tangent - c_y, x_tangent - c_x)

    # Build polygon counter-clockwise starting from bottom-left
    polygon = []

    # Bottom-left corner
    polygon.append([0, 0])

    # Bottom edge to arc start (tangent point on bottom edge)
    polygon.append([c_x, 0])

    # Arc from bottom edge to sinusoid
    # Go counter-clockwise from angle_bottom to angle_sinusoid
    arc_angles = np.linspace(angle_bottom, angle_sinusoid, 20)
    for angle in arc_angles:
        x = c_x + corner_radius * np.cos(angle)
        y = c_y + corner_radius * np.sin(angle)
        polygon.append([x, y])

    # Right edge sinusoid from tangent point up to top
    y_points = np.linspace(y_tangent, length, 100)
    for y in y_points:
        x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        polygon.append([x, y])

    # Top edge from top-right back to top-left (cabinet corner)
    polygon.append([0, length])

    # Left edge is implicit (closes back to start at bottom-left)

    return np.array(polygon), debug_info


def export_polygon_to_svg(polygon, filepath, shelf_height, width_in, height_in):
    """
    Export a polygon to SVG using exact coordinates.

    Args:
        polygon: Numpy array of (x, y) coordinates
        filepath: Output SVG file path
        shelf_height: Height in inches (from ceiling)
        width_in: Bounding width in inches
        height_in: Bounding height in inches
    """
    # Get bounds
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # SVG setup (1 inch = 96 pixels at standard 96 DPI)
    dpi = 96
    svg_width = width * dpi
    svg_height = height * dpi

    # Create SVG path
    svg_lines = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">')
    svg_lines.append(f'  <title>Kitchen Corner Shelf {shelf_height}" - {width:.2f}" x {height:.2f}"</title>')

    # Build path (flip Y coordinate for SVG)
    path_commands = []
    for i, (x, y) in enumerate(polygon):
        # Translate to origin and flip Y
        x_svg = (x - min_x) * dpi
        y_svg = (max_y - y) * dpi  # Flip Y axis

        if i == 0:
            path_commands.append(f'M {x_svg:.3f},{y_svg:.3f}')
        else:
            path_commands.append(f'L {x_svg:.3f},{y_svg:.3f}')

    path_commands.append('Z')  # Close path
    path_data = ' '.join(path_commands)

    svg_lines.append(f'  <path d="{path_data}" fill="none" stroke="black" stroke-width="2"/>')
    svg_lines.append('</svg>')

    # Write to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(svg_lines))

    print(f"  Exported: {filepath}")


def create_visualization(polygon, shelf_height, output_path):
    """Create a visualization of the shelf with dimensions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot polygon
    patch = MPLPolygon(polygon, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(patch)

    # Get bounds
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Set equal aspect and limits
    margin = 2
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Labels
    ax.set_xlabel('Depth from wall (inches)')
    ax.set_ylabel('Length along wall (inches)')
    ax.set_title(f'Kitchen Corner Shelf - Height {shelf_height}" from ceiling\nDimensions: {width:.2f}" × {height:.2f}"')

    # Add dimension annotations
    ax.annotate(f'{width:.2f}"', xy=((min_x + max_x)/2, min_y - margin/2),
                ha='center', va='top', fontsize=10, color='blue')
    ax.annotate(f'{height:.2f}"', xy=(min_x - margin/2, (min_y + max_y)/2),
                ha='right', va='center', fontsize=10, color='blue', rotation=90)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {output_path}")


def main():
    print("=" * 60)
    print("Kitchen Corner Shelf Generator")
    print("=" * 60)

    # Kitchen corner specifications
    depth = 12.0  # 12" from wall to cabinet corner
    length = 37.0  # 37" along wall to cabinet side
    amplitude = 1.0  # 1" sinusoid amplitude
    period = 24.0  # 24" sinusoid period
    corner_radius = 4.0  # 4" outer corner radius
    thickness = 1.0  # 1" thick shelves

    # Shelf heights (from ceiling to bottom of shelf)
    shelf_heights = [12, 24, 36]

    print(f"\nShelf specifications:")
    print(f"  Space: {length}\" along wall × {depth}\" depth")
    print(f"  Corner radius: {corner_radius}\"")
    print(f"  Sinusoid: {amplitude}\" amplitude, {period}\" period")
    print(f"  Thickness: {thickness}\"")
    print(f"  Number of shelves: {len(shelf_heights)}")

    # Output directory
    output_dir = Path('output/kitchen_corner')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducible results
    np.random.seed(42)

    print(f"\nGenerating shelves:")

    all_polygons = []

    for i, height in enumerate(shelf_heights):
        # Use controlled offsets that ensure good derivative at bottom
        # dx/dy = amplitude * (2π/period) * cos(2πy/period + offset)
        # Derivative goes to zero at y = (π/2 - offset) * period / (2π)
        # We want this to be well beyond the search range (>6")
        # Good offsets: -π/2, -π/4, 0 give zero crossings at 12", 9", 6"
        offsets = [-np.pi / 2, -np.pi / 4, 0]
        offset = offsets[i]

        # Generate polygon
        polygon, debug_info = generate_kitchen_corner_shelf(
            depth, length, amplitude, period, offset, corner_radius
        )

        # Get dimensions
        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        print(f"\n  Shelf at {height}\" from ceiling:")
        print(f"    Dimensions: {width:.2f}\" × {poly_height:.2f}\"")
        print(f"    Phase offset: {np.degrees(debug_info['offset']):.1f}°")
        print(f"    Arc tangent at: y={debug_info['y_tangent']:.2f}\", x={debug_info['x_tangent']:.2f}\"")

        # Export SVG
        svg_path = output_dir / f'kitchen_corner_shelf_{height}in.svg'
        export_polygon_to_svg(polygon, svg_path, height, width, poly_height)

        # Create visualization
        viz_path = output_dir / f'kitchen_corner_shelf_{height}in.png'
        create_visualization(polygon, height, viz_path)

        all_polygons.append((polygon, height))

    # Create combined PDF
    print(f"\nGenerating combined PDF...")
    pdf_path = output_dir / 'kitchen_corner_cutting_templates.pdf'

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Kitchen Corner Shelf', ha='center', fontsize=24, weight='bold')
        fig.text(0.5, 0.65, 'Cutting Templates', ha='center', fontsize=20)
        fig.text(0.5, 0.55, f'Space: {length}" along wall × {depth}" depth', ha='center', fontsize=12)
        fig.text(0.5, 0.52, f'Corner radius: {corner_radius}"', ha='center', fontsize=12)
        fig.text(0.5, 0.49, f'Sinusoid: {amplitude}" amplitude, {period}" period', ha='center', fontsize=12)
        heights_str = ", ".join(str(h) + '"' for h in shelf_heights)
        fig.text(0.5, 0.46, f'{len(shelf_heights)} shelves: {heights_str} from ceiling', ha='center', fontsize=12)
        fig.text(0.5, 0.35, 'Material: 3/4" Baltic Birch Plywood', ha='center', fontsize=10, style='italic')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Individual shelf pages
        for polygon, height in all_polygons:
            fig, ax = plt.subplots(figsize=(11, 8.5))

            # Plot polygon
            patch = MPLPolygon(polygon, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(patch)

            # Get bounds
            min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
            min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
            width = max_x - min_x
            poly_height = max_y - min_y

            # Set equal aspect and limits
            margin = 2
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # Labels
            ax.set_xlabel('Depth from wall (inches)', fontsize=12)
            ax.set_ylabel('Length along wall (inches)', fontsize=12)
            ax.set_title(f'Kitchen Corner Shelf - {height}" from ceiling\n{width:.2f}" × {poly_height:.2f}"',
                        fontsize=14, weight='bold')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"  Saved: {pdf_path}")

    print("\n" + "=" * 60)
    print(f"SUCCESS! Generated {len(shelf_heights)} kitchen corner shelves")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}/")
    print(f"  - {len(shelf_heights)} SVG cutting templates")
    print(f"  - {len(shelf_heights)} visualization images")
    print(f"  - 1 combined PDF")


if __name__ == '__main__':
    main()
