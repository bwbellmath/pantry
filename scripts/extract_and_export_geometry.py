#!/usr/bin/env python3
"""
Extract exact geometry from shelf computations and export to SVG/PDF.
Uses the EXACT same points that are drawn in visualizations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    generate_interior_mask
)
from config import ShelfConfig


def solve_tangent_circle_horizontal_sinusoid(horizontal_y, depth, amplitude, period, offset, radius, side='E'):
    """
    Solve for a circle tangent to a horizontal line and a sinusoid.

    Args:
        horizontal_y: Y-coordinate of horizontal line
        depth: Base depth of sinusoid
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Sinusoid phase offset
        radius: Desired circle radius
        side: 'E' for left (sinusoid to right), 'W' for right (sinusoid to left)

    Returns:
        (center_x, center_y, tangent_y): Circle center and tangency point y-coordinate
    """
    from scipy.optimize import minimize_scalar

    # Circle center must be at y = horizontal_y - radius to be tangent to horizontal line
    center_y = horizontal_y - radius

    # Find tangency point on sinusoid
    # For tangency, the distance from center to sinusoid must equal radius
    # and the gradient must match
    def objective(y):
        if side == 'E':
            x_sinusoid = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        else:  # 'W'
            x_sinusoid = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

        # Distance from (center_x, center_y) to (x_sinusoid, y)
        # We want to find center_x such that this distance equals radius
        # For now, estimate center_x from the horizontal tangency
        if side == 'E':
            center_x_estimate = x_sinusoid - radius
        else:
            center_x_estimate = x_sinusoid + radius

        dist = np.sqrt((x_sinusoid - center_x_estimate)**2 + (y - center_y)**2)
        return abs(dist - radius)

    # Search for tangency point between horizontal_y - 2*radius and horizontal_y
    result = minimize_scalar(objective, bounds=(horizontal_y - 2*radius, horizontal_y), method='bounded')
    tangent_y = result.x

    # Calculate center_x from tangency point
    if side == 'E':
        x_sinusoid = depth + amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid - radius
    else:
        x_sinusoid = 48 - depth - amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid + radius

    return center_x, center_y, tangent_y


def generate_intermediate_shelf(depth, length, side, amplitude, period, offset, corner_radius=3.0):
    """
    Generate a simple intermediate shelf polygon with corner radiusing at back interior corner.

    Args:
        depth: Base depth in inches (7" for left, 4" for right)
        length: Length from door inward (29")
        side: 'E' for left/east wall, 'W' for right/west wall
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Random phase offset for this shelf
        corner_radius: Radius for back interior corner (default 3")

    Returns:
        Polygon array with sinusoidal interior edge and rounded back interior corner
    """
    # Generate sinusoid points along the interior edge
    y_points = np.linspace(0, length, 100)

    # Generate corner arc (quarter circle)
    arc_angles = np.linspace(0, np.pi/2, 20)

    polygon = []

    if side == 'E':
        # Left shelf: wall at x=0, sinusoid on right side
        # Back interior corner is SE (where south edge meets sinusoid)

        # Solve for tangent circle
        center_x, center_y, tangent_y = solve_tangent_circle_horizontal_sinusoid(
            length, depth, amplitude, period, offset, corner_radius, side='E')

        # Calculate actual tangency points
        x_sinusoid_tangent = depth + amplitude * np.sin(2 * np.pi * tangent_y / period + offset)

        # Horizontal tangency point is directly above center
        x_horizontal_tangent = center_x
        y_horizontal_tangent = length

        # Calculate angles of tangency points from center
        angle_horizontal = np.pi/2  # Directly above center
        angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

        # 1. Start at NW corner (wall meets door)
        polygon.append([0, 0])

        # 2. Wall edge from [0, 0] to [0, length]
        polygon.append([0, length])

        # 3. South edge from wall to horizontal tangency point
        polygon.append([x_horizontal_tangent, y_horizontal_tangent])

        # 4. SE corner arc - from horizontal tangency to sinusoid tangency
        arc_angles_actual = np.linspace(angle_horizontal, angle_sinusoid, 20)
        for angle in arc_angles_actual:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        # 5. Sinusoid edge going north (from tangent point to y=0)
        for y in y_points[::-1]:
            if y <= tangent_y:
                x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        # 6. North/door edge from sinusoid back to wall
        x_north = depth + amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])

    else:  # side == 'W'
        # Right shelf: wall at x=48, sinusoid on left side
        # Back interior corner is SW (where south edge meets sinusoid)

        # Solve for tangent circle
        center_x, center_y, tangent_y = solve_tangent_circle_horizontal_sinusoid(
            length, depth, amplitude, period, offset, corner_radius, side='W')

        # Calculate actual tangency points
        x_sinusoid_tangent = 48 - depth - amplitude * np.sin(2 * np.pi * tangent_y / period + offset)

        # Horizontal tangency point is directly above center
        x_horizontal_tangent = center_x
        y_horizontal_tangent = length

        # Calculate angles of tangency points from center
        angle_horizontal = np.pi/2  # Directly above center
        angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

        # 1. Start at NE corner (wall meets door)
        polygon.append([48, 0])

        # 2. North/door edge from wall to sinusoid
        x_north = 48 - depth - amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])

        # 3. Sinusoid edge going south (from y=0 to tangent point)
        for y in y_points:
            if y <= tangent_y:
                x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        # 4. SW corner arc - from sinusoid tangency to horizontal tangency
        arc_angles_actual = np.linspace(angle_sinusoid, angle_horizontal, 20)
        for angle in arc_angles_actual:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        # 5. South edge from horizontal tangency to wall
        polygon.append([48, length])

        # 6. Wall edge from [48, length] back to [48, 0]
        # (will close automatically)

    return np.array(polygon)


def extract_exact_shelf_geometries(config, level):
    """
    Extract the EXACT geometry for each shelf piece using the same calculations
    as the visualization code.

    Returns dict with 'E', 'S', 'W' keys containing polygon point arrays.
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

    # Generate sinusoid curves - EXACT same as in generate_shelves.py
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

    # Solve for corner arcs - EXACT same as in generate_shelves.py
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
        lb_arc_original = generate_circle_arc(lb_center, lb_point1, lb_point2, num_points=30, interior_arc=True)
        # Reverse arc so it goes from door (lb_point1) to back (lb_point2) for side shelves
        # generate_circle_arc may swap points for counter-clockwise ordering
        if np.linalg.norm(lb_arc_original[0] - lb_point1) > 0.1:
            lb_arc = lb_arc_original[::-1]
        else:
            lb_arc = lb_arc_original
        lb_cut_y = (lb_point1[1] + lb_point2[1]) / 2
    except Exception as e:
        print(f"Error solving left-back corner: {e}")
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
        rb_arc_original = generate_circle_arc(rb_center, rb_point1, rb_point2, num_points=30, interior_arc=True)
        # Reverse arc so it goes from door (rb_point1) to back (rb_point2) for side shelves
        if np.linalg.norm(rb_arc_original[0] - rb_point1) > 0.1:
            rb_arc = rb_arc_original[::-1]
        else:
            rb_arc = rb_arc_original
        rb_cut_y = (rb_point1[1] + rb_point2[1]) / 2
    except Exception as e:
        print(f"Error solving right-back corner: {e}")
        return None

    # Now build the THREE shelf pieces by cutting at the horizontal lines
    # Key: Stop sinusoids at tangency points (lb_point1, lb_point2, rb_point1, rb_point2)

    # Helper function to find closest point in curve to target
    def find_closest_index(curve, target_point):
        distances = np.linalg.norm(curve - target_point, axis=1)
        return np.argmin(distances)

    # LEFT (East) SHELF POLYGON
    # Keep arc articulation, find intersection with horizontal cut line
    left_polygon = []
    left_polygon.append([0, 0])  # 1. NW corner (door side)

    # 2. Add sinusoid points BEFORE tangency point
    lb1_idx = find_closest_index(left_curve, lb_point1)
    for i in range(lb1_idx):  # Stop BEFORE lb1_idx (don't include tangency point)
        left_polygon.append(left_curve[i])

    # 3. Add arc points STRICTLY BELOW cut line, then interpolate exact intersection
    # Skip first arc point (tangency)
    for i, point in enumerate(lb_arc):
        if i > 0 and point[1] < lb_cut_y - 0.01:  # Strictly below cut line
            left_polygon.append(point)

    # 4. Find EXACT intersection of arc with horizontal line at y=lb_cut_y
    # Find consecutive arc points that straddle the cut line
    intersection_found = False
    for i in range(1, len(lb_arc)):
        if lb_arc[i-1][1] < lb_cut_y <= lb_arc[i][1]:
            # Interpolate between lb_arc[i-1] and lb_arc[i]
            p1, p2 = lb_arc[i-1], lb_arc[i]
            t = (lb_cut_y - p1[1]) / (p2[1] - p1[1] + 1e-10)
            intersection_x = p1[0] + t * (p2[0] - p1[0])
            left_polygon.append([intersection_x, lb_cut_y])
            intersection_found = True
            break

    if not intersection_found:
        # Fallback: use last arc point at cut line
        for i in range(len(lb_arc) - 1, 0, -1):
            if lb_arc[i][1] <= lb_cut_y:
                left_polygon.append([lb_arc[i][0], lb_cut_y])
                break

    # 5. Add wall point at cut line
    left_polygon.append([0, lb_cut_y])

    left_polygon = np.array(left_polygon)

    # RIGHT (West) SHELF POLYGON
    # Keep arc articulation, find intersection with horizontal cut line
    right_polygon = []
    right_polygon.append([pantry_width, 0])  # 1. NE corner (door side)

    # 2. Add sinusoid points BEFORE tangency point
    rb1_idx = find_closest_index(right_curve, rb_point1)
    for i in range(rb1_idx):  # Stop BEFORE rb1_idx (don't include tangency point)
        right_polygon.append(right_curve[i])

    # 3. Add arc points STRICTLY BELOW cut line, then interpolate exact intersection
    # Skip first arc point (tangency)
    for i, point in enumerate(rb_arc):
        if i > 0 and point[1] < rb_cut_y - 0.01:  # Strictly below cut line
            right_polygon.append(point)

    # 4. Find EXACT intersection of arc with horizontal line at y=rb_cut_y
    # Find consecutive arc points that straddle the cut line
    intersection_found = False
    for i in range(1, len(rb_arc)):
        if rb_arc[i-1][1] < rb_cut_y <= rb_arc[i][1]:
            # Interpolate between rb_arc[i-1] and rb_arc[i]
            p1, p2 = rb_arc[i-1], rb_arc[i]
            t = (rb_cut_y - p1[1]) / (p2[1] - p1[1] + 1e-10)
            intersection_x = p1[0] + t * (p2[0] - p1[0])
            right_polygon.append([intersection_x, rb_cut_y])
            intersection_found = True
            break

    if not intersection_found:
        # Fallback: use last arc point at cut line
        for i in range(len(rb_arc) - 1, 0, -1):
            if rb_arc[i][1] <= rb_cut_y:
                right_polygon.append([rb_arc[i][0], rb_cut_y])
                break

    # 5. Add wall point at cut line
    right_polygon.append([pantry_width, rb_cut_y])

    right_polygon = np.array(right_polygon)

    # BACK (South) SHELF POLYGON
    # Exact vertex order (going clockwise from left cut):
    # 1. (0, lb_cut_y) - left wall at cut
    # 2. (0, pantry_depth) - SW corner
    # 3. (pantry_width, pantry_depth) - SE corner
    # 4. (pantry_width, rb_cut_y) - right wall at cut
    # 5. Arc points from rb_point2 back toward cut (reversed, above cut) - STOP BEFORE rb_point2
    # 6. Sinusoid points from rb_point2 to lb_point2 (reversed) - arcs provide endpoints
    # 7. Arc points from lb_point2 toward cut (above cut) - STOP BEFORE lb_point2
    # 8. Close
    back_polygon = []

    # 1. Left wall at cut
    back_polygon.append([0, lb_cut_y])

    # 2. SW corner
    back_polygon.append([0, pantry_depth])

    # 3. SE corner (across back wall)
    back_polygon.append([pantry_width, pantry_depth])

    # 4. Right wall at cut
    back_polygon.append([pantry_width, rb_cut_y])

    # 5. Right arc points (reversed) from cut going up to rb_point2
    # Use original arc orientation (before the fix for side shelves)
    rb_arc_for_back = rb_arc_original[::-1]  # Reverse original arc
    for point in rb_arc_for_back:
        if point[1] >= rb_cut_y - 0.1:  # Above cut line
            back_polygon.append(point)

    # 6. Back sinusoid from rb_point2 to lb_point2 (REVERSED)
    # Find indices
    lb2_idx = find_closest_index(back_curve, lb_point2)
    rb2_idx = find_closest_index(back_curve, rb_point2)

    # Add points from rb2 to lb2 (going from right to left)
    for i in range(rb2_idx, lb2_idx - 1, -1):  # Reversed range
        back_polygon.append(back_curve[i])

    # 7. Left arc points from lb_point2 back down to cut
    # Use original arc orientation (before the fix for side shelves), forward direction
    for point in lb_arc_original:
        if point[1] >= lb_cut_y - 0.1:  # Above cut line
            back_polygon.append(point)

    back_polygon = np.array(back_polygon)

    return {
        'E': left_polygon,
        'S': back_polygon,
        'W': right_polygon,
        'lb_cut_y': lb_cut_y,
        'rb_cut_y': rb_cut_y,
        'lb_arc': lb_arc,
        'rb_arc': rb_arc,
        'left_curve': left_curve,
        'back_curve': back_curve,
        'right_curve': right_curve
    }


def export_polygon_to_svg(polygon, filepath, shelf_height, wall_name, width_in, height_in):
    """Export a polygon to SVG using exact coordinates.

    Args:
        polygon: Polygon vertices
        filepath: Output file path
        shelf_height: Bottom-referenced height in inches
        wall_name: 'L', 'R', or 'B' for left/right/back
        width_in: Polygon width
        height_in: Polygon height
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Get bounds
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Add margin
    margin = 5
    viewbox_width = width + 2 * margin
    viewbox_height = height + 2 * margin
    viewbox_x = min_x - margin
    viewbox_y = min_y - margin

    # Create SVG with correct viewBox
    svg_lines = []
    svg_lines.append(f'<?xml version="1.0" encoding="utf-8" ?>')
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" ')
    svg_lines.append(f'     width="{viewbox_width}in" height="{viewbox_height}in" ')
    svg_lines.append(f'     viewBox="{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}">')

    # Add title
    svg_lines.append(f'  <title>Shelf {wall_name}{shelf_height}" - {width:.2f}" x {height:.2f}"</title>')

    # Build path data
    path_data = f'M {polygon[0, 0]},{polygon[0, 1]}'
    for point in polygon[1:]:
        path_data += f' L {point[0]},{point[1]}'
    path_data += ' Z'  # Close path

    # Add polygon as path
    svg_lines.append(f'  <path d="{path_data}" ')
    svg_lines.append(f'        fill="lightblue" stroke="black" stroke-width="0.1" />')

    # Add dimension text
    svg_lines.append(f'  <text x="{min_x}" y="{min_y - 2}" font-size="2" font-family="Arial">')
    svg_lines.append(f'    Shelf {wall_name}{shelf_height}"')
    svg_lines.append(f'  </text>')
    svg_lines.append(f'  <text x="{min_x}" y="{min_y - 0.5}" font-size="1.5" font-family="Arial">')
    svg_lines.append(f'    {width:.2f}" × {height:.2f}"')
    svg_lines.append(f'  </text>')

    svg_lines.append('</svg>')

    # Write file
    with open(filepath, 'w') as f:
        f.write('\n'.join(svg_lines))


def simple_2d_pack(polygons, bin_width=96, bin_height=48):
    """
    Simple shelf-by-shelf packing.

    Args:
        polygons: List of (polygon, shelf_height, wall_name) tuples

    Returns:
        List of (bin_num, polygon, x, y, rotated, shelf_height, wall_name) tuples
    """
    placements = []
    current_bin = 0
    current_y = 0
    row_height = 0
    current_x = 0

    for polygon, shelf_height, wall_name in polygons:
        # Get bounds
        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        # Normalize polygon to origin
        norm_poly = polygon - np.array([min_x, min_y])

        # Check if fits in current row
        if current_x + width > bin_width:
            # Move to next row
            current_x = 0
            current_y += row_height + 2  # 2" spacing
            row_height = 0

            # Check if need new bin
            if current_y + height > bin_height:
                current_bin += 1
                current_y = 0
                row_height = 0

        # Place piece
        placements.append((current_bin, norm_poly, current_x, current_y, False, shelf_height, wall_name))

        # Update trackers
        current_x += width + 2  # 2" spacing
        row_height = max(row_height, height)

    return placements


def create_overview_page(ax, config, level, geom_data):
    """Create overview page matching generate_shelves.py exactly."""
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

    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Generate interior mask
    X, Y, interior_mask = generate_interior_mask(
        left_depth, amplitude, period, left_offset,
        right_depth, amplitude, period, right_offset,
        back_depth, amplitude, period, back_offset,
        pantry_width, pantry_depth,
        resolution=150
    )

    ax.contourf(X, Y, ~interior_mask, levels=[0.5, 1.5], colors=['#FFE4B5'], alpha=0.4)
    ax.contourf(X, Y, interior_mask, levels=[0.5, 1.5], colors=['#E0F7FA'], alpha=0.3)

    # Plot curves using EXACT same data
    ax.plot(geom_data['left_curve'][:, 0], geom_data['left_curve'][:, 1],
           'b-', linewidth=3, label='Left shelf', alpha=0.8)
    ax.plot(geom_data['back_curve'][:, 0], geom_data['back_curve'][:, 1],
           'r-', linewidth=3, label='Back shelf', alpha=0.8)
    ax.plot(geom_data['right_curve'][:, 0], geom_data['right_curve'][:, 1],
           'g-', linewidth=3, label='Right shelf', alpha=0.8)

    # Plot arcs
    ax.plot(geom_data['lb_arc'][:, 0], geom_data['lb_arc'][:, 1],
           'c-', linewidth=4, label='Corner arcs', alpha=0.9)
    ax.plot(geom_data['rb_arc'][:, 0], geom_data['rb_arc'][:, 1],
           'm-', linewidth=4, alpha=0.9)

    # Draw cut lines - ONLY from arc to wall
    lb_cut_y = geom_data['lb_cut_y']
    rb_cut_y = geom_data['rb_cut_y']

    # Find where arcs intersect cut lines
    lb_arc_x = geom_data['lb_arc'][-1][0]  # Last point of arc at cut
    rb_arc_x = geom_data['rb_arc'][-1][0]  # Last point of arc at cut

    ax.plot([0, lb_arc_x], [lb_cut_y, lb_cut_y], 'r--', linewidth=2, alpha=0.8, label='Cut lines')
    ax.plot([rb_arc_x, pantry_width], [rb_cut_y, rb_cut_y], 'r--', linewidth=2, alpha=0.8)

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


def create_height_level_page(ax, height, shelves_at_height, pantry_width=48, pantry_depth=49):
    """Create visualization page showing all shelves at a specific height.

    Args:
        ax: Matplotlib axes
        height: Bottom-referenced height in inches
        shelves_at_height: Dict with keys 'left', 'right', 'back' (each optional) containing polygons
        pantry_width: Pantry width in inches
        pantry_depth: Pantry depth in inches
    """
    # Draw full pantry outline
    ax.plot([0, pantry_width, pantry_width, 0, 0],
           [0, 0, pantry_depth, pantry_depth, 0],
           'k-', linewidth=2, label='Pantry walls')

    # Draw left shelf if present
    if 'left' in shelves_at_height:
        left_poly = shelves_at_height['left']
        ax.plot(left_poly[:, 0], left_poly[:, 1], 'b-', linewidth=2, label='Left shelf')
        ax.fill(left_poly[:, 0], left_poly[:, 1], color='blue', alpha=0.3)

    # Draw right shelf if present
    if 'right' in shelves_at_height:
        right_poly = shelves_at_height['right']
        ax.plot(right_poly[:, 0], right_poly[:, 1], 'g-', linewidth=2, label='Right shelf')
        ax.fill(right_poly[:, 0], right_poly[:, 1], color='green', alpha=0.3)

    # Draw back shelf if present
    if 'back' in shelves_at_height:
        back_poly = shelves_at_height['back']
        ax.plot(back_poly[:, 0], back_poly[:, 1], 'r-', linewidth=2, label='Back shelf')
        ax.fill(back_poly[:, 0], back_poly[:, 1], color='red', alpha=0.3)

    # Add door edge indicator
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Door edge')

    # Add 29" depth line if we have intermediate shelves
    if 'left' in shelves_at_height or 'right' in shelves_at_height:
        if 'back' not in shelves_at_height:
            ax.axhline(y=29, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Intermediate depth (29")')

    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=11)
    ax.set_ylabel('Y (inches)', fontsize=11)
    ax.set_title(f'Shelf Level at {height}" Height (full pantry view)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)


def create_plywood_page(ax, bin_num, placements, bin_width=96, bin_height=48):
    """Create plywood layout page with EXACT geometry at life-size scale."""
    # Draw sheet outline (thinner for printing)
    sheet_rect = mpatches.Rectangle(
        (0, 0), bin_width, bin_height,
        linewidth=0.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(sheet_rect)

    colors = {'L': 'blue', 'R': 'green', 'B': 'red'}

    for b, poly, x, y, rotated, shelf_height, wall_name in placements:
        if b != bin_num:
            continue

        # Position polygon
        poly_placed = poly + np.array([x, y])

        # Draw outline (no fill to avoid obscuring cut lines)
        color = colors.get(wall_name, 'gray')
        ax.plot(poly_placed[:, 0], poly_placed[:, 1], color=color, linewidth=0.8, alpha=0.8)

        # Label - position offset from center toward interior to avoid edges
        center_x = np.mean(poly_placed[:, 0])
        center_y = np.mean(poly_placed[:, 1])

        # Find bounding box to offset label inward
        min_x, max_x = np.min(poly_placed[:, 0]), np.max(poly_placed[:, 0])
        min_y, max_y = np.min(poly_placed[:, 1]), np.max(poly_placed[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        # Offset label toward upper-left interior (away from typical cut edges)
        label_x = center_x - width * 0.15
        label_y = center_y + height * 0.15

        # Create label with wall name and height
        label_text = f"{wall_name}{shelf_height}\""

        ax.text(label_x, label_y, label_text,
               fontsize=8, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, alpha=0.9, linewidth=0.5))

    ax.set_xlim(0, bin_width)
    ax.set_ylim(0, bin_height)
    ax.set_aspect('equal')
    ax.grid(False)  # No grid for cleaner printing
    ax.axis('off')  # Remove axes for clean template

    # Add sheet number label in corner
    ax.text(2, bin_height - 2, f'Sheet {bin_num + 1}',
           fontsize=12, fontweight='bold', ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='black', linewidth=1))


def main():
    print("="*60)
    print("Extracting and Exporting Exact Geometry - Height-Based Organization")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    print(f"Loading config: {config_path}\n")
    config = ShelfConfig.from_file(config_path)

    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define specific heights for each shelf type (bottom-referenced)
    main_shelf_heights = [19, 39, 59, 79]  # Full L-shaped shelves with left-right-back
    left_intermediate_heights = [9, 29, 49, 69]  # Left side only, 7" depth
    right_intermediate_heights = [5, 13, 26, 33, 43, 46, 66, 73, 86]  # Right side only, 4" depth

    # Get the levels from config (assuming they match the main shelf count)
    levels = sorted(set(s['level'] for s in config.shelves))

    # Extract geometry parameters
    amplitude = config.design_params['sinusoid_amplitude']
    period = config.design_params['sinusoid_period']
    left_depth = config.design_params['shelf_base_depth_east']
    right_depth = config.design_params['shelf_base_depth_west']
    shelf_length = 29.0  # 29" from door inward for intermediate shelves
    corner_radius = config.design_params['interior_corner_radius']

    # Storage for all polygons and visualizations
    all_polygons = []  # List of (polygon, height, wall_name) tuples
    shelves_by_height = {}  # Dict of {height: {'left': poly, 'right': poly, 'back': poly}}

    # =================================================================
    # MAIN SHELVES (Full L-shaped with left-right-back portions)
    # =================================================================
    print(f"Generating main shelves at heights: {main_shelf_heights}")
    for i, height in enumerate(main_shelf_heights):
        if i >= len(levels):
            print(f"  Warning: Not enough levels in config for height {height}\"")
            continue

        level = levels[i]
        print(f"  Extracting geometry for height {height}\" (config level {level})...")
        geom_data = extract_exact_shelf_geometries(config, level)

        if not geom_data:
            print(f"    Failed to extract geometry")
            continue

        # Initialize height in shelves_by_height
        if height not in shelves_by_height:
            shelves_by_height[height] = {}

        # Export SVG for each piece with height-based naming
        wall_to_name = {'E': 'L', 'S': 'B', 'W': 'R'}
        for wall in ['E', 'S', 'W']:
            polygon = geom_data[wall]
            wall_name = wall_to_name[wall]

            min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
            min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
            width = max_x - min_x
            poly_height = max_y - min_y

            # Export SVG with height-based naming: shelf_L19_exact.svg
            svg_path = output_dir / f'shelf_{wall_name}{height}_exact.svg'
            export_polygon_to_svg(polygon, svg_path, height, wall_name, width, poly_height)
            print(f"    Exported: {svg_path}")

            # Add to polygons list for packing
            all_polygons.append((polygon, height, wall_name))

            # Store for visualization
            if wall == 'E':
                shelves_by_height[height]['left'] = polygon
            elif wall == 'S':
                shelves_by_height[height]['back'] = polygon
            elif wall == 'W':
                shelves_by_height[height]['right'] = polygon

    # =================================================================
    # LEFT INTERMEDIATE SHELVES (7" depth, 29" length)
    # =================================================================
    print(f"\nGenerating left intermediate shelves at heights: {left_intermediate_heights}")
    np.random.seed(42)  # For reproducible random offsets
    for height in left_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(left_depth, shelf_length, 'E',
                                              amplitude, period, offset, corner_radius)

        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        # Export SVG: shelf_L9_exact.svg
        svg_path = output_dir / f'shelf_L{height}_exact.svg'
        export_polygon_to_svg(polygon, svg_path, height, 'L', width, poly_height)
        print(f"  Exported: {svg_path}")

        # Add to polygons and visualization
        all_polygons.append((polygon, height, 'L'))
        if height not in shelves_by_height:
            shelves_by_height[height] = {}
        shelves_by_height[height]['left'] = polygon

    # =================================================================
    # RIGHT INTERMEDIATE SHELVES (4" depth, 29" length)
    # =================================================================
    print(f"\nGenerating right intermediate shelves at heights: {right_intermediate_heights}")
    for height in right_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(right_depth, shelf_length, 'W',
                                              amplitude, period, offset, corner_radius)

        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        # Export SVG: shelf_R5_exact.svg
        svg_path = output_dir / f'shelf_R{height}_exact.svg'
        export_polygon_to_svg(polygon, svg_path, height, 'R', width, poly_height)
        print(f"  Exported: {svg_path}")

        # Add to polygons and visualization
        all_polygons.append((polygon, height, 'R'))
        if height not in shelves_by_height:
            shelves_by_height[height] = {}
        shelves_by_height[height]['right'] = polygon

    # =================================================================
    # PLYWOOD PACKING
    # =================================================================
    print("\nPacking on plywood...")
    placements = simple_2d_pack(all_polygons)
    num_bins = max(p[0] for p in placements) + 1
    print(f"  Using {num_bins} sheet(s)")
    print(f"  Total pieces: {len(all_polygons)}")

    # =================================================================
    # PDF GENERATION
    # =================================================================
    pdf_path = output_dir / 'exact_cutting_templates.pdf'
    print(f"\nGenerating PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = (
            f'Pantry Shelf Cutting Templates - EXACT GEOMETRY\n\n'
            f'Config: {config.version}\n'
            f'Pantry: {config.pantry["width"]}" × {config.pantry["depth"]}" × {config.pantry["height"]}"\n'
            f'Main Shelves: {len(main_shelf_heights)} at heights {main_shelf_heights}\n'
            f'Left Intermediate: {len(left_intermediate_heights)} at heights {left_intermediate_heights}\n'
            f'Right Intermediate: {len(right_intermediate_heights)} at heights {right_intermediate_heights}\n'
            f'Total Pieces: {len(all_polygons)}\n'
            f'Sheets: {num_bins}\n\n'
            f'Geometry extracted from exact solver\n'
            f'SVG files use identical coordinates'
        )

        ax.text(0.5, 0.5, title_text,
               transform=ax.transAxes, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Visualization pages - one per height level
        print("  Generating height-level visualization pages...")
        for height in sorted(shelves_by_height.keys()):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_height_level_page(ax, height, shelves_by_height[height])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            print(f"    Added visualization for height {height}\"")

        # Plywood pages - LIFE-SIZE at high resolution
        print("  Generating plywood layout pages...")
        for bin_num in range(num_bins):
            # Life-size: 96" x 48" at 150 DPI for high-quality printing
            fig = plt.figure(figsize=(96, 48), dpi=150)
            ax = fig.add_subplot(111)
            create_plywood_page(ax, bin_num, placements)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Added plywood sheet {bin_num + 1} (life-size, 150 DPI)")

    print(f"\nDone!")
    print(f"Total shelf heights: {len(shelves_by_height)}")
    print(f"Total pieces: {len(all_polygons)}")
    print("="*60)


if __name__ == '__main__':
    main()
