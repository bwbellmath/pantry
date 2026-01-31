"""
Geometric calculations for sinusoidal shelf edges and corner radiusing.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize


def sinusoid_depth(position: float, base_depth: float, amplitude: float,
                   period: float, offset: float) -> float:
    """
    Calculate depth at a position along a sinusoidal edge.

    Args:
        position: Position along the wall
        base_depth: Base shelf depth
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Phase offset

    Returns:
        Depth at the given position
    """
    return base_depth + amplitude * np.sin(2 * np.pi * position / period + offset)


def sinusoid_depth_derivative(position: float, amplitude: float,
                              period: float, offset: float) -> float:
    """
    Calculate the derivative of depth with respect to position.

    Args:
        position: Position along the wall
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Phase offset

    Returns:
        Derivative dd/dposition
    """
    return amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * position / period + offset)


def generate_sinusoid_points(extent_start: float, extent_end: float,
                            base_depth: float, amplitude: float,
                            period: float, offset: float,
                            num_points: int = 100) -> np.ndarray:
    """
    Generate points along a sinusoidal curve.

    Args:
        extent_start: Start position along wall
        extent_end: End position along wall
        base_depth: Base shelf depth
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Phase offset
        num_points: Number of points to generate

    Returns:
        Array of shape (num_points, 2) with [position, depth] pairs
    """
    positions = np.linspace(extent_start, extent_end, num_points)
    depths = sinusoid_depth(positions, base_depth, amplitude, period, offset)
    return np.column_stack([positions, depths])


def solve_interior_corner(depth1: float, depth2: float, radius: float,
                         corner_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for interior corner (South wall corners) where material is added.

    The circle is positioned at 45° angle, radius distance from the corner point.

    Args:
        depth1: Depth of first shelf (East or West) at the corner
        depth2: Depth of second shelf (South) at the corner
        radius: Corner radius (6 inches)
        corner_type: 'E' for East-South, 'W' for West-South

    Returns:
        Tuple of (arc_center, arc_start, arc_end) in local coordinates
    """
    # The corner point is where the two shelves would meet
    # For East-South: corner is at (depth_east, pantry_depth) in pantry coords
    # But we work in relative coordinates here

    # Position arc center at 45° angle, radius away from corner
    # 45° means equal distance in both directions
    offset = radius / np.sqrt(2)  # radius * cos(45°) = radius * sin(45°)

    if corner_type == 'E':
        # East-South corner: add material extending from east shelf
        # Arc center is offset inward (positive x, negative y from corner)
        arc_center = np.array([offset, -offset])
    else:  # 'W'
        # West-South corner: add material extending from west shelf
        # Arc center is offset inward (negative x, negative y from corner)
        arc_center = np.array([-offset, -offset])

    # Arc spans from one shelf edge to the other
    # From 135° to 180° for East-South
    # From 0° to 45° for West-South
    if corner_type == 'E':
        angles = np.linspace(np.pi * 3/4, np.pi, 20)  # 135° to 180°
    else:
        angles = np.linspace(0, np.pi/4, 20)  # 0° to 45°

    # Calculate arc points
    arc_points = arc_center[:, np.newaxis] + radius * np.array([
        np.cos(angles),
        np.sin(angles)
    ])

    arc_start = arc_points[:, 0]
    arc_end = arc_points[:, -1]

    return arc_center, arc_start, arc_end


def solve_door_corner(depth: float, radius: float, corner_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for door corner (North wall corners) where material is removed.

    Args:
        depth: Depth of shelf (East or West) at the door
        radius: Corner radius (3 inches)
        corner_type: 'E' for East-North, 'W' for West-North

    Returns:
        Tuple of (arc_center, arc_points) in local coordinates
    """
    # Remove material to create rounded corner
    # Arc center is positioned radius away from both edges

    if corner_type == 'E':
        # East-North corner: remove material from northeast corner
        # Arc center at (radius, radius) from origin (0, 0)
        arc_center = np.array([radius, radius])
        # Arc from (depth, 0) curving to (0, depth)
        # Angles from -90° to 0° (fourth quadrant to positive x-axis)
        angles = np.linspace(-np.pi/2, 0, 20)
    else:  # 'W'
        # West-North corner: remove material from northwest corner
        # Arc center at (-radius, radius) from origin (0, 0)
        arc_center = np.array([-radius, radius])
        # Arc from (0, depth) curving to (-depth, 0)
        # Angles from 90° to 180° (positive y-axis to negative x-axis)
        angles = np.linspace(np.pi/2, np.pi, 20)

    # Calculate arc points
    arc_points = arc_center[:, np.newaxis] + radius * np.array([
        np.cos(angles),
        np.sin(angles)
    ])

    return arc_center, arc_points


class CornerSolver:
    """Solves for corner radius geometry where shelves meet."""

    def __init__(self, interior_radius: float, door_radius: float):
        """
        Initialize corner solver.

        Args:
            interior_radius: Interior corner radius (6 inches, South wall)
            door_radius: Door corner radius (3 inches, North wall)
        """
        self.interior_radius = interior_radius
        self.door_radius = door_radius

    def solve_corner_subtractive(self, position: float, depth: float,
                                 wall_orientation: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for a corner where material is removed (door side corners).

        For East/West walls meeting North (door) wall, we remove material
        to create a rounded corner.

        Args:
            position: Position where corner occurs
            depth: Depth of shelf at corner position
            wall_orientation: 'E' or 'W'

        Returns:
            Tuple of (arc_center, arc_points)
            - arc_center: [x, y] position of arc center
            - arc_points: Array of [x, y] points along the arc
        """
        if wall_orientation == 'E':
            # East wall: shelf extends from x=0 towards positive x
            # Corner at (depth, 0) - northeast corner
            # Arc center is at (radius, radius)
            arc_center = np.array([self.radius, self.radius])
            # Arc from (depth, 0) to (0, depth), curving inward
            angles = np.linspace(-np.pi/2, 0, 20)
            arc_points = arc_center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

        else:  # 'W'
            # West wall: shelf extends from x=width towards negative x (towards center)
            # Corner at (width - depth, 0) - northwest corner
            # Arc center is positioned relative to pantry width
            # This will be handled in the shelf generator with proper pantry coordinates
            arc_center = np.array([-self.radius, self.radius])
            angles = np.linspace(np.pi, np.pi/2, 20)
            arc_points = arc_center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

        return arc_center, arc_points

    def solve_corner_additive(self, pos1: float, depth1: float,
                             pos2: float, depth2: float,
                             wall_orientation: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for a corner where material is added (South wall corners).

        For East/West walls meeting South wall, we add material to create
        a smooth transition.

        Args:
            pos1: Position on first wall (East or West)
            depth1: Depth of first wall shelf at junction
            pos2: Position on second wall (South)
            depth2: Depth of second wall shelf at junction
            wall_orientation: 'E' or 'W' (which wall is being processed)

        Returns:
            Tuple of (arc_center, arc_points)
            - arc_center: [x, y] position of arc center
            - arc_points: Array of [x, y] points along the arc
        """
        # For additive corners, the arc fills the gap between two shelves
        # The center is positioned such that the arc is tangent to both shelf edges

        if wall_orientation == 'E':
            # East wall meets South wall at southeast corner
            # Arc center is at (depth1 + radius, pos1 - radius)
            arc_center = np.array([depth1 + self.radius, pos1 - self.radius])
            # Arc curves outward from the corner
            angles = np.linspace(np.pi/2, np.pi, 20)
            arc_points = arc_center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

        else:  # 'W'
            # West wall meets South wall at southwest corner
            # Similar logic but mirrored
            arc_center = np.array([-depth1 - self.radius, pos1 - self.radius])
            angles = np.linspace(0, np.pi/2, 20)
            arc_points = arc_center + self.radius * np.column_stack([np.cos(angles), np.sin(angles)])

        return arc_center, arc_points


def wall_to_pantry_coords(wall_position: float, depth: float,
                          wall_type: str, pantry_width: float,
                          pantry_depth: float) -> Tuple[float, float]:
    """
    Convert wall-relative coordinates to pantry coordinates.

    Args:
        wall_position: Position along the wall
        depth: Depth from the wall
        wall_type: 'E', 'S', or 'W'
        pantry_width: Width of pantry (x-dimension)
        pantry_depth: Depth of pantry (y-dimension)

    Returns:
        (x, y) coordinates in pantry space
    """
    if wall_type == 'E':
        # East wall: x=0, position is y-coordinate, depth extends in +x
        return (depth, wall_position)
    elif wall_type == 'S':
        # South wall: y=depth, position is x-coordinate, depth extends in -y
        return (wall_position, pantry_depth - depth)
    elif wall_type == 'W':
        # West wall: x=width, position is y-coordinate, depth extends in -x
        return (pantry_width - depth, wall_position)
    else:
        raise ValueError(f"Unknown wall type: {wall_type}")


def pantry_to_wall_coords(x: float, y: float, wall_type: str,
                         pantry_width: float, pantry_depth: float) -> Tuple[float, float]:
    """
    Convert pantry coordinates to wall-relative coordinates.

    Args:
        x: X-coordinate in pantry space
        y: Y-coordinate in pantry space
        wall_type: 'E', 'S', or 'W'
        pantry_width: Width of pantry (x-dimension)
        pantry_depth: Depth of pantry (y-dimension)

    Returns:
        (position_along_wall, depth_from_wall)
    """
    if wall_type == 'E':
        return (y, x)
    elif wall_type == 'S':
        return (x, pantry_depth - y)
    elif wall_type == 'W':
        return (y, pantry_width - x)
    else:
        raise ValueError(f"Unknown wall type: {wall_type}")


def create_shelf_outline(wall_type: str, extent_start: float, extent_end: float,
                        base_depth: float, amplitude: float, period: float,
                        offset: float, pantry_width: float, pantry_depth: float,
                        num_points: int = 100) -> np.ndarray:
    """
    Create the outline of a shelf section in pantry coordinates.

    This generates the outer edge (sinusoidal) and inner edge (straight wall)
    of a shelf section.

    Args:
        wall_type: 'E', 'S', or 'W'
        extent_start: Start position along wall
        extent_end: End position along wall
        base_depth: Base shelf depth
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Phase offset
        pantry_width: Width of pantry
        pantry_depth: Depth of pantry
        num_points: Number of points along sinusoid

    Returns:
        Array of [x, y] points forming a closed polygon
    """
    # Generate sinusoidal outer edge in wall coordinates
    wall_points = generate_sinusoid_points(extent_start, extent_end,
                                          base_depth, amplitude, period,
                                          offset, num_points)

    # Convert to pantry coordinates
    pantry_points = []
    for pos, depth in wall_points:
        x, y = wall_to_pantry_coords(pos, depth, wall_type, pantry_width, pantry_depth)
        pantry_points.append([x, y])

    # Add inner edge points (along the wall)
    # These are the same positions but with depth=0
    inner_points = []
    for pos in [extent_end, extent_start]:  # Reverse order to close polygon
        x, y = wall_to_pantry_coords(pos, 0, wall_type, pantry_width, pantry_depth)
        inner_points.append([x, y])

    # Combine: outer edge + inner edge to form closed polygon
    all_points = pantry_points + inner_points

    return np.array(all_points)


def solve_tangent_circle_two_sinusoids(
    pos1_init: float, base_depth1: float, amplitude1: float, period1: float, offset1: float,
    pos2_init: float, base_depth2: float, amplitude2: float, period2: float, offset2: float,
    radius: float,
    wall1_type: str, wall2_type: str,
    pantry_width: float, pantry_depth: float,
    corner_type: str = 'left-back',
    left_params: Optional[Tuple] = None,
    right_params: Optional[Tuple] = None,
    back_params: Optional[Tuple] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Solve for a circle of given radius that is tangent to two sinusoid curves.

    This finds a circle that:
    1. Has the specified radius
    2. Intersects both sinusoid curves at exactly 2 points (one on each curve)
    3. Is tangent (matching slopes) to both curves at the intersection points

    Args:
        pos1_init: Initial guess for position on first sinusoid
        base_depth1: Base depth of first sinusoid
        amplitude1: Amplitude of first sinusoid
        period1: Period of first sinusoid
        offset1: Phase offset of first sinusoid
        pos2_init: Initial guess for position on second sinusoid
        base_depth2: Base depth of second sinusoid
        amplitude2: Amplitude of second sinusoid
        period2: Period of second sinusoid
        offset2: Phase offset of second sinusoid
        radius: Circle radius (3 inches for door corners)
        wall1_type: Wall type for first sinusoid ('L'=left, 'B'=back, 'R'=right)
        wall2_type: Wall type for second sinusoid ('L'=left, 'B'=back, 'R'=right)
        pantry_width: Pantry width
        pantry_depth: Pantry depth
        corner_type: Type of corner ('left-back' or 'right-back')
        left_params: Tuple of (base_depth, amplitude, period, offset) for left shelf
        right_params: Tuple of (base_depth, amplitude, period, offset) for right shelf
        back_params: Tuple of (base_depth, amplitude, period, offset) for back shelf

    Returns:
        Tuple of (center, point1, point2, pos1, pos2):
        - center: [x, y] position of circle center in pantry coordinates
        - point1: [x, y] tangent point on first curve in pantry coordinates
        - point2: [x, y] tangent point on second curve in pantry coordinates
        - pos1: Position along first wall where tangent occurs
        - pos2: Position along second wall where tangent occurs
    """

    def get_point_and_tangent(pos: float, base_depth: float, amplitude: float,
                             period: float, offset: float, wall_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get point and tangent direction in pantry coordinates.

        Wall types: 'L' = left (x≈0), 'B' = back (y≈depth), 'R' = right (x≈width)
        """
        # Calculate depth from sinusoid
        depth = sinusoid_depth(pos, base_depth, amplitude, period, offset)

        # Map new wall types to old for coordinate conversion
        wall_type_map = {'L': 'E', 'B': 'S', 'R': 'W'}
        old_wall_type = wall_type_map.get(wall_type, wall_type)

        # Get point in pantry coordinates
        x, y = wall_to_pantry_coords(pos, depth, old_wall_type, pantry_width, pantry_depth)
        point = np.array([x, y])

        # Calculate tangent direction in wall coordinates
        # For wall coordinates: horizontal is position, vertical is depth
        ddepth_dpos = sinusoid_depth_derivative(pos, amplitude, period, offset)

        # Convert tangent direction to pantry coordinates
        if wall_type in ['L', 'E']:
            # Left wall: x = depth, y = position
            # d/dpos [depth, position] = [ddepth_dpos, 1]
            tangent = np.array([ddepth_dpos, 1.0])
        elif wall_type in ['B', 'S']:
            # Back wall: x = position, y = pantry_depth - depth
            # d/dpos [position, pantry_depth - depth] = [1, -ddepth_dpos]
            tangent = np.array([1.0, -ddepth_dpos])
        elif wall_type in ['R', 'W']:
            # Right wall: x = pantry_width - depth, y = position
            # d/dpos [pantry_width - depth, position] = [-ddepth_dpos, 1]
            tangent = np.array([-ddepth_dpos, 1.0])
        else:
            raise ValueError(f"Unknown wall type: {wall_type}")

        # Normalize tangent
        tangent = tangent / np.linalg.norm(tangent)

        return point, tangent

    def objective(params: np.ndarray) -> float:
        """
        Objective function to minimize. params = [pos1, pos2]

        We want to find positions on the two sinusoids such that:
        1. There exists a circle of radius R tangent to both at these points
        2. The circle center is uniquely determined by the two tangency constraints
        3. The circle center is in the interior region (if params provided)
        """
        pos1, pos2 = params

        # Get points and tangents
        point1, tangent1 = get_point_and_tangent(pos1, base_depth1, amplitude1, period1, offset1, wall1_type)
        point2, tangent2 = get_point_and_tangent(pos2, base_depth2, amplitude2, period2, offset2, wall2_type)

        # Normal directions (perpendicular to tangents, pointing inward/outward)
        # Rotate tangent by 90 degrees: [x, y] -> [-y, x] or [y, -x]
        normal1_option1 = np.array([-tangent1[1], tangent1[0]])
        normal1_option2 = np.array([tangent1[1], -tangent1[0]])

        normal2_option1 = np.array([-tangent2[1], tangent2[0]])
        normal2_option2 = np.array([tangent2[1], -tangent2[0]])

        # Try both normal directions for each point (4 combinations)
        # The circle center must be at distance R along the normal from each point
        # center1 = point1 + R * normal1
        # center2 = point2 + R * normal2
        # We want center1 = center2

        best_error = float('inf')
        has_interior_solution = False

        # Check if we can verify interior constraint
        can_check_interior = (left_params is not None and
                             right_params is not None and
                             back_params is not None)

        for n1 in [normal1_option1, normal1_option2]:
            for n2 in [normal2_option1, normal2_option2]:
                center1 = point1 + radius * n1
                center2 = point2 + radius * n2
                center = (center1 + center2) / 2

                # Error is the distance between the two center estimates
                error = np.linalg.norm(center1 - center2)

                # Check if center is in interior
                if can_check_interior:
                    in_interior = is_point_in_interior(
                        center,
                        left_params[0], left_params[1], left_params[2], left_params[3],
                        right_params[0], right_params[1], right_params[2], right_params[3],
                        back_params[0], back_params[1], back_params[2], back_params[3],
                        pantry_width, pantry_depth
                    )
                    if in_interior:
                        has_interior_solution = True
                        # Prefer interior solutions
                        if error < best_error:
                            best_error = error
                    else:
                        # Heavily penalize non-interior solutions
                        error += 1000.0
                        if error < best_error:
                            best_error = error
                else:
                    if error < best_error:
                        best_error = error

        return best_error

    # For back corners, try multiple starting points and pick the best solution
    # where point2 is on the correct side of the sinusoid
    if corner_type in ['left-back', 'right-back']:
        best_result = None
        best_score = float('inf')
        back_centerline_y = pantry_depth - base_depth2

        # Try multiple starting points
        for offset_mult in [0.5, 1.0, 1.5, 2.0, 2.5]:
            if wall2_type == 'B':  # pos2 is x-coordinate on back wall
                if corner_type == 'left-back':
                    x0_pos2 = base_depth1 + offset_mult * radius
                else:  # right-back
                    x0_pos2 = pantry_width - base_depth1 - offset_mult * radius
            else:
                x0_pos2 = pos2_init

            x0 = np.array([pos1_init, x0_pos2])
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})

            if result.success:
                pos1_test, pos2_test = result.x
                point2_test, _ = get_point_and_tangent(pos2_test, base_depth2, amplitude2, period2, offset2, wall2_type)

                # Score: prefer solutions where point2 is on front side (y < centerline)
                # and has low optimization error
                dist_from_centerline = point2_test[1] - back_centerline_y
                if dist_from_centerline < 0:  # Front side (desired)
                    score = result.fun + 0.1 * abs(dist_from_centerline)
                else:  # Back side (penalize heavily)
                    score = result.fun + 1000.0 * dist_from_centerline

                if score < best_score:
                    best_score = score
                    best_result = result

        if best_result is None:
            # Fallback to single optimization
            x0 = np.array([pos1_init, pos2_init])
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
        else:
            result = best_result
    else:
        # Original single optimization for non-back corners
        x0 = np.array([pos1_init, pos2_init])
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    # Extract solution
    pos1_opt, pos2_opt = result.x

    # Reconstruct the solution
    point1, tangent1 = get_point_and_tangent(pos1_opt, base_depth1, amplitude1, period1, offset1, wall1_type)
    point2, tangent2 = get_point_and_tangent(pos2_opt, base_depth2, amplitude2, period2, offset2, wall2_type)


    # Find the correct normals and center
    normal1_option1 = np.array([-tangent1[1], tangent1[0]])
    normal1_option2 = np.array([tangent1[1], -tangent1[0]])
    normal2_option1 = np.array([-tangent2[1], tangent2[0]])
    normal2_option2 = np.array([tangent2[1], -tangent2[0]])

    best_error = float('inf')
    best_center = None

    # Check if we can verify interior constraint
    can_check_interior = (left_params is not None and
                         right_params is not None and
                         back_params is not None)

    # For back corners (left-back, right-back), ALWAYS use corner_type method
    # because the interior check doesn't work correctly for these cases
    use_corner_type_method = (corner_type in ['left-back', 'right-back'])

    # If no interior check available OR if we should use corner_type method, use corner_type to determine correct normals
    if not can_check_interior or use_corner_type_method:
        # For interior corners, we need specific normal directions:
        # - left-back: left normal points right (+x), back normal points down (-y)
        # - right-back: right normal points left (-x), back normal points down (-y)

        if corner_type == 'left-back':
            # Left sinusoid: choose normal pointing right (positive x direction)
            # Back sinusoid: choose normal pointing down (negative y direction)
            n1 = normal1_option1 if normal1_option1[0] > 0 else normal1_option2
            n2 = normal2_option1 if normal2_option1[1] < 0 else normal2_option2
        elif corner_type == 'right-back':
            # Right sinusoid: choose normal pointing left (negative x direction)
            # Back sinusoid: choose normal pointing down (negative y direction)
            n1 = normal1_option1 if normal1_option1[0] < 0 else normal1_option2
            n2 = normal2_option1 if normal2_option1[1] < 0 else normal2_option2
        else:
            # Fallback: try all combinations
            n1, n2 = None, None

        if n1 is not None and n2 is not None:
            center1 = point1 + radius * n1
            center2 = point2 + radius * n2
            best_center = (center1 + center2) / 2
            best_error = np.linalg.norm(center1 - center2)

    # Only try all combinations if we have interior check or if corner_type didn't give us a solution
    if best_center is None:
        for n1 in [normal1_option1, normal1_option2]:
            for n2 in [normal2_option1, normal2_option2]:
                center1 = point1 + radius * n1
                center2 = point2 + radius * n2
                center = (center1 + center2) / 2
                error = np.linalg.norm(center1 - center2)

                # Check if center is in interior region
                valid_orientation = True
                if can_check_interior:
                    valid_orientation = is_point_in_interior(
                        center,
                        left_params[0], left_params[1], left_params[2], left_params[3],
                        right_params[0], right_params[1], right_params[2], right_params[3],
                        back_params[0], back_params[1], back_params[2], back_params[3],
                        pantry_width, pantry_depth
                    )

                # Only consider solutions with valid orientation (in interior)
                if valid_orientation and error < best_error:
                    best_error = error
                    best_center = center

        if best_center is None:
            # Final fallback if no valid interior solution found
            print("Warning: No valid interior solution found, using best geometric match")
            for n1 in [normal1_option1, normal1_option2]:
                for n2 in [normal2_option1, normal2_option2]:
                    center1 = point1 + radius * n1
                    center2 = point2 + radius * n2
                    error = np.linalg.norm(center1 - center2)
                    if error < best_error:
                        best_error = error
                        best_center = (center1 + center2) / 2

    return best_center, point1, point2, pos1_opt, pos2_opt


def generate_circle_arc(center: np.ndarray, point1: np.ndarray, point2: np.ndarray,
                       num_points: int = 50, interior_arc: bool = True) -> np.ndarray:
    """
    Generate points along a circular arc from point1 to point2.

    Always generates the SHORTER arc by:
    1. Ordering the points counter-clockwise
    2. Drawing the arc clockwise between them

    Args:
        center: [x, y] center of circle
        point1: [x, y] start point on circle
        point2: [x, y] end point on circle
        num_points: Number of points to generate along arc
        interior_arc: Kept for compatibility but not used (always returns shorter arc)

    Returns:
        Array of [x, y] points along the arc
    """
    # Calculate angles for both points
    angle1 = np.arctan2(point1[1] - center[1], point1[0] - center[0])
    angle2 = np.arctan2(point2[1] - center[1], point2[0] - center[0])

    # Normalize angles to [0, 2π]
    if angle1 < 0:
        angle1 += 2 * np.pi
    if angle2 < 0:
        angle2 += 2 * np.pi

    # Order points counter-clockwise (increasing angles)
    # If angle2 < angle1, swap them so we go counter-clockwise
    if angle2 < angle1:
        angle1, angle2 = angle2, angle1
        point1, point2 = point2, point1

    # Now we have angle1 < angle2 (counter-clockwise order)
    # Draw the arc clockwise (decreasing angles) from angle1 to angle2
    # This means going the "short way" which is actually just angle1 to angle2
    # in the positive direction, which is the shorter of the two possible arcs

    # Check which direction is shorter
    ccw_distance = angle2 - angle1  # Counter-clockwise distance
    cw_distance = (2 * np.pi) - ccw_distance  # Clockwise distance

    if ccw_distance <= cw_distance:
        # Going counter-clockwise (increasing angles) is shorter
        angles = np.linspace(angle1, angle2, num_points)
    else:
        # Going clockwise (through 0/2π) is shorter
        angles = np.linspace(angle1, angle2 - 2*np.pi, num_points)

    # Calculate radius
    radius = np.linalg.norm(point1 - center)

    # Generate arc points
    arc_points = center[:, np.newaxis] + radius * np.array([
        np.cos(angles),
        np.sin(angles)
    ])

    return arc_points.T


def is_point_in_interior(point: np.ndarray,
                        left_base_depth: float, left_amplitude: float, left_period: float, left_offset: float,
                        right_base_depth: float, right_amplitude: float, right_period: float, right_offset: float,
                        back_base_depth: float, back_amplitude: float, back_period: float, back_offset: float,
                        pantry_width: float, pantry_depth: float) -> bool:
    """
    Check if a point is in the interior (usable) region of the pantry.

    Interior region is defined by:
    - x > left_shelf_edge(y)  (to the right of left shelf)
    - x < right_shelf_edge(y) (to the left of right shelf)
    - y < back_shelf_edge(x)  (in front of back shelf)
    - y > 0                   (behind the door)

    Args:
        point: [x, y] coordinates
        left_base_depth, left_amplitude, left_period, left_offset: Left shelf sinusoid params
        right_base_depth, right_amplitude, right_period, right_offset: Right shelf sinusoid params
        back_base_depth, back_amplitude, back_period, back_offset: Back shelf sinusoid params
        pantry_width: Pantry width
        pantry_depth: Pantry depth

    Returns:
        True if point is in interior region
    """
    x, y = point

    # Left shelf edge: x = depth_at_y
    left_edge = sinusoid_depth(y, left_base_depth, left_amplitude, left_period, left_offset)

    # Right shelf edge: x = pantry_width - depth_at_y
    right_depth = sinusoid_depth(y, right_base_depth, right_amplitude, right_period, right_offset)
    right_edge = pantry_width - right_depth

    # Back shelf edge: y = pantry_depth - depth_at_x
    back_depth = sinusoid_depth(x, back_base_depth, back_amplitude, back_period, back_offset)
    back_edge = pantry_depth - back_depth

    # Check all constraints
    return (x > left_edge and
            x < right_edge and
            y < back_edge and
            y > 0)


def generate_interior_mask(left_base_depth: float, left_amplitude: float, left_period: float, left_offset: float,
                          right_base_depth: float, right_amplitude: float, right_period: float, right_offset: float,
                          back_base_depth: float, back_amplitude: float, back_period: float, back_offset: float,
                          pantry_width: float, pantry_depth: float,
                          resolution: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a mask array indicating interior vs shelf regions.

    Returns:
        Tuple of (X, Y, mask) where mask is True for interior regions
    """
    x = np.linspace(0, pantry_width, resolution)
    y = np.linspace(0, pantry_depth, resolution)
    X, Y = np.meshgrid(x, y)

    mask = np.zeros_like(X, dtype=bool)

    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            mask[i, j] = is_point_in_interior(
                point,
                left_base_depth, left_amplitude, left_period, left_offset,
                right_base_depth, right_amplitude, right_period, right_offset,
                back_base_depth, back_amplitude, back_period, back_offset,
                pantry_width, pantry_depth
            )

    return X, Y, mask


# ============================================================================
# NEW ROBUST TANGENT CIRCLE SOLVER - Newton's Method
# ============================================================================

def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _short_arc_span(theta_a: float, theta_b: float) -> float:
    """Calculate the shorter arc span between two angles."""
    return abs(_wrap_to_pi(theta_b - theta_a))


def _newton_2d(
    F, x0: Tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 60,
    fd_eps: float = 1e-7
) -> Optional[Tuple[float, float]]:
    """2D Newton with finite-difference Jacobian + backtracking."""
    x, y = x0

    def norm2(v):
        return math.hypot(v[0], v[1])

    fx, fy = F(x, y)
    fn = norm2((fx, fy))

    for _ in range(max_iter):
        if fn < tol:
            return (x, y)

        # FD Jacobian J = [[dFx/dx, dFx/dy],[dFy/dx, dFy/dy]]
        hx = fd_eps * (1.0 + abs(x))
        hy = fd_eps * (1.0 + abs(y))

        fpx = F(x + hx, y)
        fmx = F(x - hx, y)
        dFdx = ((fpx[0] - fmx[0]) / (2 * hx), (fpx[1] - fmx[1]) / (2 * hx))

        fpy = F(x, y + hy)
        fmy = F(x, y - hy)
        dFdy = ((fpy[0] - fmy[0]) / (2 * hy), (fpy[1] - fmy[1]) / (2 * hy))

        a, c = dFdx  # dFx/dx, dFy/dx
        b, d = dFdy  # dFx/dy, dFy/dy

        det = a * d - b * c
        if abs(det) < 1e-18:
            return None

        # Solve J * [dx, dy]^T = -F via 2x2 inverse
        dx = (-fx * d + b * fy) / det
        dy = ( c * fx - a * fy) / det

        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ls in range(20):
            xt = x + alpha * dx
            yt = y + alpha * dy
            g0, g1 = F(xt, yt)
            gn = norm2((g0, g1))
            if gn < fn:
                x, y = xt, yt
                fx, fy = g0, g1
                fn = gn
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return None

    return None


def tangent_circle_two_sinusoids_offset_intersection(
    Av: float, x_pos: float, y_off_v: float,
    Ah: float, y_pos: float, x_off_h: float,
    r: float,
    quadrant: str,
    *,
    tol: float = 1e-12
) -> Dict[str, object]:
    """
    Find tangent circle to two sinusoids using Newton's method.
    
    Horizontal sinusoid: y = y_pos + Ah * sin(x - x_off_h)
    Vertical   sinusoid: x = x_pos + Av * sin(y - y_off_v)

    Args:
        Av: Amplitude of vertical sinusoid
        x_pos: Base position of vertical sinusoid
        y_off_v: Phase offset of vertical sinusoid
        Ah: Amplitude of horizontal sinusoid
        y_pos: Base position of horizontal sinusoid
        x_off_h: Phase offset of horizontal sinusoid
        r: Radius of tangent circle
        quadrant: Which tangent circle: 'br', 'tr', 'tl', 'bl'
        tol: Convergence tolerance

    Returns:
      - center: (cx, cy)
      - points: ordered tangency points per quadrant rule
      - short_arc_span: radians of the shorter arc between them
      - angles: (theta1, theta2) angles to tangency points
      - raw_points: dict with 'vertical' and 'horizontal' tangency points
      - quadrant: the quadrant used
    """

    quadrant = quadrant.lower().strip()
    if quadrant not in {"br", "tr", "tl", "bl"}:
        raise ValueError("quadrant must be one of {'br','tr','tl','bl'}")
    if r <= 0:
        raise ValueError("r must be positive")

    # Side selection from quadrant
    # sh: +1 above horizontal curve, -1 below
    # sv: +1 right of vertical curve, -1 left
    sh = +1.0 if quadrant in {"tr", "tl"} else -1.0
    sv = +1.0 if quadrant in {"tr", "br"} else -1.0

    # Horizontal curve point + unit normal
    def horiz_point_and_unit_normal(x: float):
        y = y_pos + Ah * math.sin(x - x_off_h)
        yp = Ah * math.cos(x - x_off_h)  # dy/dx
        # normal direction (-yp, 1)
        inv = 1.0 / math.sqrt(1.0 + yp * yp)
        nx, ny = (-yp * inv, 1.0 * inv)
        return (x, y), (nx, ny), yp

    # Vertical curve point + unit normal
    def vert_point_and_unit_normal(y: float):
        x = x_pos + Av * math.sin(y - y_off_v)
        xp = Av * math.cos(y - y_off_v)  # dx/dy
        # normal direction (1, -xp)
        inv = 1.0 / math.sqrt(1.0 + xp * xp)
        mx, my = (1.0 * inv, -xp * inv)
        return (x, y), (mx, my), xp

    # Offset-center curves:
    def Ch(x: float):
        P, n, _ = horiz_point_and_unit_normal(x)
        return (P[0] + sh * r * n[0], P[1] + sh * r * n[1])

    def Cv(y: float):
        P, m, _ = vert_point_and_unit_normal(y)
        return (P[0] + sv * r * m[0], P[1] + sv * r * m[1])

    # Solve Ch(xh) - Cv(yv) = 0
    def F(xh: float, yv: float):
        cxh, cyh = Ch(xh)
        cxv, cyv = Cv(yv)
        return (cxh - cxv, cyh - cyv)

    # Initial guess: pretend both curves are locally flat at midlines
    # => center is (x_pos +/- r, y_pos +/- r)
    cx0 = x_pos + sv * r
    cy0 = y_pos + sh * r
    xh0 = cx0
    yv0 = cy0

    # Try a few 2π shifts because sinusoids repeat
    # Use more shifts for robustness
    shifts = [0.0, 2.0 * math.pi, -2.0 * math.pi, 4.0 * math.pi, -4.0 * math.pi]
    best = None
    best_score = float("inf")

    for sx in shifts:
        for sy in shifts:
            guess = (xh0 + sx, yv0 + sy)
            sol = _newton_2d(F, guess, tol=tol)
            if sol is None:
                continue
            xh, yv = sol
            cxh, cyh = Ch(xh)

            # Quadrant sanity vs midlines
            if sv > 0 and not (cxh > x_pos): 
                continue
            if sv < 0 and not (cxh < x_pos):
                continue
            if sh > 0 and not (cyh > y_pos):
                continue
            if sh < 0 and not (cyh < y_pos):
                continue

            score = (cxh - cx0) ** 2 + (cyh - cy0) ** 2
            if score < best_score:
                best_score = score
                best = (xh, yv)

    if best is None:
        raise RuntimeError("No solution found (try a different branch/initialization or check radius feasibility).")

    xh, yv = best
    (xh_p, yh_p), _, _ = horiz_point_and_unit_normal(xh)
    (xv_p, yv_p), _, _ = vert_point_and_unit_normal(yv)

    # Both computed centers should match; use horizontal one
    cx, cy = Ch(xh)

    p_horz = (xh_p, yh_p)   # tangency on horizontal sinusoid
    p_vert = (xv_p, yv_p)   # tangency on vertical sinusoid

    # Order rule you specified:
    # br: vert then horiz
    # tr: horiz then vert
    # bl: horiz then vert
    # tl: vert then horiz
    if quadrant in {"br", "tl"}:
        ordered = [p_vert, p_horz]
    else:
        ordered = [p_horz, p_vert]

    theta1 = math.atan2(ordered[0][1] - cy, ordered[0][0] - cx)
    theta2 = math.atan2(ordered[1][1] - cy, ordered[1][0] - cx)
    span = _short_arc_span(theta1, theta2)

    return {
        "center": (cx, cy),
        "points": ordered,
        "short_arc_span": span,
        "angles": (theta1, theta2),
        "raw_points": {"vertical": p_vert, "horizontal": p_horz},
        "quadrant": quadrant,
    }


def solve_tangent_circle_two_sinusoids_newton(
    base_depth1: float, amplitude1: float, period1: float, offset1: float, wall1_type: str,
    base_depth2: float, amplitude2: float, period2: float, offset2: float, wall2_type: str,
    radius: float,
    pantry_width: float, pantry_depth: float,
    corner_type: str,
    debug: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper to use the new Newton solver with our existing parameter format.
    
    Args:
        base_depth1: Base depth of first sinusoid (offset from wall)
        amplitude1: Amplitude of first sinusoid
        period1: Period of first sinusoid
        offset1: Phase offset of first sinusoid
        wall1_type: 'L' (left/east), 'R' (right/west), or 'B' (back/south)
        base_depth2: Base depth of second sinusoid
        amplitude2: Amplitude of second sinusoid
        period2: Period of second sinusoid
        offset2: Phase offset of second sinusoid
        wall2_type: 'L', 'R', or 'B'
        radius: Radius of tangent circle
        pantry_width: Width of pantry
        pantry_depth: Depth of pantry
        corner_type: 'left-back' or 'right-back'
    
    Returns:
        Tuple of (center, point1, point2) as numpy arrays
    """
    
    # For left-back corner:
    if corner_type == 'left-back':
        # Vertical (left) sinusoid: x = base_depth1 + amplitude1 * sin(2π*y/period1 - offset1)
        # In scaled form: x = base_depth1 + amplitude1 * sin(y_scaled - offset1) where y_scaled = 2π*y/period1
        # But the new solver works in unscaled coordinates, so:
        # x = base_depth1 + amplitude1 * sin(2π/period1 * (y - offset1*period1/(2π)))
        x_pos = base_depth1
        Av = amplitude1
        # For phase: we want sin(2π*y/period - offset) = sin(2π/period * (y - offset*period/(2π)))
        y_off_v = offset1 * period1 / (2 * math.pi)
        
        # Horizontal (back) sinusoid: y = (pantry_depth - base_depth2) - amplitude2 * sin(2π*x/period2 - offset2)
        # Note the negative sign! y = y_pos - amplitude * sin(...)
        y_pos = pantry_depth - base_depth2
        Ah = -amplitude2  # Negative!
        x_off_h = offset2 * period2 / (2 * math.pi)
        
        # Quadrant: we want circle to right of left sinusoid (sv=+1) and below back sinusoid (sh=-1)
        quadrant = 'br'
        
    elif corner_type == 'right-back':
        # Vertical (right) sinusoid: x = (pantry_width - base_depth1) - amplitude1 * sin(2π*y/period1 - offset1)
        x_pos = pantry_width - base_depth1
        Av = -amplitude1  # Negative!
        y_off_v = offset1 * period1 / (2 * math.pi)
        
        # Horizontal (back) sinusoid: same as left-back
        y_pos = pantry_depth - base_depth2
        Ah = -amplitude2
        x_off_h = offset2 * period2 / (2 * math.pi)
        
        # Quadrant: we want circle to left of right sinusoid (sv=-1) and below back sinusoid (sh=-1)
        quadrant = 'bl'
    else:
        raise ValueError(f"Unsupported corner_type: {corner_type}")
    
    # Call the new solver
    result = tangent_circle_two_sinusoids_offset_intersection(
        Av=Av, x_pos=x_pos, y_off_v=y_off_v,
        Ah=Ah, y_pos=y_pos, x_off_h=x_off_h,
        r=radius,
        quadrant=quadrant
    )
    
    # Extract results and convert to numpy arrays
    center = np.array(result['center'])
    
    # Get the tangency points in the right order
    # For left-back: ordered is [p_vert, p_horz], so point1=p_vert (on left), point2=p_horz (on back)
    # For right-back: ordered is [p_horz, p_vert], so point1=p_horz (on back), point2=p_vert (on right)
    # But we want point1 on wall1 (left/right) and point2 on wall2 (back)
    
    if corner_type == 'left-back':
        # ordered = [p_vert, p_horz]
        point1 = np.array(result['points'][0])  # on left (vertical)
        point2 = np.array(result['points'][1])  # on back (horizontal)
    else:  # right-back
        # ordered = [p_horz, p_vert]
        point1 = np.array(result['points'][1])  # on right (vertical)
        point2 = np.array(result['points'][0])  # on back (horizontal)
    
    return center, point1, point2
