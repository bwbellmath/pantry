"""
Generate complete shelf footprints with corner radiusing.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import geometry
import config

create_shelf_outline = geometry.create_shelf_outline
sinusoid_depth = geometry.sinusoid_depth
wall_to_pantry_coords = geometry.wall_to_pantry_coords
solve_interior_corner = geometry.solve_interior_corner
solve_door_corner = geometry.solve_door_corner
generate_circle_arc = geometry.generate_circle_arc
CornerSolver = geometry.CornerSolver
ShelfConfig = config.ShelfConfig


class ShelfFootprint:
    """Represents a complete shelf footprint with corners."""

    def __init__(self, outline_points: np.ndarray, wall: str, level: int,
                 shelf_data: Dict[str, Any]):
        """
        Initialize shelf footprint.

        Args:
            outline_points: Array of [x, y] points forming the shelf outline
            wall: Wall identifier ('E', 'S', 'W')
            level: Shelf level number
            shelf_data: Dictionary containing shelf parameters
        """
        self.outline_points = outline_points
        self.wall = wall
        self.level = level
        self.shelf_data = shelf_data

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of shelf.

        Returns:
            (min_x, max_x, min_y, max_y)
        """
        min_x = np.min(self.outline_points[:, 0])
        max_x = np.max(self.outline_points[:, 0])
        min_y = np.min(self.outline_points[:, 1])
        max_y = np.max(self.outline_points[:, 1])
        return (min_x, max_x, min_y, max_y)


class ShelfGenerator:
    """Generates complete shelf footprints from configuration."""

    def __init__(self, config: ShelfConfig):
        """
        Initialize shelf generator.

        Args:
            config: ShelfConfig instance
        """
        self.config = config
        self.pantry = config.pantry
        self.design = config.design_params
        self.corner_solver = CornerSolver(
            self.design['interior_corner_radius'],
            self.design['door_corner_radius']
        )

    def generate_shelf_footprint(self, level: int, wall: str,
                                num_points: int = 200) -> ShelfFootprint:
        """
        Generate footprint for a single shelf section.

        Args:
            level: Shelf level number
            wall: Wall identifier ('E', 'S', 'W')
            num_points: Number of points for sinusoid

        Returns:
            ShelfFootprint instance
        """
        shelf_data = self.config.get_shelf(level, wall)
        if shelf_data is None:
            raise ValueError(f"No shelf found for level {level}, wall {wall}")

        # Get wall-specific base depth
        base_depth = self.config.get_base_depth(wall)

        # Generate basic outline without corners
        outline = create_shelf_outline(
            wall_type=wall,
            extent_start=shelf_data['extent_start'],
            extent_end=shelf_data['extent_end'],
            base_depth=base_depth,
            amplitude=self.design['sinusoid_amplitude'],
            period=self.design['sinusoid_period'],
            offset=shelf_data['sinusoid_offset'],
            pantry_width=self.pantry['width'],
            pantry_depth=self.pantry['depth'],
            num_points=num_points
        )

        # Apply corner modifications
        outline = self._apply_corners(outline, level, wall, shelf_data)

        return ShelfFootprint(outline, wall, level, shelf_data)

    def _apply_corners(self, outline: np.ndarray, level: int, wall: str,
                      shelf_data: Dict[str, Any]) -> np.ndarray:
        """
        Apply corner radiusing to shelf outline.

        For East/West walls:
        - Door (North) corners: REMOVE material with 3" radius
        - South corners: ADD material with 6" radius

        Args:
            outline: Original outline points
            level: Shelf level
            wall: Wall identifier
            shelf_data: Shelf configuration data

        Returns:
            Modified outline with corners
        """
        if wall == 'S':
            # South wall has no corner modifications
            return outline

        if wall not in ['E', 'W']:
            return outline

        # Get base depth for this wall
        base_depth = self.config.get_base_depth(wall)

        # Get depths at corners
        extent_start = shelf_data['extent_start']  # North (y=0)
        extent_end = shelf_data['extent_end']      # South (y=pantry_depth)

        depth_at_north = sinusoid_depth(
            extent_start,
            base_depth,
            self.design['sinusoid_amplitude'],
            self.design['sinusoid_period'],
            shelf_data['sinusoid_offset']
        )

        depth_at_south = sinusoid_depth(
            extent_end,
            base_depth,
            self.design['sinusoid_amplitude'],
            self.design['sinusoid_period'],
            shelf_data['sinusoid_offset']
        )

        # Apply corner modifications
        outline_with_corners = self._modify_outline_with_corners(
            outline, wall, depth_at_north, depth_at_south, level
        )

        return outline_with_corners

    def _modify_outline_with_corners(self, outline: np.ndarray, wall: str,
                                    depth_north: float, depth_south: float,
                                    level: int) -> np.ndarray:
        """
        Modify outline to add/remove material at corners.

        Args:
            outline: Original outline points [outer_edge..., inner_edge...]
            wall: 'E' or 'W'
            depth_north: Depth at north (door) corner
            depth_south: Depth at south corner
            level: Shelf level

        Returns:
            Modified outline with corner arcs
        """
        # Outline structure: [sinusoid_points (start to end), inner_wall_end, inner_wall_start]
        # For East wall in pantry coords:
        #   - sinusoid_points go from (depth, 0) to (depth, pantry_depth)
        #   - inner points are (0, pantry_depth) and (0, 0)

        # Split outline into outer and inner edges
        # The last 2 points are the inner edge (wall)
        outer_edge = outline[:-2]  # Sinusoidal edge
        inner_edge_end = outline[-2]
        inner_edge_start = outline[-1]

        # Door corner (North): REMOVE material
        door_arc_center_local, _ = solve_door_corner(
            depth_north,
            self.design['door_corner_radius'],
            wall
        )

        # Convert center to pantry coordinates
        if wall == 'E':
            door_center_pantry = door_arc_center_local
        else:  # 'W'
            door_center_pantry = np.array([self.pantry['width'] + door_arc_center_local[0], door_arc_center_local[1]])

        # Define the two points that the arc should connect
        # Start: on the wall at y=0
        # End: on the shelf edge at y=0
        if wall == 'E':
            # East wall
            door_point1 = np.array([0, 0])  # Wall at door
            door_point2 = np.array([depth_north, 0])  # Shelf edge at door
        else:  # 'W'
            # West wall
            door_point1 = np.array([self.pantry['width'], 0])  # Wall at door
            door_point2 = np.array([self.pantry['width'] - depth_north, 0])  # Shelf edge at door

        # Generate arc using the shorter path
        door_arc_pantry = generate_circle_arc(
            door_center_pantry,
            door_point1,
            door_point2,
            num_points=20,
            interior_arc=True  # Always use the shorter arc
        )

        # Interior corner (South): ADD material
        # Get south shelf depth at this corner
        south_shelf = self.config.get_shelf(level, 'S')
        if south_shelf:
            south_base_depth = self.config.get_base_depth('S')
            # Position along south shelf depends on which wall we're on
            if wall == 'E':
                south_position = 0  # West edge of south shelf
            else:  # 'W'
                south_position = self.pantry['width']  # East edge of south shelf

            south_depth = sinusoid_depth(
                south_position,
                south_base_depth,
                self.design['sinusoid_amplitude'],
                self.design['sinusoid_period'],
                south_shelf['sinusoid_offset']
            )

            # Solve for interior corner arc
            interior_arc_center, interior_arc_start, interior_arc_end = solve_interior_corner(
                depth_south,
                south_depth,
                self.design['interior_corner_radius'],
                wall
            )

            # Convert to pantry coordinates
            if wall == 'E':
                # Corner is at (depth_south, pantry_depth) in pantry coords
                corner_pantry = np.array([depth_south, self.pantry['depth']])
            else:  # 'W'
                # Corner is at (pantry_width - depth_south, pantry_depth)
                corner_pantry = np.array([self.pantry['width'] - depth_south, self.pantry['depth']])

            # Calculate the two tangent points on the circle
            # The circle is tangent to both the east/west shelf edge and the south shelf edge
            # We need to find where these tangencies occur

            # For now, use the start and end points of the arc based on the geometry
            # The arc connects the shelf edge to the adjacent shelf edge
            # Start point: on the east/west shelf at the south corner
            # End point: on the south shelf at the east/west corner

            # Start point (on E/W shelf edge at south)
            if wall == 'E':
                # Point on east shelf edge at y=pantry_depth
                point1 = np.array([depth_south, self.pantry['depth']])
            else:  # 'W'
                # Point on west shelf edge at y=pantry_depth
                point1 = np.array([self.pantry['width'] - depth_south, self.pantry['depth']])

            # End point (on south shelf edge at east/west corner)
            if wall == 'E':
                # Point on south shelf edge at x=0
                point2 = np.array([south_depth, self.pantry['depth'] - south_depth])
            else:  # 'W'
                # Point on south shelf edge at x=pantry_width
                point2 = np.array([self.pantry['width'] - south_depth, self.pantry['depth'] - south_depth])

            # Generate the shorter arc between point1 and point2
            interior_center_pantry = corner_pantry + interior_arc_center
            interior_arc_pantry = generate_circle_arc(
                interior_center_pantry,
                point1,
                point2,
                num_points=20,
                interior_arc=True  # Always use the shorter arc
            )
        else:
            interior_arc_pantry = np.array([])

        # Reconstruct outline:
        # 1. Door arc (replaces start of outer edge)
        # 2. Outer edge (trimmed at both ends)
        # 3. Interior arc (extends outer edge at south)
        # 4. Inner edge points

        # Trim outer edge to avoid overlap with arcs
        # Remove first few points near north corner and last few near south corner
        trim_count = 5
        outer_edge_trimmed = outer_edge[trim_count:-trim_count] if len(outer_edge) > 2*trim_count else outer_edge

        # Assemble new outline
        new_outline = []

        # Start with door arc
        for point in door_arc_pantry:
            new_outline.append(point)

        # Add trimmed outer edge
        for point in outer_edge_trimmed:
            new_outline.append(point)

        # Add interior arc if exists
        if len(interior_arc_pantry) > 0:
            for point in interior_arc_pantry:
                new_outline.append(point)

        # Add inner edge (close the polygon)
        new_outline.append(inner_edge_end)
        new_outline.append(inner_edge_start)

        return np.array(new_outline)

    def generate_all_footprints(self, num_points: int = 200) -> List[ShelfFootprint]:
        """
        Generate footprints for all shelves in configuration.

        Args:
            num_points: Number of points for sinusoid curves

        Returns:
            List of ShelfFootprint instances
        """
        footprints = []

        # Get unique levels
        levels = sorted(set(shelf['level'] for shelf in self.config.shelves))

        for level in levels:
            for wall in ['E', 'S', 'W']:
                try:
                    footprint = self.generate_shelf_footprint(level, wall, num_points)
                    footprints.append(footprint)
                except ValueError:
                    # Shelf doesn't exist for this level/wall combination
                    continue

        return footprints

    def get_footprints_by_level(self, level: int,
                               num_points: int = 200) -> List[ShelfFootprint]:
        """
        Generate footprints for a specific level.

        Args:
            level: Shelf level number
            num_points: Number of points for sinusoid curves

        Returns:
            List of ShelfFootprint instances for that level
        """
        footprints = []
        for wall in ['E', 'S', 'W']:
            try:
                footprint = self.generate_shelf_footprint(level, wall, num_points)
                footprints.append(footprint)
            except ValueError:
                continue
        return footprints
