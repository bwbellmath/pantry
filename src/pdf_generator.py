"""
Generate 2D technical drawings of shelf footprints as PDF cutting templates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple
from pathlib import Path

import shelf_generator
import config

ShelfFootprint = shelf_generator.ShelfFootprint
ShelfGenerator = shelf_generator.ShelfGenerator
ShelfConfig = config.ShelfConfig


class PDFGenerator:
    """Generates technical 2D drawings for shelf cutting templates."""

    def __init__(self, config: ShelfConfig):
        """
        Initialize PDF generator.

        Args:
            config: ShelfConfig instance
        """
        self.config = config
        self.generator = ShelfGenerator(config)

    def create_shelf_drawing(self, footprint: ShelfFootprint,
                           ax: plt.Axes, show_dimensions: bool = True) -> None:
        """
        Draw a single shelf footprint with dimensions.

        Args:
            footprint: ShelfFootprint to draw
            ax: Matplotlib axes to draw on
            show_dimensions: Whether to show dimension annotations
        """
        # Draw the shelf outline
        outline = footprint.outline_points
        ax.plot(outline[:, 0], outline[:, 1], 'b-', linewidth=2, label='Shelf edge')
        ax.plot([outline[-1, 0], outline[0, 0]],
               [outline[-1, 1], outline[0, 1]], 'b-', linewidth=2)

        # Fill the shelf
        ax.fill(outline[:, 0], outline[:, 1], alpha=0.3, color='lightblue')

        # Add circle centers for debugging (if East or West wall)
        if footprint.wall in ['E', 'W']:
            self._draw_circle_centers(ax, footprint)

        # Get bounds
        min_x, max_x, min_y, max_y = footprint.get_bounds()

        # Add dimensions if requested
        if show_dimensions:
            self._add_dimensions(ax, footprint, min_x, max_x, min_y, max_y)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Labels
        wall_names = {'E': 'East', 'S': 'South', 'W': 'West'}
        ax.set_xlabel('X (inches)', fontsize=10)
        ax.set_ylabel('Y (inches)', fontsize=10)
        ax.set_title(
            f'Level {footprint.level} - {wall_names[footprint.wall]} Wall\n'
            f'Height: {footprint.shelf_data["height"]:.1f}"',
            fontsize=12, fontweight='bold'
        )

        # Add parameter text box
        self._add_parameter_box(ax, footprint)

    def _add_dimensions(self, ax: plt.Axes, footprint: ShelfFootprint,
                       min_x: float, max_x: float,
                       min_y: float, max_y: float) -> None:
        """
        Add dimension annotations to the drawing.

        Args:
            ax: Matplotlib axes
            footprint: ShelfFootprint being drawn
            min_x, max_x, min_y, max_y: Bounding box coordinates
        """
        # Add overall dimensions
        width = max_x - min_x
        height = max_y - min_y

        # Width dimension line (below the shelf)
        y_offset = min_y - 3
        ax.plot([min_x, max_x], [y_offset, y_offset], 'k-', linewidth=1)
        ax.plot([min_x, min_x], [y_offset - 0.5, y_offset + 0.5], 'k-', linewidth=1)
        ax.plot([max_x, max_x], [y_offset - 0.5, y_offset + 0.5], 'k-', linewidth=1)
        ax.text((min_x + max_x) / 2, y_offset - 1.5, f'{width:.2f}"',
               ha='center', va='top', fontsize=9, fontweight='bold')

        # Height dimension line (to the right of the shelf)
        x_offset = max_x + 3
        ax.plot([x_offset, x_offset], [min_y, max_y], 'k-', linewidth=1)
        ax.plot([x_offset - 0.5, x_offset + 0.5], [min_y, min_y], 'k-', linewidth=1)
        ax.plot([x_offset - 0.5, x_offset + 0.5], [max_y, max_y], 'k-', linewidth=1)
        ax.text(x_offset + 1.5, (min_y + max_y) / 2, f'{height:.2f}"',
               ha='left', va='center', fontsize=9, fontweight='bold', rotation=90)

    def _draw_circle_centers(self, ax: plt.Axes, footprint: ShelfFootprint) -> None:
        """
        Draw circle centers for corner radiusing (debugging).

        Args:
            ax: Matplotlib axes
            footprint: ShelfFootprint being drawn
        """
        from geometry import solve_door_corner, solve_interior_corner, sinusoid_depth

        wall = footprint.wall
        level = footprint.level

        # Get base depth for this wall
        base_depth = self.config.get_base_depth(wall)

        # Get depths at corners
        extent_start = footprint.shelf_data['extent_start']  # North (y=0)
        extent_end = footprint.shelf_data['extent_end']      # South (y=pantry_depth)

        depth_at_north = sinusoid_depth(
            extent_start,
            base_depth,
            self.config.design_params['sinusoid_amplitude'],
            self.config.design_params['sinusoid_period'],
            footprint.shelf_data['sinusoid_offset']
        )

        depth_at_south = sinusoid_depth(
            extent_end,
            base_depth,
            self.config.design_params['sinusoid_amplitude'],
            self.config.design_params['sinusoid_period'],
            footprint.shelf_data['sinusoid_offset']
        )

        # Door corner (North) - 3" radius
        door_arc_center, _ = solve_door_corner(
            depth_at_north,
            self.config.design_params['door_corner_radius'],
            wall
        )

        # Convert to pantry coordinates
        if wall == 'E':
            door_center_pantry = door_arc_center  # Already in pantry coords for East
        else:  # 'W'
            door_center_pantry = np.array([self.config.pantry['width'] + door_arc_center[0], door_arc_center[1]])

        # Interior corner (South) - 6" radius
        south_shelf = self.config.get_shelf(level, 'S')
        if south_shelf:
            south_base_depth = self.config.get_base_depth('S')
            if wall == 'E':
                south_position = 0
            else:  # 'W'
                south_position = self.config.pantry['width']

            south_depth = sinusoid_depth(
                south_position,
                south_base_depth,
                self.config.design_params['sinusoid_amplitude'],
                self.config.design_params['sinusoid_period'],
                south_shelf['sinusoid_offset']
            )

            interior_arc_center, _, _ = solve_interior_corner(
                depth_at_south,
                south_depth,
                self.config.design_params['interior_corner_radius'],
                wall
            )

            # Convert to pantry coordinates
            if wall == 'E':
                corner_pantry = np.array([depth_at_south, self.config.pantry['depth']])
            else:  # 'W'
                corner_pantry = np.array([self.config.pantry['width'] - depth_at_south, self.config.pantry['depth']])

            interior_center_pantry = corner_pantry + interior_arc_center

            # Draw interior circle center
            ax.plot(interior_center_pantry[0], interior_center_pantry[1], 'ro',
                   markersize=8, label='Interior corner center (6")', zorder=10)
            ax.annotate(f'6" radius\n({interior_center_pantry[0]:.1f}, {interior_center_pantry[1]:.1f})',
                       xy=interior_center_pantry, xytext=(10, 10),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Draw door circle center
        ax.plot(door_center_pantry[0], door_center_pantry[1], 'go',
               markersize=8, label='Door corner center (3")', zorder=10)
        ax.annotate(f'3" radius\n({door_center_pantry[0]:.1f}, {door_center_pantry[1]:.1f})',
                   xy=door_center_pantry, xytext=(10, -20),
                   textcoords='offset points', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def _add_parameter_box(self, ax: plt.Axes, footprint: ShelfFootprint) -> None:
        """
        Add a text box with shelf parameters.

        Args:
            ax: Matplotlib axes
            footprint: ShelfFootprint being drawn
        """
        params = self.config.design_params
        wall = footprint.wall
        base_depth = self.config.get_base_depth(wall)
        text = (
            f'Parameters:\n'
            f'  Base depth ({wall}): {base_depth:.1f}"\n'
            f'  Amplitude: {params["sinusoid_amplitude"]:.1f}"\n'
            f'  Period: {params["sinusoid_period"]:.1f}"\n'
            f'  Offset: {footprint.shelf_data["sinusoid_offset"]:.4f} rad\n'
            f'  Thickness: {params["shelf_thickness"]:.1f}"\n'
            f'  Interior radius: {params.get("interior_corner_radius", 0):.1f}"\n'
            f'  Door radius: {params.get("door_corner_radius", 0):.1f}"'
        )

        # Position text box in upper right
        ax.text(0.98, 0.98, text,
               transform=ax.transAxes,
               fontsize=8,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def create_level_overview(self, level: int, ax: plt.Axes) -> None:
        """
        Create an overview drawing showing all shelves at one level.

        Args:
            level: Shelf level number
            ax: Matplotlib axes to draw on
        """
        footprints = self.generator.get_footprints_by_level(level)

        if not footprints:
            ax.text(0.5, 0.5, 'No shelves at this level',
                   transform=ax.transAxes, ha='center', va='center')
            return

        # Draw pantry outline
        pantry_width = self.config.pantry['width']
        pantry_depth = self.config.pantry['depth']

        pantry_rect = patches.Rectangle(
            (0, 0), pantry_width, pantry_depth,
            linewidth=2, edgecolor='black', facecolor='none',
            linestyle='--', label='Pantry walls'
        )
        ax.add_patch(pantry_rect)

        # Draw door (North wall)
        # Support both old and new door clearance format
        if 'door_clearance_east' in self.config.pantry:
            door_clearance_east = self.config.pantry['door_clearance_east']
            door_clearance_west = self.config.pantry['door_clearance_west']
        else:
            door_clearance_east = self.config.pantry.get('door_clearance_sides', 4.5)
            door_clearance_west = self.config.pantry.get('door_clearance_sides', 4.5)

        door_width = pantry_width - door_clearance_east - door_clearance_west
        door_rect = patches.Rectangle(
            (door_clearance_east, -2), door_width, 2,
            linewidth=1, edgecolor='brown', facecolor='lightgray',
            label='Door'
        )
        ax.add_patch(door_rect)

        # Draw each shelf
        colors = ['red', 'green', 'blue']
        for i, footprint in enumerate(footprints):
            outline = footprint.outline_points
            color = colors[i % len(colors)]
            ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.5,
                   label=f'{footprint.wall} wall')
            ax.plot([outline[-1, 0], outline[0, 0]],
                   [outline[-1, 1], outline[0, 1]], color=color, linewidth=1.5)
            ax.fill(outline[:, 0], outline[:, 1], alpha=0.2, color=color)

        # Set limits with some margin
        ax.set_xlim(-5, pantry_width + 5)
        ax.set_ylim(-5, pantry_depth + 5)
        ax.set_aspect('equal', adjustable='box')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Labels and legend
        ax.set_xlabel('X (inches)', fontsize=10)
        ax.set_ylabel('Y (inches)', fontsize=10)
        ax.set_title(f'Level {level} Overview - Height: {footprints[0].shelf_data["height"]:.1f}"',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)

    def generate_pdf(self, output_path: Path, include_overview: bool = True) -> None:
        """
        Generate complete PDF with all shelf drawings.

        Args:
            output_path: Path to save PDF file
            include_overview: Whether to include overview pages for each level
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate all footprints
        footprints = self.generator.generate_all_footprints()

        if not footprints:
            print("No shelves to draw!")
            return

        # Get unique levels
        levels = sorted(set(fp.level for fp in footprints))

        with PdfPages(output_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.axis('off')

            title_text = (
                f'Pantry Shelf Cutting Templates\n\n'
                f'Configuration: {self.config.version}\n'
                f'Pantry: {self.config.pantry["width"]}" × '
                f'{self.config.pantry["depth"]}" × '
                f'{self.config.pantry["height"]}"\n\n'
                f'Levels: {len(levels)}\n'
                f'Total shelf sections: {len(footprints)}\n\n'
                f'Design Parameters:\n'
                f'  Base depths: E:{self.config.design_params.get("shelf_base_depth_east", "N/A")}" '
                f'S:{self.config.design_params.get("shelf_base_depth_south", "N/A")}" '
                f'W:{self.config.design_params.get("shelf_base_depth_west", "N/A")}"\n'
                f'  Sinusoid amplitude: {self.config.design_params["sinusoid_amplitude"]}"\n'
                f'  Sinusoid period: {self.config.design_params["sinusoid_period"]}"\n'
                f'  Shelf thickness: {self.config.design_params["shelf_thickness"]}"\n'
                f'  Corner radii: Interior:{self.config.design_params.get("interior_corner_radius", 0)}" '
                f'Door:{self.config.design_params.get("door_corner_radius", 0)}"'
            )

            ax.text(0.5, 0.6, title_text, transform=ax.transAxes,
                   fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Generate pages for each level
            for level in levels:
                level_footprints = [fp for fp in footprints if fp.level == level]

                # Overview page for this level
                if include_overview:
                    fig = plt.figure(figsize=(11, 8.5))
                    ax = fig.add_subplot(111)
                    self.create_level_overview(level, ax)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

                # Individual shelf pages
                for footprint in level_footprints:
                    fig = plt.figure(figsize=(11, 8.5))
                    ax = fig.add_subplot(111)
                    self.create_shelf_drawing(footprint, ax, show_dimensions=True)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

        print(f"PDF generated: {output_path}")
        print(f"  Pages: {len(levels) * (4 if include_overview else 3) + 1}")
        print(f"  Levels: {len(levels)}")
        print(f"  Shelf sections: {len(footprints)}")
