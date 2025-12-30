"""
Configuration management for pantry shelf designs.
Handles loading, saving, and validating JSON configuration files.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


class ShelfConfig:
    """Manages pantry shelf configuration data."""

    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_data: Configuration dictionary. If None, creates default config.
        """
        if config_data is None:
            self.data = self._create_default_config()
        else:
            self.data = config_data
            self.validate()

    @staticmethod
    def _create_default_config() -> Dict[str, Any]:
        """Create a default configuration with placeholder values."""
        return {
            "config_version": "0000",
            "pantry": {
                "width": 48.0,
                "depth": 49.0,
                "height": 105.0,
                "door_clearance_east": 6.0,  # Left side when looking in
                "door_clearance_west": 4.0   # Right side when looking in
            },
            "design_params": {
                "sinusoid_period": 24.0,
                "sinusoid_amplitude": 1.0,  # Halved from 2.0
                "shelf_base_depth_east": 7.0,
                "shelf_base_depth_south": 19.0,
                "shelf_base_depth_west": 4.0,
                "shelf_thickness": 1.0,
                "interior_corner_radius": 3.0,  # South wall corners (add material)
                "door_corner_radius": 3.0       # North wall corners (remove material)
            },
            "shelves": []
        }

    @classmethod
    def from_file(cls, filepath: Path) -> 'ShelfConfig':
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file

        Returns:
            ShelfConfig instance
        """
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        return cls(config_data)

    def to_file(self, filepath: Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            filepath: Path to save JSON configuration
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def validate(self) -> bool:
        """
        Validate configuration data structure.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['config_version', 'pantry', 'design_params', 'shelves']
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"Missing required key: {key}")

        required_pantry_keys = ['width', 'depth', 'height']
        for key in required_pantry_keys:
            if key not in self.data['pantry']:
                raise ValueError(f"Missing required pantry key: {key}")

        # Support both old and new door clearance format
        if 'door_clearance_sides' in self.data['pantry']:
            # Convert old format to new
            clearance = self.data['pantry']['door_clearance_sides']
            self.data['pantry']['door_clearance_east'] = clearance
            self.data['pantry']['door_clearance_west'] = clearance
        elif 'door_clearance_east' not in self.data['pantry'] or 'door_clearance_west' not in self.data['pantry']:
            raise ValueError("Missing door clearance keys")

        required_design_keys = ['sinusoid_period', 'sinusoid_amplitude', 'shelf_thickness']
        for key in required_design_keys:
            if key not in self.data['design_params']:
                raise ValueError(f"Missing required design_params key: {key}")

        # Support both old and new depth format
        if 'shelf_base_depth' in self.data['design_params']:
            # Convert old format to new
            depth = self.data['design_params']['shelf_base_depth']
            self.data['design_params']['shelf_base_depth_east'] = depth
            self.data['design_params']['shelf_base_depth_south'] = depth
            self.data['design_params']['shelf_base_depth_west'] = depth

        return True

    def generate_shelf_entries(self, num_levels: int = 4,
                              randomize: bool = True,
                              seed: Optional[int] = None) -> None:
        """
        Generate shelf entries for all levels and walls.

        Args:
            num_levels: Number of shelf levels to create
            randomize: Whether to use random sinusoid offsets
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        shelves = []
        pantry_height = self.data['pantry']['height']
        pantry_depth = self.data['pantry']['depth']
        pantry_width = self.data['pantry']['width']

        # Calculate even spacing for shelf heights
        # Leave some space at bottom and top
        spacing = pantry_height / (num_levels + 1)

        for level in range(num_levels):
            height = spacing * (level + 1)

            # Each level has 3 shelves: East, South, West
            # East wall (x=0, runs along y-axis)
            east_offset = np.random.uniform(0, 2 * np.pi) if randomize else 0.0
            shelves.append({
                "level": level,
                "height": height,
                "wall": "E",
                "extent_start": 0.0,
                "extent_end": pantry_depth,
                "sinusoid_offset": float(east_offset),
                "corner_points_solved": []
            })

            # South wall (y=depth, runs along x-axis)
            south_offset = np.random.uniform(0, 2 * np.pi) if randomize else 0.0
            shelves.append({
                "level": level,
                "height": height,
                "wall": "S",
                "extent_start": 0.0,
                "extent_end": pantry_width,
                "sinusoid_offset": float(south_offset),
                "corner_points_solved": []
            })

            # West wall (x=width, runs along y-axis)
            west_offset = np.random.uniform(0, 2 * np.pi) if randomize else 0.0
            shelves.append({
                "level": level,
                "height": height,
                "wall": "W",
                "extent_start": 0.0,
                "extent_end": pantry_depth,
                "sinusoid_offset": float(west_offset),
                "corner_points_solved": []
            })

        self.data['shelves'] = shelves

    def get_shelves_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        Get all shelf entries for a specific level.

        Args:
            level: Shelf level number

        Returns:
            List of shelf dictionaries for that level
        """
        return [s for s in self.data['shelves'] if s['level'] == level]

    def get_shelf(self, level: int, wall: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific shelf entry.

        Args:
            level: Shelf level number
            wall: Wall identifier ('E', 'S', or 'W')

        Returns:
            Shelf dictionary or None if not found
        """
        for shelf in self.data['shelves']:
            if shelf['level'] == level and shelf['wall'] == wall:
                return shelf
        return None

    def get_next_version_number(self, config_dir: Path) -> str:
        """
        Get the next sequential version number based on existing configs.

        Args:
            config_dir: Directory containing config files

        Returns:
            Next version number as 4-digit string (e.g., "0001")
        """
        if not config_dir.exists():
            return "0000"

        existing_configs = list(config_dir.glob("pantry_*.json"))
        if not existing_configs:
            return "0000"

        max_version = -1
        for config_path in existing_configs:
            try:
                version_str = config_path.stem.split('_')[1]
                version_num = int(version_str)
                max_version = max(max_version, version_num)
            except (IndexError, ValueError):
                continue

        return f"{max_version + 1:04d}"

    @property
    def pantry(self) -> Dict[str, float]:
        """Get pantry dimensions."""
        return self.data['pantry']

    @property
    def design_params(self) -> Dict[str, float]:
        """Get design parameters."""
        return self.data['design_params']

    @property
    def shelves(self) -> List[Dict[str, Any]]:
        """Get all shelf entries."""
        return self.data['shelves']

    @property
    def version(self) -> str:
        """Get configuration version."""
        return self.data['config_version']

    @version.setter
    def version(self, value: str) -> None:
        """Set configuration version."""
        self.data['config_version'] = value

    def get_base_depth(self, wall: str) -> float:
        """
        Get base depth for a specific wall.

        Args:
            wall: Wall identifier ('E', 'S', or 'W')

        Returns:
            Base depth in inches
        """
        wall_map = {
            'E': 'shelf_base_depth_east',
            'S': 'shelf_base_depth_south',
            'W': 'shelf_base_depth_west'
        }
        if wall not in wall_map:
            raise ValueError(f"Unknown wall: {wall}")
        return self.design_params[wall_map[wall]]
