#!/usr/bin/env python3
"""
Generate pantry shelf configuration files with random sinusoid offsets.

Usage:
    python scripts/generate_config.py --output configs/pantry_0000.json
    python scripts/generate_config.py --base configs/pantry_0000.json --output configs/pantry_0001.json
    python scripts/generate_config.py --auto-version
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config

ShelfConfig = config.ShelfConfig


def main():
    parser = argparse.ArgumentParser(
        description='Generate pantry shelf configuration with random offsets'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--base', '-b',
        type=Path,
        help='Base configuration to copy parameters from'
    )
    parser.add_argument(
        '--num-levels', '-n',
        type=int,
        default=4,
        help='Number of shelf levels (default: 4)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--auto-version', '-a',
        action='store_true',
        help='Automatically determine next version number'
    )

    args = parser.parse_args()

    # Load or create config
    if args.base:
        print(f"Loading base configuration from {args.base}")
        config = ShelfConfig.from_file(args.base)
    else:
        print("Creating new default configuration")
        config = ShelfConfig()

    # Generate shelf entries with random offsets
    print(f"Generating {args.num_levels} shelf levels with random offsets...")
    config.generate_shelf_entries(
        num_levels=args.num_levels,
        randomize=True,
        seed=args.seed
    )

    # Determine output path
    if args.auto_version:
        config_dir = Path('configs')
        version = config.get_next_version_number(config_dir)
        output_path = config_dir / f'pantry_{version}.json'
    elif args.output:
        output_path = args.output
    else:
        print("Error: Must specify --output or --auto-version")
        sys.exit(1)

    # Update version in config
    version_str = output_path.stem.split('_')[1] if '_' in output_path.stem else "0000"
    config.version = version_str

    # Save configuration
    print(f"Saving configuration to {output_path}")
    config.to_file(output_path)

    # Print summary
    print("\nConfiguration Summary:")
    print(f"  Version: {config.version}")
    print(f"  Pantry: {config.pantry['width']}\" × {config.pantry['depth']}\" × {config.pantry['height']}\"")
    print(f"  Door clearances: East {config.pantry.get('door_clearance_east', 'N/A')}\" / "
          f"West {config.pantry.get('door_clearance_west', 'N/A')}\"")
    print(f"  Shelves: {len(config.shelves)} sections across {args.num_levels} levels")
    print(f"  Sinusoid: {config.design_params['sinusoid_amplitude']}\" amplitude, "
          f"{config.design_params['sinusoid_period']}\" period")
    print(f"  Base depths: East {config.design_params.get('shelf_base_depth_east', 'N/A')}\", "
          f"South {config.design_params.get('shelf_base_depth_south', 'N/A')}\", "
          f"West {config.design_params.get('shelf_base_depth_west', 'N/A')}\"")
    print(f"  Corner radii: Interior {config.design_params.get('interior_corner_radius', 'N/A')}\", "
          f"Door {config.design_params.get('door_corner_radius', 'N/A')}\"")

    print("\nShelf offsets (radians):")
    for shelf in config.shelves:
        print(f"  Level {shelf['level']} {shelf['wall']}: {shelf['sinusoid_offset']:.4f}")


if __name__ == '__main__':
    main()
