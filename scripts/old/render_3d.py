#!/usr/bin/env blender --background --python
"""
Generate 3D renderings of pantry shelves using Blender.

Usage:
    blender --background --python scripts/render_3d.py -- configs/pantry_0000.json
    blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --preset corner
    blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --camera-position 30 -40 50
    blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --all-presets
    blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --samples 256 --resolution 3840 2160

Note: Must be run with Blender's Python interpreter, not system Python.
"""

import sys
import argparse
from pathlib import Path

# Parse Blender-style arguments (after the --)
try:
    argv = sys.argv[sys.argv.index("--") + 1:]
except ValueError:
    argv = []

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import config
import blender_renderer

ShelfConfig = config.ShelfConfig
BlenderRenderer = blender_renderer.BlenderRenderer


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D renderings of pantry shelves'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory (default: output/renders/)'
    )
    parser.add_argument(
        '--preset', '-p',
        type=str,
        choices=['doorway', 'front', 'corner', 'inside', 'top', 'southeast', 'eye_level'],
        default='doorway',
        help='Camera preset view (default: doorway with fisheye)'
    )
    parser.add_argument(
        '--all-presets',
        action='store_true',
        help='Render all preset views'
    )
    parser.add_argument(
        '--fisheye',
        action='store_true',
        help='Use fisheye lens (panoramic camera)'
    )
    parser.add_argument(
        '--fov',
        type=float,
        default=170.0,
        help='Field of view in degrees for fisheye (default: 170)'
    )
    parser.add_argument(
        '--camera-position',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        help='Custom camera position'
    )
    parser.add_argument(
        '--camera-target',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        help='Camera look-at target (used with --camera-position)'
    )
    parser.add_argument(
        '--lens',
        type=float,
        default=35.0,
        help='Camera focal length in mm (default: 35)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[1920, 1080],
        help='Render resolution (default: 1920 1080)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=128,
        help='Render samples for quality (default: 128, higher=better/slower)'
    )
    parser.add_argument(
        '--no-walls',
        action='store_true',
        help='Render shelves only without walls'
    )

    args = parser.parse_args(argv)

    # Validate input file
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    try:
        cfg = ShelfConfig.from_file(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path('output/renders')

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInitializing Blender renderer...")
    print(f"  Configuration: {cfg.version}")
    print(f"  Shelves: {len(cfg.shelves)} sections")
    print(f"  Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"  Samples: {args.samples}")

    try:
        # Create renderer
        renderer = BlenderRenderer(cfg)

        # Clear scene
        print("\nClearing scene...")
        renderer.clear_scene()

        # Create shelves
        print("Creating shelf geometry...")
        shelf_objects = renderer.create_all_shelves()
        print(f"  Created {len(shelf_objects)} shelf objects")

        # Create walls if requested
        if not args.no_walls:
            print("Creating pantry walls...")
            wall_objects = renderer.create_pantry_room()
            print(f"  Created {len(wall_objects)} wall objects")

        # Apply materials
        print("Applying materials...")
        renderer.apply_materials()

        # Setup lighting
        print("Setting up lighting...")
        renderer.setup_lighting()

        # Determine which views to render
        if args.all_presets:
            presets_to_render = ['doorway', 'front', 'corner', 'inside', 'top', 'southeast', 'eye_level']
        else:
            presets_to_render = [args.preset] if not args.camera_position else []

        # Determine if fisheye should be used
        # Default to fisheye for 'doorway' preset
        use_fisheye = args.fisheye or (args.preset == 'doorway' and not args.camera_position)

        # Render preset views
        for preset in presets_to_render:
            print(f"\nRendering '{preset}' view...")
            location, rotation = renderer.create_preset_camera_view(preset)

            # Use fisheye for doorway preset by default
            preset_fisheye = use_fisheye if preset == args.preset else (preset == 'doorway')

            camera = renderer.setup_camera(
                location=location,
                rotation=rotation,
                lens=args.lens,
                fisheye=preset_fisheye,
                fov=args.fov if preset_fisheye else None
            )

            # Point at center of pantry
            center = (
                cfg.pantry['width'] / 2,
                cfg.pantry['depth'] / 2,
                cfg.pantry['height'] / 2
            )
            renderer.point_camera_at(camera, center)

            output_file = output_dir / f'pantry_{cfg.version}_{preset}.png'
            print(f"  Camera: {location}")
            if preset_fisheye:
                print(f"  Fisheye lens: {args.fov}° FOV")
            print(f"  Output: {output_file}")

            renderer.render(
                output_path=output_file,
                resolution=tuple(args.resolution),
                samples=args.samples
            )
            print(f"  ✓ Rendered successfully")

        # Render custom camera view
        if args.camera_position:
            print(f"\nRendering custom view...")
            location = tuple(args.camera_position)

            camera = renderer.setup_camera(
                location=location,
                rotation=(0, 0, 0),  # Will be overridden by point_camera_at
                lens=args.lens,
                fisheye=args.fisheye,
                fov=args.fov if args.fisheye else None
            )

            # Determine target
            if args.camera_target:
                target = tuple(args.camera_target)
            else:
                target = (
                    cfg.pantry['width'] / 2,
                    cfg.pantry['depth'] / 2,
                    cfg.pantry['height'] / 2
                )

            renderer.point_camera_at(camera, target)

            output_file = output_dir / f'pantry_{cfg.version}_custom.png'
            print(f"  Camera: {location}")
            print(f"  Target: {target}")
            print(f"  Output: {output_file}")

            renderer.render(
                output_path=output_file,
                resolution=tuple(args.resolution),
                samples=args.samples
            )
            print(f"  ✓ Rendered successfully")

        print("\n" + "="*60)
        print("SUCCESS! All renders complete.")
        print("="*60)

        if args.all_presets:
            print(f"\nGenerated {len(presets_to_render)} views:")
            for preset in presets_to_render:
                print(f"  - {output_dir}/pantry_{cfg.version}_{preset}.png")
        else:
            print(f"\nView renders: ls {output_dir}")

    except Exception as e:
        print(f"\nError during rendering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
