#!/usr/bin/env blender --background --python
"""
Generate 3D renderings of complete pantry with all shelves.

Usage:
    blender --background --python scripts/render_complete_pantry.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Add user site-packages to path for scipy
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

try:
    import bpy
except ImportError:
    print("Error: This script must be run with Blender's Python interpreter")
    sys.exit(1)

import config
import blender_renderer

ShelfConfig = config.ShelfConfig
BlenderRenderer = blender_renderer.BlenderRenderer


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


def generate_intermediate_shelf_polygon(depth, length, side, amplitude, period, offset, corner_radius=3.0):
    """Generate intermediate shelf polygon."""
    y_points = np.linspace(0, length, 100)
    center_y = length - corner_radius

    def objective(y):
        if side == 'E':
            x_sinusoid = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
            center_x_estimate = x_sinusoid - corner_radius
        else:
            x_sinusoid = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
            center_x_estimate = x_sinusoid + corner_radius

        dist = np.sqrt((x_sinusoid - center_x_estimate)**2 + (y - center_y)**2)
        return abs(dist - corner_radius)

    result = simple_minimize_scalar(objective, bounds=(length - 2*corner_radius, length))
    tangent_y = result.x

    if side == 'E':
        x_sinusoid_tangent = depth + amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid_tangent - corner_radius
    else:
        x_sinusoid_tangent = 48 - depth - amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid_tangent + corner_radius

    angle_horizontal = np.pi/2
    angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

    polygon = []

    if side == 'E':
        polygon.append([0, 0])
        polygon.append([0, length])
        polygon.append([center_x, length])

        arc_angles = np.linspace(angle_horizontal, angle_sinusoid, 20)
        for angle in arc_angles:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        for y in y_points[::-1]:
            if y <= tangent_y:
                x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        x_north = depth + amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])
    else:  # 'W'
        polygon.append([48, 0])
        x_north = 48 - depth - amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])

        for y in y_points:
            if y <= tangent_y:
                x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        arc_angles = np.linspace(angle_sinusoid, angle_horizontal, 20)
        for angle in arc_angles:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        polygon.append([48, length])

    return np.array(polygon)


def main():
    print("="*60)
    print("Rendering Complete Pantry")
    print("="*60)

    # Load configuration
    config_path = Path('configs/pantry_0002.json')
    print(f"\nLoading config: {config_path}")
    cfg = ShelfConfig.from_file(config_path)

    # Create renderer
    print("\nInitializing Blender renderer...")
    renderer = BlenderRenderer(cfg)

    # Clear scene
    print("Clearing scene...")
    renderer.clear_scene()

    # Create main shelves (from config)
    print("Creating main shelves...")
    main_shelves = renderer.create_all_shelves()
    print(f"  Created {len(main_shelves)} main shelf objects")

    # Create intermediate shelves
    print("\nCreating intermediate shelves...")
    amplitude = cfg.design_params['sinusoid_amplitude']
    period = cfg.design_params['sinusoid_period']
    left_depth = cfg.design_params['shelf_base_depth_east']
    right_depth = cfg.design_params['shelf_base_depth_west']
    thickness = cfg.design_params['shelf_thickness']
    corner_radius = cfg.design_params['interior_corner_radius']

    left_intermediate_heights = [9, 29, 49, 69]
    right_intermediate_heights = [5, 13, 26, 33, 46, 53, 66, 73, 86]

    np.random.seed(42)

    # Left intermediate shelves
    for height in left_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf_polygon(
            left_depth, 29.0, 'E', amplitude, period, offset, corner_radius
        )

        # Create mesh using renderer's method
        import bmesh
        mesh = bpy.data.meshes.new(f"shelf_L{height}")
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        bm = bmesh.new()
        bottom_verts = [bm.verts.new((x, y, height)) for x, y in polygon]
        top_verts = [bm.verts.new((x, y, height + thickness)) for x, y in polygon]
        bm.verts.ensure_lookup_table()

        bm.faces.new(bottom_verts)
        bm.faces.new(reversed(top_verts))

        for i in range(len(polygon)):
            next_i = (i + 1) % len(polygon)
            bm.faces.new([bottom_verts[i], bottom_verts[next_i], top_verts[next_i], top_verts[i]])

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        renderer.shelf_objects.append(obj)

    # Right intermediate shelves
    for height in right_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf_polygon(
            right_depth, 29.0, 'W', amplitude, period, offset, corner_radius
        )

        import bmesh
        mesh = bpy.data.meshes.new(f"shelf_R{height}")
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        bm = bmesh.new()
        bottom_verts = [bm.verts.new((x, y, height)) for x, y in polygon]
        top_verts = [bm.verts.new((x, y, height + thickness)) for x, y in polygon]
        bm.verts.ensure_lookup_table()

        bm.faces.new(bottom_verts)
        bm.faces.new(reversed(top_verts))

        for i in range(len(polygon)):
            next_i = (i + 1) % len(polygon)
            bm.faces.new([bottom_verts[i], bottom_verts[next_i], top_verts[next_i], top_verts[i]])

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        renderer.shelf_objects.append(obj)

    print(f"  Created {len(left_intermediate_heights)} left + {len(right_intermediate_heights)} right = {len(left_intermediate_heights) + len(right_intermediate_heights)} intermediate shelves")
    print(f"\nTotal shelves: {len(renderer.shelf_objects)}")

    # Create walls
    print("\nCreating pantry walls...")
    wall_objects = renderer.create_pantry_room()
    print(f"  Created {len(wall_objects)} wall objects")

    # Apply materials
    print("Applying materials...")
    renderer.apply_materials()

    # Setup lighting
    print("Setting up lighting...")
    renderer.setup_lighting()

    # Output directory
    output_dir = Path('output/renders')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render settings
    resolution = (1920, 1080)
    samples = 128
    fov = 60.0  # Horizontal FOV

    print(f"\nRender settings:")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Samples: {samples}")
    print(f"  FOV: {fov}°")

    pantry_width = cfg.pantry['width']

    # RENDER 1: Center of doorway, 66" high, looking straight ahead
    print("\n" + "="*60)
    print("RENDER 1: Doorway center view")
    print("="*60)

    camera_pos = (pantry_width / 2, -12, 66)
    camera_rot = (90, 0, 0)

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    # Calculate lens from FOV: focal_length = sensor / (2 * tan(FOV/2))
    sensor_width = 36.0
    fov_rad = np.radians(fov)
    lens = sensor_width / (2 * np.tan(fov_rad / 2))

    camera = renderer.setup_camera(
        location=camera_pos,
        rotation=camera_rot,
        lens=lens,
        sensor_width=sensor_width,
        fisheye=False
    )

    output_file = output_dir / 'pantry_view1_center.png'
    print(f"  Rendering to: {output_file}")
    renderer.render(output_file, resolution, samples)
    print("  ✓ Complete")

    # RENDER 2: 6" right of center, looking 15° left
    print("\n" + "="*60)
    print("RENDER 2: Right of center, looking left")
    print("="*60)

    camera_pos = (pantry_width / 2 + 6, -12, 66)
    camera_rot = (90, 0, 15)

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    camera = renderer.setup_camera(
        location=camera_pos,
        rotation=camera_rot,
        lens=lens,
        sensor_width=sensor_width,
        fisheye=False
    )

    output_file = output_dir / 'pantry_view2_right_looking_left.png'
    print(f"  Rendering to: {output_file}")
    renderer.render(output_file, resolution, samples)
    print("  ✓ Complete")

    print("\n" + "="*60)
    print("SUCCESS! All renders complete.")
    print("="*60)
    print(f"\nView renders:")
    print(f"  - {output_dir}/pantry_view1_center.png")
    print(f"  - {output_dir}/pantry_view2_right_looking_left.png")


if __name__ == '__main__':
    main()
