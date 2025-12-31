#!/usr/bin/env blender --background --python
"""
Generate 3D renderings of complete pantry with all shelves (main + intermediate).

Usage:
    blender --background --python scripts/render_full_pantry.py

Renders two views:
  1. From 66" high, doorway center, looking straight ahead
  2. From 66" high, 6" right of center, looking 15° left
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

try:
    import bpy
    import bmesh
    from mathutils import Vector, Euler
except ImportError:
    print("Error: This script must be run with Blender's Python interpreter")
    print("Usage: blender --background --python scripts/render_full_pantry.py")
    sys.exit(1)

import config

ShelfConfig = config.ShelfConfig


def simple_minimize_scalar(func, bounds, tol=1e-6, max_iter=100):
    """
    Simple golden section search for scalar minimization (no scipy needed).

    Args:
        func: Function to minimize
        bounds: (lower, upper) bounds tuple
        tol: Tolerance for convergence
        max_iter: Maximum iterations

    Returns:
        Result object with .x attribute containing the minimum
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
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

    # Return object with .x attribute like scipy
    class Result:
        def __init__(self, x):
            self.x = x

    return Result((a + b) / 2)


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def create_shelf_material():
    """Create material for shelves (Baltic birch)."""
    mat = bpy.data.materials.new(name="ShelfMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.85, 0.75, 0.55, 1.0)  # Light birch
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Specular IOR Level'].default_value = 0.3

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_wall_material():
    """Create material for walls."""
    mat = bpy.data.materials.new(name="WallMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.85, 1.0)  # Light gray
    bsdf.inputs['Roughness'].default_value = 0.8

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_shelf_from_polygon(polygon, height, thickness, name):
    """
    Create a 3D shelf mesh from a 2D polygon by extruding.

    Args:
        polygon: Nx2 array of (x, y) coordinates
        height: Z-height for bottom of shelf (in inches)
        thickness: Shelf thickness (in inches)
        name: Object name

    Returns:
        Blender object
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()

    # Create bottom vertices
    bottom_verts = []
    for x, y in polygon:
        v = bm.verts.new((x, y, height))
        bottom_verts.append(v)

    # Create top vertices
    top_verts = []
    for x, y in polygon:
        v = bm.verts.new((x, y, height + thickness))
        top_verts.append(v)

    bm.verts.ensure_lookup_table()

    # Create faces
    bottom_face = bm.faces.new(bottom_verts)
    bottom_face.normal_update()

    top_face = bm.faces.new(reversed(top_verts))
    top_face.normal_update()

    # Side faces
    num_points = len(polygon)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        face = bm.faces.new([
            bottom_verts[i],
            bottom_verts[next_i],
            top_verts[next_i],
            top_verts[i]
        ])
        face.normal_update()

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return obj


def generate_intermediate_shelf(depth, length, side, amplitude, period, offset, corner_radius=3.0):
    """
    Generate intermediate shelf polygon (duplicated from extract_and_export_geometry.py).

    Args:
        depth: Base depth in inches (7" for left, 4" for right)
        length: Length from door inward (29")
        side: 'E' for left/east wall, 'W' for right/west wall
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Random phase offset
        corner_radius: Radius for back interior corner (default 3")

    Returns:
        Polygon array with sinusoidal interior edge and rounded back interior corner
    """
    # Generate sinusoid points
    y_points = np.linspace(0, length, 100)

    # Solve for tangent circle
    center_y = length - corner_radius

    def objective(y):
        if side == 'E':
            x_sinusoid = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
            center_x_estimate = x_sinusoid - corner_radius
        else:  # 'W'
            x_sinusoid = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
            center_x_estimate = x_sinusoid + corner_radius

        dist = np.sqrt((x_sinusoid - center_x_estimate)**2 + (y - center_y)**2)
        return abs(dist - corner_radius)

    result = simple_minimize_scalar(objective, bounds=(length - 2*corner_radius, length))
    tangent_y = result.x

    # Calculate center and tangency points
    if side == 'E':
        x_sinusoid_tangent = depth + amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid_tangent - corner_radius
    else:
        x_sinusoid_tangent = 48 - depth - amplitude * np.sin(2 * np.pi * tangent_y / period + offset)
        center_x = x_sinusoid_tangent + corner_radius

    x_horizontal_tangent = center_x
    y_horizontal_tangent = length

    angle_horizontal = np.pi/2
    angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

    polygon = []

    if side == 'E':
        # Left shelf
        polygon.append([0, 0])
        polygon.append([0, length])
        polygon.append([x_horizontal_tangent, y_horizontal_tangent])

        # Arc
        arc_angles = np.linspace(angle_horizontal, angle_sinusoid, 20)
        for angle in arc_angles:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        # Sinusoid
        for y in y_points[::-1]:
            if y <= tangent_y:
                x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        x_north = depth + amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])

    else:  # 'W'
        # Right shelf
        polygon.append([48, 0])
        x_north = 48 - depth - amplitude * np.sin(2 * np.pi * 0 / period + offset)
        polygon.append([x_north, 0])

        # Sinusoid
        for y in y_points:
            if y <= tangent_y:
                x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                polygon.append([x, y])

        # Arc
        arc_angles = np.linspace(angle_sinusoid, angle_horizontal, 20)
        for angle in arc_angles:
            x = center_x + corner_radius * np.cos(angle)
            y = center_y + corner_radius * np.sin(angle)
            polygon.append([x, y])

        polygon.append([48, length])

    return np.array(polygon)


def create_pantry_walls(pantry_width, pantry_depth, pantry_height,
                       door_clearance_east=6, door_clearance_west=4):
    """Create pantry room walls."""
    walls = []
    wall_thickness = 0.5

    # East wall (x=0)
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(-wall_thickness/2, pantry_depth/2, pantry_height/2)
    )
    east_wall = bpy.context.active_object
    east_wall.scale = (wall_thickness, pantry_depth, pantry_height)
    east_wall.name = "wall_east"
    walls.append(east_wall)

    # West wall (x=width)
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(pantry_width + wall_thickness/2, pantry_depth/2, pantry_height/2)
    )
    west_wall = bpy.context.active_object
    west_wall.scale = (wall_thickness, pantry_depth, pantry_height)
    west_wall.name = "wall_west"
    walls.append(west_wall)

    # South wall (y=depth)
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(pantry_width/2, pantry_depth + wall_thickness/2, pantry_height/2)
    )
    south_wall = bpy.context.active_object
    south_wall.scale = (pantry_width, wall_thickness, pantry_height)
    south_wall.name = "wall_south"
    walls.append(south_wall)

    # North wall - East side of door
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(door_clearance_east/2, -wall_thickness/2, pantry_height/2)
    )
    north_east = bpy.context.active_object
    north_east.scale = (door_clearance_east, wall_thickness, pantry_height)
    north_east.name = "wall_north_east"
    walls.append(north_east)

    # North wall - West side of door
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(pantry_width - door_clearance_west/2, -wall_thickness/2, pantry_height/2)
    )
    north_west = bpy.context.active_object
    north_west.scale = (door_clearance_west, wall_thickness, pantry_height)
    north_west.name = "wall_north_west"
    walls.append(north_west)

    return walls


def setup_camera(location, rotation_euler, fov_horizontal=60.0):
    """
    Setup camera with regular perspective (no fisheye).

    Args:
        location: (x, y, z) position
        rotation_euler: (x, y, z) rotation in degrees
        fov_horizontal: Horizontal field of view in degrees (default 60°, 20% wider than normal 50°)
    """
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_data.type = 'PERSP'

    # Set FOV using sensor width and focal length
    # Default sensor width is 36mm (full frame)
    # FOV = 2 * atan(sensor / (2 * focal_length))
    # Solving for focal_length: focal_length = sensor / (2 * tan(FOV/2))
    sensor_width = 36.0
    fov_rad = np.radians(fov_horizontal)
    focal_length = sensor_width / (2 * np.tan(fov_rad / 2))

    camera_data.lens = focal_length
    camera_data.sensor_width = sensor_width

    camera_object = bpy.data.objects.new("Camera", camera_data)
    camera_object.location = location
    camera_object.rotation_euler = Euler([
        np.radians(rotation_euler[0]),
        np.radians(rotation_euler[1]),
        np.radians(rotation_euler[2])
    ], 'XYZ')

    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object

    return camera_object


def setup_lighting(pantry_width, pantry_depth, pantry_height):
    """Setup scene lighting."""
    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Main light - center, near ceiling
    light_data = bpy.data.lights.new(name="MainLight", type='POINT')
    light_data.energy = 2000
    light_data.shadow_soft_size = 2.0
    light_object = bpy.data.objects.new(name="MainLight", object_data=light_data)
    light_object.location = (pantry_width/2, pantry_depth/2, pantry_height * 0.85)
    bpy.context.scene.collection.objects.link(light_object)

    # Fill light - near door
    fill_data = bpy.data.lights.new(name="FillLight", type='POINT')
    fill_data.energy = 800
    fill_object = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    fill_object.location = (pantry_width/2, -10, 60)
    bpy.context.scene.collection.objects.link(fill_object)


def render_scene(output_path, resolution=(1920, 1080), samples=128):
    """Render the scene."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def main():
    print("="*60)
    print("Rendering Full Pantry with All Shelves")
    print("="*60)

    # Load configuration
    config_path = Path('configs/pantry_0002.json')
    print(f"\nLoading config: {config_path}")
    cfg = ShelfConfig.from_file(config_path)

    # Pantry dimensions
    pantry_width = cfg.pantry['width']  # 48"
    pantry_depth = cfg.pantry['depth']  # 49"
    pantry_height = cfg.pantry['height']  # 105"

    # Shelf parameters
    amplitude = cfg.design_params['sinusoid_amplitude']
    period = cfg.design_params['sinusoid_period']
    left_depth = cfg.design_params['shelf_base_depth_east']  # 7"
    right_depth = cfg.design_params['shelf_base_depth_west']  # 4"
    corner_radius = cfg.design_params['interior_corner_radius']  # 3"
    thickness = cfg.design_params['shelf_thickness']  # 1"

    # Heights for all shelves (bottom-referenced)
    main_shelf_heights = [19, 39, 59, 79]
    left_intermediate_heights = [9, 29, 49, 69]
    right_intermediate_heights = [5, 13, 26, 33, 43, 46, 66, 73, 86]

    print(f"\nPantry: {pantry_width}\" × {pantry_depth}\" × {pantry_height}\"")
    print(f"Main shelves: {len(main_shelf_heights)} at heights {main_shelf_heights}")
    print(f"Left intermediate: {len(left_intermediate_heights)} at heights {left_intermediate_heights}")
    print(f"Right intermediate: {len(right_intermediate_heights)} at heights {right_intermediate_heights}")

    # Clear scene
    print("\nClearing scene...")
    clear_scene()

    # Create materials
    shelf_material = create_shelf_material()
    wall_material = create_wall_material()

    # Create walls
    print("Creating pantry walls...")
    walls = create_pantry_walls(pantry_width, pantry_depth, pantry_height,
                                door_clearance_east=6, door_clearance_west=4)
    for wall in walls:
        if wall.data.materials:
            wall.data.materials[0] = wall_material
        else:
            wall.data.materials.append(wall_material)

    # Import geometry functions to extract main shelves
    # We'll use the same approach as extract_and_export_geometry.py
    sys.path.insert(0, str(Path(__file__).parent))
    from extract_and_export_geometry import extract_exact_shelf_geometries

    levels = sorted(set(s['level'] for s in cfg.shelves))

    # Create main shelves
    print("\nCreating main shelves...")
    shelf_count = 0
    for i, height in enumerate(main_shelf_heights):
        if i >= len(levels):
            continue

        level = levels[i]
        print(f"  Height {height}\": extracting geometry...")
        geom_data = extract_exact_shelf_geometries(cfg, level)

        if not geom_data:
            print(f"    Failed!")
            continue

        # Create left, back, right shelves
        wall_to_name = {'E': 'L', 'S': 'B', 'W': 'R'}
        for wall in ['E', 'S', 'W']:
            polygon = geom_data[wall]
            wall_name = wall_to_name[wall]
            obj = create_shelf_from_polygon(
                polygon, height, thickness,
                f"shelf_{wall_name}{height}"
            )
            obj.data.materials.append(shelf_material)
            shelf_count += 1

    print(f"  Created {shelf_count} main shelf pieces")

    # Create left intermediate shelves
    print("\nCreating left intermediate shelves...")
    np.random.seed(42)
    for height in left_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(
            left_depth, 29.0, 'E', amplitude, period, offset, corner_radius
        )
        obj = create_shelf_from_polygon(
            polygon, height, thickness,
            f"shelf_L{height}"
        )
        obj.data.materials.append(shelf_material)
        shelf_count += 1
    print(f"  Created {len(left_intermediate_heights)} left intermediate shelves")

    # Create right intermediate shelves
    print("\nCreating right intermediate shelves...")
    for height in right_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(
            right_depth, 29.0, 'W', amplitude, period, offset, corner_radius
        )
        obj = create_shelf_from_polygon(
            polygon, height, thickness,
            f"shelf_R{height}"
        )
        obj.data.materials.append(shelf_material)
        shelf_count += 1
    print(f"  Created {len(right_intermediate_heights)} right intermediate shelves")

    print(f"\nTotal shelves created: {shelf_count}")

    # Setup lighting
    print("\nSetting up lighting...")
    setup_lighting(pantry_width, pantry_depth, pantry_height)

    # Output directory
    output_dir = Path('output/renders')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render settings
    resolution = (1920, 1080)
    samples = 128
    fov = 60.0  # 20% wider than typical 50°

    print(f"\nRender settings:")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Samples: {samples}")
    print(f"  FOV: {fov}° (horizontal)")

    # RENDER 1: Center of doorway, 66" high, looking straight ahead
    print("\n" + "="*60)
    print("RENDER 1: Doorway center view")
    print("="*60)

    camera_pos = (pantry_width / 2, -12, 66)  # Center, 12" outside door, 66" high
    camera_rot = (90, 0, 0)  # Looking straight ahead (90° pitch = horizontal)

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    setup_camera(camera_pos, camera_rot, fov_horizontal=fov)

    output_file = output_dir / 'pantry_view1_center.png'
    print(f"  Rendering to: {output_file}")
    render_scene(output_file, resolution, samples)
    print("  ✓ Complete")

    # RENDER 2: 6" right of center, 66" high, looking 15° left
    print("\n" + "="*60)
    print("RENDER 2: Right of center, looking left")
    print("="*60)

    camera_pos = (pantry_width / 2 + 6, -12, 66)  # 6" right of center
    camera_rot = (90, 0, 15)  # 15° yaw to the left

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    setup_camera(camera_pos, camera_rot, fov_horizontal=fov)

    output_file = output_dir / 'pantry_view2_right_looking_left.png'
    print(f"  Rendering to: {output_file}")
    render_scene(output_file, resolution, samples)
    print("  ✓ Complete")

    print("\n" + "="*60)
    print("SUCCESS! All renders complete.")
    print("="*60)
    print(f"\nView renders:")
    print(f"  - {output_dir}/pantry_view1_center.png")
    print(f"  - {output_dir}/pantry_view2_right_looking_left.png")


if __name__ == '__main__':
    main()
