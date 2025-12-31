#!/usr/bin/env blender --background --python
"""
Generate 3D renderings using exact geometry from SVG files.

Usage:
    blender --background --python scripts/render_from_svg.py
"""

import sys
import numpy as np
from pathlib import Path
import re
import xml.etree.ElementTree as ET

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Add user site-packages for any dependencies
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

try:
    import bpy
    import bmesh
    from mathutils import Euler
except ImportError:
    print("Error: This script must be run with Blender's Python interpreter")
    sys.exit(1)


def parse_svg_path(path_data):
    """
    Parse SVG path data to extract polygon vertices.

    Args:
        path_data: SVG path 'd' attribute string

    Returns:
        List of (x, y) coordinate tuples
    """
    vertices = []

    # Extract all coordinate pairs
    # Match patterns like "M x,y" or "L x,y"
    pattern = r'([ML])\s*([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)'
    matches = re.findall(pattern, path_data)

    for cmd, x, y in matches:
        vertices.append((float(x), float(y)))

    return vertices


def load_shelf_from_svg(svg_path):
    """
    Load shelf geometry from SVG file.

    Args:
        svg_path: Path to SVG file

    Returns:
        List of (x, y) vertices
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Find the path element (should be only one for our shelves)
    # Handle XML namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    path = root.find('.//svg:path', ns)
    if path is None:
        # Try without namespace
        path = root.find('.//path')

    if path is None:
        raise ValueError(f"No path found in SVG file: {svg_path}")

    path_data = path.get('d')
    vertices = parse_svg_path(path_data)

    return vertices


def create_shelf_mesh_from_svg(svg_path, height, thickness, name):
    """
    Create a 3D shelf mesh from SVG file.

    Args:
        svg_path: Path to SVG file
        height: Z-height for bottom of shelf (in inches)
        thickness: Shelf thickness (in inches)
        name: Object name

    Returns:
        Blender object
    """
    vertices_2d = load_shelf_from_svg(svg_path)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()

    # Create bottom vertices
    bottom_verts = []
    for x, y in vertices_2d:
        v = bm.verts.new((x, y, height))
        bottom_verts.append(v)

    # Create top vertices
    top_verts = []
    for x, y in vertices_2d:
        v = bm.verts.new((x, y, height + thickness))
        top_verts.append(v)

    bm.verts.ensure_lookup_table()

    # Create faces
    bottom_face = bm.faces.new(bottom_verts)
    bottom_face.normal_update()

    top_face = bm.faces.new(reversed(top_verts))
    top_face.normal_update()

    # Side faces
    num_verts = len(vertices_2d)
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
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
    bsdf.inputs['Base Color'].default_value = (0.85, 0.75, 0.55, 1.0)
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
    bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.85, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.8

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_pantry_walls(pantry_width=48, pantry_depth=49, pantry_height=105,
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


def setup_camera(location, rotation_euler, fov_horizontal=90.0):
    """Setup camera with specified FOV."""
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_data.type = 'PERSP'

    # Set FOV using sensor width and focal length
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


def setup_lighting(pantry_width=48, pantry_depth=49, pantry_height=105):
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
    print("Rendering Pantry from SVG Files")
    print("="*60)

    output_dir = Path('output')
    svg_dir = output_dir

    # Shelf definitions with heights
    main_shelf_heights = [19, 39, 59, 79]
    left_intermediate_heights = [9, 29, 49, 69]
    right_intermediate_heights = [5, 13, 26, 33, 46, 53, 66, 73, 86]

    thickness = 1.0  # 1" thick shelves

    print(f"\nClearing scene...")
    clear_scene()

    # Create materials
    shelf_material = create_shelf_material()
    wall_material = create_wall_material()

    # Create walls
    print("Creating pantry walls...")
    walls = create_pantry_walls()
    for wall in walls:
        if wall.data.materials:
            wall.data.materials[0] = wall_material
        else:
            wall.data.materials.append(wall_material)

    # Load shelves from SVG files
    print("\nLoading shelves from SVG files...")
    shelf_objects = []

    # Main shelves
    print("  Main shelves:")
    for height in main_shelf_heights:
        for wall_name in ['L', 'B', 'R']:
            svg_path = svg_dir / f'shelf_{wall_name}{height}_exact.svg'
            if svg_path.exists():
                print(f"    Loading {svg_path.name}...")
                obj = create_shelf_mesh_from_svg(svg_path, height, thickness, f'shelf_{wall_name}{height}')
                obj.data.materials.append(shelf_material)
                shelf_objects.append(obj)
            else:
                print(f"    WARNING: {svg_path} not found!")

    # Left intermediate shelves
    print("  Left intermediate shelves:")
    for height in left_intermediate_heights:
        svg_path = svg_dir / f'shelf_L{height}_exact.svg'
        if svg_path.exists():
            print(f"    Loading {svg_path.name}...")
            obj = create_shelf_mesh_from_svg(svg_path, height, thickness, f'shelf_L{height}')
            obj.data.materials.append(shelf_material)
            shelf_objects.append(obj)
        else:
            print(f"    WARNING: {svg_path} not found!")

    # Right intermediate shelves
    print("  Right intermediate shelves:")
    for height in right_intermediate_heights:
        svg_path = svg_dir / f'shelf_R{height}_exact.svg'
        if svg_path.exists():
            print(f"    Loading {svg_path.name}...")
            obj = create_shelf_mesh_from_svg(svg_path, height, thickness, f'shelf_R{height}')
            obj.data.materials.append(shelf_material)
            shelf_objects.append(obj)
        else:
            print(f"    WARNING: {svg_path} not found!")

    print(f"\nTotal shelves loaded: {len(shelf_objects)}")

    # Setup lighting
    print("\nSetting up lighting...")
    setup_lighting()

    # Render settings
    resolution = (2000, 2000)
    samples = 128
    fov = 110.0  # Wide FOV

    print(f"\nRender settings:")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Samples: {samples}")
    print(f"  FOV: {fov}°")

    render_dir = Path('output/renders')
    render_dir.mkdir(parents=True, exist_ok=True)

    # RENDER 1: Center of doorway, 66" high, looking down 30°
    print("\n" + "="*60)
    print("RENDER 1: Doorway center view, looking down 30°")
    print("="*60)

    camera_pos = (24.0, -12, 66)
    camera_rot = (60, 0, 0)  # 90 - 30 = 60 (looking down 30°)

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    setup_camera(camera_pos, camera_rot, fov_horizontal=fov)

    output_file = render_dir / 'pantry_view1_center_svg.png'
    print(f"  Rendering to: {output_file}")
    render_scene(output_file, resolution, samples)
    print("  ✓ Complete")

    # RENDER 2: 6" right of center, 33" high, 6" into pantry, looking 15° left and 20° up
    print("\n" + "="*60)
    print("RENDER 2: Low angle view, looking left and up")
    print("="*60)

    camera_pos = (30.0, -6, 33)  # 6" into pantry: y = -12 + 6 = -6
    camera_rot = (110, 0, 15)  # 90 + 20 = 110 (looking up 20°), yaw 15° left

    print(f"  Camera position: {camera_pos}")
    print(f"  Camera rotation: {camera_rot}°")

    setup_camera(camera_pos, camera_rot, fov_horizontal=fov)

    output_file = render_dir / 'pantry_view2_right_looking_left_svg.png'
    print(f"  Rendering to: {output_file}")
    render_scene(output_file, resolution, samples)
    print("  ✓ Complete")

    print("\n" + "="*60)
    print("SUCCESS! All renders complete.")
    print("="*60)
    print(f"\nView renders:")
    print(f"  - {render_dir}/pantry_view1_center_svg.png")
    print(f"  - {render_dir}/pantry_view2_right_looking_left_svg.png")


if __name__ == '__main__':
    main()
