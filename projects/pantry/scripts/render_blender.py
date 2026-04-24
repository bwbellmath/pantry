#!/usr/bin/env blender --background --python
"""
Blender rendering script for pantry shelves.

Reads pre-extracted shelf geometry from a JSON file and renders
the pantry with walls, door-jam trim, and all shelves.

Usage (called by render_from_svg.py):
    blender --background --python scripts/render_blender.py -- /path/to/geometry.json
"""

import json
import math
import sys
from pathlib import Path

try:
    import bpy
    import bmesh
    from mathutils import Euler
except ImportError:
    print("Error: This script must be run with Blender's Python interpreter")
    sys.exit(1)

# Parse arguments after '--'
try:
    argv = sys.argv[sys.argv.index("--") + 1:]
except ValueError:
    argv = []

if not argv:
    print("Error: No geometry JSON path provided")
    sys.exit(1)

geometry_path = argv[0]


# ---------------------------------------------------------------------------
# Pantry constants
# ---------------------------------------------------------------------------
DOOR_CLEARANCE_EAST = 6.0
DOOR_CLEARANCE_WEST = 4.0
WALL_THICKNESS = 3.5
DOOR_JAM_THICKNESS = 0.75
DOOR_JAM_WIDTH = 3.5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def add_box(name, location, scale):
    """Add a unit cube scaled to desired dimensions."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.scale = scale
    obj.name = name
    return obj


def create_shelf_mesh(vertices_2d, height, thickness, name):
    """Extrude a 2D polygon into a 3D solid shelf."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()

    bottom_verts = [bm.verts.new((x, y, height)) for x, y in vertices_2d]
    top_verts = [bm.verts.new((x, y, height + thickness)) for x, y in vertices_2d]
    bm.verts.ensure_lookup_table()

    bm.faces.new(bottom_verts)
    bm.faces.new(list(reversed(top_verts)))

    n = len(vertices_2d)
    for i in range(n):
        ni = (i + 1) % n
        bm.faces.new([bottom_verts[i], bottom_verts[ni], top_verts[ni], top_verts[i]])

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def create_shelf_material():
    """Baltic birch plywood material."""
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
    """Drywall / painted wall material."""
    mat = bpy.data.materials.new(name="WallMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.92, 0.91, 0.87, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.85

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


def create_doorjam_material():
    """Painted wood trim material."""
    mat = bpy.data.materials.new(name="DoorJamMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.93, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.3

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return mat


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def create_pantry_walls(pw, pd, ph):
    """Create pantry walls with door opening."""
    walls = []
    wt = WALL_THICKNESS

    # East wall (x=0)
    walls.append(add_box("wall_east",
        location=(-wt/2, pd/2, ph/2),
        scale=(wt, pd + wt, ph)))

    # West wall (x=width)
    walls.append(add_box("wall_west",
        location=(pw + wt/2, pd/2, ph/2),
        scale=(wt, pd + wt, ph)))

    # South wall (y=depth, back wall)
    walls.append(add_box("wall_south",
        location=(pw/2, pd + wt/2, ph/2),
        scale=(pw, wt, ph)))

    # North wall — east side of door
    ew = DOOR_CLEARANCE_EAST
    walls.append(add_box("wall_north_east",
        location=(ew/2, -wt/2, ph/2),
        scale=(ew, wt, ph)))

    # North wall — west side of door
    ww = DOOR_CLEARANCE_WEST
    walls.append(add_box("wall_north_west",
        location=(pw - ww/2, -wt/2, ph/2),
        scale=(ww, wt, ph)))

    # Header above door
    door_width = pw - DOOR_CLEARANCE_EAST - DOOR_CLEARANCE_WEST
    header_h = 3.5
    walls.append(add_box("wall_header",
        location=(pw/2, -wt/2, ph - header_h/2),
        scale=(door_width, wt, header_h)))

    return walls


def create_door_jams(pw, ph):
    """Create door-jam trim that frames the door opening."""
    jams = []
    jt = DOOR_JAM_THICKNESS
    jw = DOOR_JAM_WIDTH

    door_left_x = DOOR_CLEARANCE_EAST
    door_right_x = pw - DOOR_CLEARANCE_WEST
    door_top_z = ph - 3.5  # bottom of header

    # Left jam (east side)
    jams.append(add_box("doorjam_left",
        location=(door_left_x + jt/2, -jw/2, door_top_z/2),
        scale=(jt, jw, door_top_z)))

    # Right jam (west side)
    jams.append(add_box("doorjam_right",
        location=(door_right_x - jt/2, -jw/2, door_top_z/2),
        scale=(jt, jw, door_top_z)))

    # Head jam
    head_width = door_right_x - door_left_x
    jams.append(add_box("doorjam_head",
        location=(pw/2, -jw/2, door_top_z + jt/2),
        scale=(head_width, jw, jt)))

    return jams


# ---------------------------------------------------------------------------
# Lighting & Camera
# ---------------------------------------------------------------------------

def setup_lighting(pw, pd, ph):
    """Setup scene lighting."""
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    light = bpy.data.lights.new(name="MainLight", type='POINT')
    light.energy = 2000
    light.shadow_soft_size = 2.0
    lo = bpy.data.objects.new("MainLight", light)
    lo.location = (pw/2, pd/2, ph * 0.85)
    bpy.context.scene.collection.objects.link(lo)

    fill = bpy.data.lights.new(name="FillLight", type='POINT')
    fill.energy = 800
    fo = bpy.data.objects.new("FillLight", fill)
    fo.location = (pw/2, -10, 60)
    bpy.context.scene.collection.objects.link(fo)


def setup_camera(location, rotation_euler, fov_horizontal=90.0):
    """Setup perspective camera with given horizontal FOV (degrees)."""
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.type = 'PERSP'
    sensor_width = 36.0
    fov_rad = math.radians(fov_horizontal)
    cam_data.lens = sensor_width / (2 * math.tan(fov_rad / 2))
    cam_data.sensor_width = sensor_width

    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = location
    cam.rotation_euler = Euler([
        math.radians(rotation_euler[0]),
        math.radians(rotation_euler[1]),
        math.radians(rotation_euler[2])
    ], 'XYZ')

    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def render_scene(output_path, resolution=(1920, 1080), samples=128):
    """Render the current scene to a PNG file."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Blender Pantry Renderer")
    print("=" * 60)

    # Load geometry
    print(f"\nLoading geometry from: {geometry_path}")
    with open(geometry_path, 'r') as f:
        data = json.load(f)

    pw = data['pantry_width']
    pd = data['pantry_depth']
    ph = data['pantry_height']
    thickness = data['shelf_thickness']
    shelves = data['shelves']

    print(f"  Pantry: {pw}\" x {pd}\" x {ph}\"")
    print(f"  Shelves: {len(shelves)}")

    render_dir = Path('output/renders')
    render_dir.mkdir(parents=True, exist_ok=True)

    # Clear scene
    print("\nClearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Materials
    shelf_mat = create_shelf_material()
    wall_mat = create_wall_material()
    jam_mat = create_doorjam_material()

    # Walls
    print("Creating pantry walls...")
    walls = create_pantry_walls(pw, pd, ph)
    for w in walls:
        w.data.materials.append(wall_mat)
    print(f"  {len(walls)} wall objects")

    # Door jams
    print("Creating door-jam trim...")
    jams = create_door_jams(pw, ph)
    for j in jams:
        j.data.materials.append(jam_mat)
    print(f"  {len(jams)} door-jam pieces")

    # Create shelf meshes
    print(f"\nCreating {len(shelves)} shelf meshes...")
    for shelf_info in shelves:
        verts = [tuple(v) for v in shelf_info['vertices']]
        obj = create_shelf_mesh(verts, shelf_info['height'], thickness, shelf_info['name'])
        obj.data.materials.append(shelf_mat)
        print(f"  {shelf_info['name']}: {len(verts)} verts at {shelf_info['height']}\"")

    # Lighting
    print("\nSetting up lighting...")
    setup_lighting(pw, pd, ph)

    # Render settings
    resolution = (2000, 2000)
    samples = 128
    fov = 110.0

    print(f"\nRender settings: {resolution[0]}x{resolution[1]}, {samples} samples")

    # RENDER 1: Doorway center, eye level, looking down
    print("\n" + "=" * 60)
    print("RENDER 1: Doorway center view")
    print("=" * 60)
    setup_camera((24.0, -12, 66), (60, 0, 0), fov_horizontal=fov)
    out1 = render_dir / 'pantry_doorway_center.png'
    render_scene(out1, resolution, samples)
    print(f"  Saved: {out1}")

    # RENDER 2: Low angle, right of center
    print("\n" + "=" * 60)
    print("RENDER 2: Low angle, looking left and up")
    print("=" * 60)
    setup_camera((30.0, -6, 33), (110, 0, 15), fov_horizontal=fov)
    out2 = render_dir / 'pantry_low_right.png'
    render_scene(out2, resolution, samples)
    print(f"  Saved: {out2}")

    # RENDER 3: Door-jam close-up (right side)
    print("\n" + "=" * 60)
    print("RENDER 3: Door-jam close-up (right side)")
    print("=" * 60)
    setup_camera((40.0, -8, 40), (75, 0, -10), fov_horizontal=80.0)
    out3 = render_dir / 'pantry_doorjam_right.png'
    render_scene(out3, resolution, samples)
    print(f"  Saved: {out3}")

    # RENDER 4: Door-jam close-up (left side)
    print("\n" + "=" * 60)
    print("RENDER 4: Door-jam close-up (left side)")
    print("=" * 60)
    setup_camera((8.0, -8, 40), (75, 0, 10), fov_horizontal=80.0)
    out4 = render_dir / 'pantry_doorjam_left.png'
    render_scene(out4, resolution, samples)
    print(f"  Saved: {out4}")

    print("\n" + "=" * 60)
    print("All renders complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    for p in [out1, out2, out3, out4]:
        print(f"  {p}")


if __name__ == '__main__':
    main()
