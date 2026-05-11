#!/usr/bin/env blender --background --python
"""
Generic furniture renderer — runs inside Blender.

Usage:
    blender --background --python lib/blender_render_furniture.py \\
            -- /path/to/geometry.json /path/to/output_dir

The geometry JSON is produced by blender_renderer.build_furniture_geometry().
Each piece specifies a 2D polygon, a u/v axis mapping to 3D, an extrusion
normal, and a thickness. This covers horizontal shelves, vertical boards,
and any flat-cut piece that needs extruding.
"""

import json
import math
import sys
from pathlib import Path

try:
    import bpy
    import bmesh
    from mathutils import Vector
except ImportError:
    print("Error: must be run with Blender's Python interpreter")
    sys.exit(1)

# ── Parse CLI args ─────────────────────────────────────────────────────────

try:
    after_dash = sys.argv[sys.argv.index("--") + 1:]
except ValueError:
    after_dash = []

if len(after_dash) < 2:
    print("Usage: blender --background --python blender_render_furniture.py "
          "-- geometry.json output_dir")
    sys.exit(1)

GEO_JSON = Path(after_dash[0])
OUTPUT_DIR = Path(after_dash[1])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Materials ──────────────────────────────────────────────────────────────

def _mat_plywood():
    mat = bpy.data.materials.new("plywood")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.82, 0.70, 0.50, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.55
    bsdf.inputs['Specular IOR Level'].default_value = 0.2
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def _mat_walnut():
    mat = bpy.data.materials.new("walnut")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.07, 0.038, 0.018, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.40
    bsdf.inputs['Specular IOR Level'].default_value = 0.45
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def _mat_wall():
    mat = bpy.data.materials.new("wall")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.92, 0.91, 0.87, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.90
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    out = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


_MAT_FACTORIES = {
    'plywood': _mat_plywood,
    'walnut':  _mat_walnut,
    'wall':    _mat_wall,
}
MATERIALS = {}

def get_material(name: str):
    if name not in MATERIALS:
        factory = _MAT_FACTORIES.get(name, _mat_plywood)
        MATERIALS[name] = factory()
    return MATERIALS[name]


# ── Mesh creation ──────────────────────────────────────────────────────────

def create_piece_mesh(piece: dict) -> bpy.types.Object:
    """
    Extrude a 2D polygon into a 3D solid.

    World position of polygon vertex (u, v):
        P_world = origin + u * u_axis + v * v_axis

    The solid is then extruded by `thickness` along `normal`.
    """
    poly  = piece['polygon']          # [[u,v], ...]
    orig  = Vector(piece['origin'])
    u_ax  = Vector(piece['u_axis'])
    v_ax  = Vector(piece['v_axis'])
    norm  = Vector(piece['normal'])
    thick = piece['thickness']
    name  = piece['name']

    # Compute 3D positions of bottom face
    bottom_pts = [orig + pt[0] * u_ax + pt[1] * v_ax for pt in poly]
    top_pts    = [p + thick * norm for p in bottom_pts]

    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    bm = bmesh.new()

    bv = [bm.verts.new(p) for p in bottom_pts]
    tv = [bm.verts.new(p) for p in top_pts]
    bm.verts.ensure_lookup_table()

    n = len(poly)
    try:
        bm.faces.new(bv)
    except Exception:
        pass  # non-planar — side faces still provide the shape
    try:
        bm.faces.new(list(reversed(tv)))
    except Exception:
        pass

    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([bv[i], bv[j], tv[j], tv[i]])
        except Exception:
            pass  # skip degenerate edges

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    mat = get_material(piece.get('material', 'plywood'))
    obj.data.materials.append(mat)
    return obj


# ── Lighting ───────────────────────────────────────────────────────────────

def setup_lighting(scene_center: Vector, scene_height: float):
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    cx, cy, cz = scene_center

    # Key light — area lamp above and to one side
    key = bpy.data.lights.new("key", type='AREA')
    key.energy   = 8000
    key.size     = 48.0
    key.color    = (1.0, 0.97, 0.93)
    ko = bpy.data.objects.new("key", key)
    ko.location  = (cx - 30, cy - 40, scene_height * 1.2)
    ko.rotation_euler = (math.radians(40), 0, math.radians(-30))
    bpy.context.scene.collection.objects.link(ko)

    # Fill light — softer, from the right
    fill = bpy.data.lights.new("fill", type='AREA')
    fill.energy  = 3000
    fill.size    = 60.0
    fill.color   = (0.88, 0.92, 1.0)
    fo = bpy.data.objects.new("fill", fill)
    fo.location  = (cx + 50, cy + 20, scene_height * 0.7)
    fo.rotation_euler = (math.radians(50), 0, math.radians(120))
    bpy.context.scene.collection.objects.link(fo)

    # Rim light — thin highlight from behind
    rim = bpy.data.lights.new("rim", type='AREA')
    rim.energy   = 2000
    rim.size     = 30.0
    ro = bpy.data.objects.new("rim", rim)
    ro.location  = (cx - 5, cy + 60, scene_height * 0.9)
    ro.rotation_euler = (math.radians(-30), 0, math.radians(180))
    bpy.context.scene.collection.objects.link(ro)


# ── Camera ─────────────────────────────────────────────────────────────────

def setup_camera(location, target, fov_deg: float = 60.0) -> bpy.types.Object:
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    sensor_w = 36.0
    fov_rad  = math.radians(fov_deg)
    focal    = sensor_w / (2 * math.tan(fov_rad / 2))

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type         = 'PERSP'
    cam_data.lens         = focal
    cam_data.sensor_width = sensor_w
    cam_data.clip_end     = 5000.0

    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = Vector(location)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Point camera at target
    direction = Vector(target) - Vector(location)
    rot_q = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_q.to_euler()

    return cam


# ── Render ─────────────────────────────────────────────────────────────────

def render_view(render_spec: dict):
    scene = bpy.context.scene

    w, h = render_spec.get('resolution', [1920, 1080])
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    scene.render.engine = 'CYCLES'
    scene.cycles.samples = render_spec.get('samples', 64)
    scene.cycles.use_denoising = True

    # Use GPU if available
    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        try:
            bpy.context.scene.cycles.device = 'GPU'
        except Exception:
            pass

    out_path = OUTPUT_DIR / f"{render_spec['name']}.png"
    scene.render.filepath = str(out_path)

    setup_camera(
        render_spec['camera_location'],
        render_spec['camera_target'],
        render_spec.get('fov_deg', 60.0),
    )

    bpy.ops.render.render(write_still=True)
    print(f"  Saved: {out_path}")


# ── Scene context helpers ──────────────────────────────────────────────────

def add_floor(geo: dict):
    """Thin floor plane under the assembly for grounding shadows."""
    all_pts = []
    for p in geo['pieces']:
        o  = Vector(p['origin'])
        ua = Vector(p['u_axis'])
        va = Vector(p['v_axis'])
        for uv in p['polygon']:
            all_pts.append(o + uv[0]*ua + uv[1]*va)

    if not all_pts:
        return

    xs = [p.x for p in all_pts]; ys = [p.y for p in all_pts]
    margin = 10.0
    x0, x1 = min(xs)-margin, max(xs)+margin
    y0, y1 = min(ys)-margin, max(ys)+margin

    mesh = bpy.data.meshes.new("floor")
    obj  = bpy.data.objects.new("floor", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.verts.new((x0, y0, -0.1))
    bm.verts.new((x1, y0, -0.1))
    bm.verts.new((x1, y1, -0.1))
    bm.verts.new((x0, y1, -0.1))
    bm.verts.ensure_lookup_table()
    bm.faces.new(bm.verts)
    bm.to_mesh(mesh); bm.free(); mesh.update()
    obj.data.materials.append(get_material('wall'))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Furniture Blender Renderer")
    print(f"  Geometry: {GEO_JSON}")
    print(f"  Output:   {OUTPUT_DIR}")
    print("=" * 60)

    geo = json.loads(GEO_JSON.read_text())

    # Clear default scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in [bpy.data.meshes, bpy.data.lights, bpy.data.cameras,
                  bpy.data.materials]:
        for item in list(block):
            block.remove(item)

    # Set world background to a neutral light grey
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or \
         world.node_tree.nodes.new('ShaderNodeBackground')
    bg.inputs['Color'].default_value = (0.85, 0.87, 0.90, 1.0)
    bg.inputs['Strength'].default_value = 0.8

    # Build pieces
    print(f"\nBuilding {len(geo['pieces'])} pieces...")
    all_z = []
    for piece in geo['pieces']:
        obj = create_piece_mesh(piece)
        orig = Vector(piece['origin'])
        norm = Vector(piece['normal'])
        all_z.append(orig.z)
        all_z.append((orig + piece['thickness'] * norm).z)
        print(f"  {piece['name']}: {len(piece['polygon'])} verts")

    add_floor(geo)

    # Compute scene bounds for lighting
    scene_height = max(all_z) if all_z else 80.0
    # Find approximate scene center from first piece's bounding box
    scene_center = Vector((16.0, 10.0, scene_height / 2))
    setup_lighting(scene_center, scene_height)

    # Render each view
    print(f"\nRendering {len(geo['renders'])} views...")
    for render_spec in geo['renders']:
        print(f"\n  View: {render_spec['name']}")
        render_view(render_spec)

    print("\nDone.")


if __name__ == '__main__':
    main()
