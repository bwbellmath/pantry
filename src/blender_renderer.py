"""
3D rendering of pantry shelves using Blender Python API.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

try:
    import bpy
    import bmesh
    from mathutils import Vector, Euler
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    bpy = None  # type: ignore
    print("Warning: Blender Python API (bpy) not available. Run with Blender's Python interpreter.")

import shelf_generator
import config

ShelfFootprint = shelf_generator.ShelfFootprint
ShelfGenerator = shelf_generator.ShelfGenerator
ShelfConfig = config.ShelfConfig


class BlenderRenderer:
    """Renders pantry shelves in 3D using Blender."""

    def __init__(self, config: ShelfConfig):
        """
        Initialize Blender renderer.

        Args:
            config: ShelfConfig instance
        """
        if not BLENDER_AVAILABLE:
            raise ImportError("Blender Python API not available")

        self.config = config
        self.generator = ShelfGenerator(config)
        self.scene = bpy.context.scene
        self.shelf_objects = []

    def clear_scene(self):
        """Clear all objects from the scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

    def create_shelf_mesh(self, footprint: ShelfFootprint, height: float) -> 'bpy.types.Object':
        """
        Create a 3D mesh for a shelf by extruding its 2D footprint.

        Args:
            footprint: ShelfFootprint with 2D outline
            height: Z-height to place the shelf

        Returns:
            Blender mesh object
        """
        # Create mesh and object
        mesh = bpy.data.meshes.new(f"shelf_L{footprint.level}_{footprint.wall}")
        obj = bpy.data.objects.new(mesh.name, mesh)

        # Link to scene
        self.scene.collection.objects.link(obj)

        # Create bmesh
        bm = bmesh.new()

        # Get outline points
        outline = footprint.outline_points
        num_points = len(outline)

        # Create bottom vertices (at z = height)
        bottom_verts = []
        for x, y in outline:
            # Convert inches to Blender units (we'll use inches as units)
            v = bm.verts.new((x, y, height))
            bottom_verts.append(v)

        # Create top vertices (at z = height + thickness)
        thickness = self.config.design_params['shelf_thickness']
        top_verts = []
        for x, y in outline:
            v = bm.verts.new((x, y, height + thickness))
            top_verts.append(v)

        # Ensure lookup table is built
        bm.verts.ensure_lookup_table()

        # Create bottom face
        bottom_face = bm.faces.new(bottom_verts)
        bottom_face.normal_update()

        # Create top face
        top_face = bm.faces.new(reversed(top_verts))
        top_face.normal_update()

        # Create side faces
        for i in range(num_points):
            next_i = (i + 1) % num_points
            # Create quad face for each edge
            face = bm.faces.new([
                bottom_verts[i],
                bottom_verts[next_i],
                top_verts[next_i],
                top_verts[i]
            ])
            face.normal_update()

        # Write bmesh to mesh
        bm.to_mesh(mesh)
        bm.free()

        # Update mesh
        mesh.update()

        return obj

    def create_all_shelves(self) -> List['bpy.types.Object']:
        """
        Create all shelf meshes from configuration.

        Returns:
            List of Blender objects
        """
        footprints = self.generator.generate_all_footprints(num_points=100)
        objects = []

        for footprint in footprints:
            obj = self.create_shelf_mesh(footprint, footprint.shelf_data['height'])
            objects.append(obj)
            self.shelf_objects.append(obj)

        return objects

    def create_pantry_room(self) -> List['bpy.types.Object']:
        """
        Create the pantry room walls for context.

        Returns:
            List of wall objects
        """
        pantry = self.config.pantry
        width = pantry['width']
        depth = pantry['depth']
        height = pantry['height']

        walls = []

        # Create walls as thin boxes
        wall_thickness = 0.5

        # East wall (x=0)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(-wall_thickness/2, depth/2, height/2)
        )
        east_wall = bpy.context.active_object
        east_wall.scale = (wall_thickness, depth, height)
        east_wall.name = "wall_east"
        walls.append(east_wall)

        # West wall (x=width)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(width + wall_thickness/2, depth/2, height/2)
        )
        west_wall = bpy.context.active_object
        west_wall.scale = (wall_thickness, depth, height)
        west_wall.name = "wall_west"
        walls.append(west_wall)

        # South wall (y=depth)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(width/2, depth + wall_thickness/2, height/2)
        )
        south_wall = bpy.context.active_object
        south_wall.scale = (width, wall_thickness, height)
        south_wall.name = "wall_south"
        walls.append(south_wall)

        # North wall (y=0) - with door opening
        # Support both old and new door clearance format
        if 'door_clearance_east' in pantry:
            east_clearance = pantry['door_clearance_east']
            west_clearance = pantry['door_clearance_west']
        else:
            east_clearance = pantry.get('door_clearance_sides', 4.5)
            west_clearance = pantry.get('door_clearance_sides', 4.5)

        # East side of door (left when looking in)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(east_clearance/2, -wall_thickness/2, height/2)
        )
        north_east = bpy.context.active_object
        north_east.scale = (east_clearance, wall_thickness, height)
        north_east.name = "wall_north_east"
        walls.append(north_east)

        # West side of door (right when looking in)
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(width - west_clearance/2, -wall_thickness/2, height/2)
        )
        north_west = bpy.context.active_object
        north_west.scale = (west_clearance, wall_thickness, height)
        north_west.name = "wall_north_west"
        walls.append(north_west)

        # Apply wall material
        wall_material = self.create_wall_material()
        for wall in walls:
            if wall.data.materials:
                wall.data.materials[0] = wall_material
            else:
                wall.data.materials.append(wall_material)

        return walls

    def create_shelf_material(self) -> 'bpy.types.Material':
        """
        Create a basic material for shelves (Baltic birch placeholder).

        Returns:
            Blender material
        """
        mat = bpy.data.materials.new(name="ShelfMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Create principled BSDF
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)

        # Wood color (light birch)
        bsdf.inputs['Base Color'].default_value = (0.85, 0.75, 0.55, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Specular IOR Level'].default_value = 0.3

        # Output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)

        # Link
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        return mat

    def create_wall_material(self) -> 'bpy.types.Material':
        """
        Create material for pantry walls.

        Returns:
            Blender material
        """
        mat = bpy.data.materials.new(name="WallMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()

        # Create principled BSDF
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)

        # Light gray drywall color
        bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.85, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.8

        # Output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)

        # Link
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        return mat

    def apply_materials(self):
        """Apply materials to all shelf objects."""
        shelf_material = self.create_shelf_material()

        for obj in self.shelf_objects:
            if obj.data.materials:
                obj.data.materials[0] = shelf_material
            else:
                obj.data.materials.append(shelf_material)

    def setup_lighting(self, light_position: Optional[Tuple[float, float, float]] = None):
        """
        Set up lighting for the scene.

        Args:
            light_position: (x, y, z) position for light. If None, places at center.
        """
        # Remove existing lights
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Default light position: center of pantry, near top
        if light_position is None:
            light_position = (
                self.config.pantry['width'] / 2,
                self.config.pantry['depth'] / 2,
                self.config.pantry['height'] * 0.8
            )

        # Create point light
        light_data = bpy.data.lights.new(name="PantryLight", type='POINT')
        light_data.energy = 2000
        light_data.shadow_soft_size = 2.0

        light_object = bpy.data.objects.new(name="PantryLight", object_data=light_data)
        light_object.location = light_position

        self.scene.collection.objects.link(light_object)

    def setup_camera(self, location: Tuple[float, float, float],
                    rotation: Tuple[float, float, float],
                    lens: float = 35.0,
                    sensor_width: float = 36.0,
                    fisheye: bool = False,
                    fov: Optional[float] = None) -> 'bpy.types.Object':
        """
        Set up camera with specified parameters.

        Args:
            location: (x, y, z) camera position
            rotation: (x, y, z) rotation in degrees
            lens: Focal length in mm (ignored if fisheye=True)
            sensor_width: Sensor width in mm
            fisheye: Use fisheye/panoramic lens
            fov: Field of view in degrees (for fisheye, default 180)

        Returns:
            Camera object
        """
        # Remove existing cameras
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Create camera
        camera_data = bpy.data.cameras.new(name="PantryCamera")

        if fisheye:
            # Use perspective camera with very wide angle to simulate fisheye
            camera_data.type = 'PERSP'
            # Calculate lens for desired FOV using sensor width
            # FOV formula: fov = 2 * atan(sensor / (2 * focal_length))
            # Rearranged: focal_length = sensor / (2 * tan(fov/2))
            fov_value = fov if fov is not None else 170.0
            fov_rad = np.radians(fov_value)
            camera_data.lens = sensor_width / (2 * np.tan(fov_rad / 2))
            camera_data.sensor_width = sensor_width
        else:
            camera_data.type = 'PERSP'
            camera_data.lens = lens
            camera_data.sensor_width = sensor_width

        camera_object = bpy.data.objects.new("PantryCamera", camera_data)
        camera_object.location = location

        # Convert degrees to radians and apply rotation
        camera_object.rotation_euler = Euler([
            np.radians(rotation[0]),
            np.radians(rotation[1]),
            np.radians(rotation[2])
        ], 'XYZ')

        self.scene.collection.objects.link(camera_object)
        self.scene.camera = camera_object

        return camera_object

    def point_camera_at(self, camera: 'bpy.types.Object',
                       target: Tuple[float, float, float]):
        """
        Point camera at a specific target location.

        Args:
            camera: Camera object
            target: (x, y, z) target position
        """
        # Create temporary empty as target
        bpy.ops.object.empty_add(location=target)
        target_obj = bpy.context.active_object

        # Add track-to constraint
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        # Apply constraint
        bpy.context.view_layer.update()

        # Remove constraint and empty (we just wanted the rotation)
        camera.constraints.remove(constraint)
        bpy.data.objects.remove(target_obj, do_unlink=True)

    def render(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080),
              samples: int = 128):
        """
        Render the scene to an image file.

        Args:
            output_path: Path to save rendered image
            resolution: (width, height) in pixels
            samples: Number of samples for rendering quality
        """
        # Set render settings
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.samples = samples
        self.scene.render.resolution_x = resolution[0]
        self.scene.render.resolution_y = resolution[1]
        self.scene.render.resolution_percentage = 100

        # Set output format
        self.scene.render.image_settings.file_format = 'PNG'
        self.scene.render.filepath = str(output_path)

        # Render
        bpy.ops.render.render(write_still=True)

    def create_preset_camera_view(self, preset: str) -> Tuple[Tuple[float, float, float],
                                                               Tuple[float, float, float]]:
        """
        Get camera position and rotation for preset views.

        Args:
            preset: View preset name ('doorway', 'front', 'corner', 'inside', 'top')

        Returns:
            Tuple of (location, rotation)
        """
        width = self.config.pantry['width']
        depth = self.config.pantry['depth']
        height = self.config.pantry['height']
        door_clearance = self.config.pantry['door_clearance_sides']

        center_x = width / 2
        center_y = depth / 2
        center_z = height / 2

        presets = {
            'doorway': (
                # Right side of door (West side when looking in), 6 feet high
                # Position slightly outside and to the right
                (width - door_clearance - 2, -6, 72),
                (90, 0, 0)  # Looking straight in
            ),
            'front': (
                (center_x, -depth * 0.8, center_z),
                (90, 0, 0)
            ),
            'corner': (
                (-width * 0.6, -depth * 0.6, height * 0.6),
                (65, 0, -45)
            ),
            'inside': (
                (center_x, depth * 0.3, height * 0.4),
                (80, 0, 180)
            ),
            'top': (
                (center_x, center_y, height * 1.5),
                (0, 0, 0)
            ),
            'southeast': (
                (width * 1.3, depth * 1.3, height * 0.5),
                (75, 0, -135)
            ),
            'eye_level': (
                (center_x, -depth * 0.5, 66),  # 66" is ~5'6" eye level
                (90, 0, 0)
            )
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        return presets[preset]
