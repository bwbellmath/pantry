# Procedural Pantry Shelf Designer

A Python-based tool for designing and rendering procedurally-generated pantry shelves with sinusoidal edge perturbations.

## Project Overview

This project generates custom pantry shelving with organic, sinusoidal edges. Each shelf's depth varies according to a sine wave with a random phase offset, creating unique, flowing shapes. The system produces both technical 2D cutting templates (PDF) and photorealistic 3D renderings using Blender.

### Pantry Specifications
- **Dimensions**: 48" wide × 49" deep × 105" tall (8' 9")
- **Layout**: Door on North wall, shelves on East (left), South (back), and West (right) walls
- **Door clearance**: 6" East side, 4" West side

### Shelf Design Parameters
- **Base depth**: 7" from wall (East/left), 4" from wall (West/right)
- **Sinusoid perturbation**: 2" amplitude, 24" (2 feet) period
- **Depth formula**: `depth(position) = base_depth + 2" × sin(2π × position / 24" + random_offset)`
- **Thickness**: 1" (extruded solid)
- **Corner radius**: 3" chamfer (interior corners)
- **Total shelves**: 25 pieces across 17 height levels

### Corner Treatment
- **East/West × South corners**: Material ADDED to create smooth 2" radius transition
- **East/West × North corners**: Material REMOVED to create 2" radius (door side)
- **South wall**: No radiusing at straight back wall

## Project Structure

```
pantry/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── configs/                       # JSON configuration files
│   ├── pantry_0000.json          # Initial baseline configuration
│   └── pantry_NNNN.json          # Sequentially numbered variants
├── src/
│   ├── __init__.py
│   ├── config.py                 # JSON config read/write/modify
│   ├── geometry.py               # Sinusoid calculations, chamfering math
│   ├── shelf_generator.py        # Generate shelf 2D footprints
│   ├── blender_renderer.py       # Blender 3D rendering engine
│   ├── pdf_generator.py          # Matplotlib 2D cutting templates
│   └── utils.py                  # Shared utilities
├── scripts/
│   ├── generate_config.py        # Create new config with random offsets
│   ├── render_3d.py              # Generate Blender renderings
│   ├── render_2d.py              # Generate PDF cutting templates
│   └── modify_config.py          # Procedurally modify existing configs
└── output/
    ├── renders/                   # 3D rendered images
    │   └── pantry_NNNN_view.png
    └── cutting_templates/         # 2D PDF drawings
        └── pantry_NNNN_shelf_L.pdf
```

## JSON Configuration Format

Each configuration file contains complete specifications for one shelf design variant:

```json
{
  "config_version": "0000",
  "pantry": {
    "width": 48.0,
    "depth": 54.0,
    "height": 96.0,
    "door_clearance_sides": 4.5
  },
  "design_params": {
    "sinusoid_period": 24.0,
    "sinusoid_amplitude": 2.0,
    "shelf_base_depth": 6.0,
    "shelf_thickness": 1.0,
    "corner_radius": 2.0
  },
  "shelves": [
    {
      "level": 0,
      "height": 24.0,
      "wall": "E",
      "extent_start": 0.0,
      "extent_end": 54.0,
      "sinusoid_offset": 1.2345,
      "corner_points_solved": [
        {"type": "north_door", "position": [0, 0], "radius_center": [x, y]},
        {"type": "south_transition", "position": [0, 54], "radius_center": [x, y]}
      ]
    },
    {
      "level": 0,
      "height": 24.0,
      "wall": "S",
      "extent_start": 0.0,
      "extent_end": 48.0,
      "sinusoid_offset": 2.3456,
      "corner_points_solved": []
    },
    {
      "level": 0,
      "height": 24.0,
      "wall": "W",
      "extent_start": 0.0,
      "extent_end": 54.0,
      "sinusoid_offset": 3.4567,
      "corner_points_solved": [
        {"type": "north_door", "position": [48, 0], "radius_center": [x, y]},
        {"type": "south_transition", "position": [48, 54], "radius_center": [x, y]}
      ]
    },
    // ... repeat for levels 1, 2, 3
  ]
}
```

## Modules and Dependencies

### Python Packages
- **numpy**: Mathematical operations, sinusoid calculations
- **matplotlib**: 2D technical drawings and PDF export
- **scipy**: Curve intersection solving, optimization
- **bpy** (Blender Python API): 3D modeling and rendering
- **json**: Configuration file handling (built-in)
- **pathlib**: File system operations (built-in)

### Core Modules

#### `config.py`
- Load/save JSON configurations
- Validate configuration data
- Generate sequential config numbers
- Helper functions for config manipulation

#### `geometry.py`
- Calculate sinusoidal edge points along walls
- Solve for chamfer arc intersections with sinusoids
- Compute corner radius centers and tangent points
- Generate 2D polygon footprints for each shelf section

#### `shelf_generator.py`
- Orchestrate shelf footprint generation
- Handle wall-to-wall transitions
- Apply corner radiusing (additive and subtractive)
- Generate complete 2D paths for each shelf piece

#### `blender_renderer.py`
- Import shelf geometries into Blender
- Extrude 2D footprints to 1" thick solids
- Apply Baltic birch plywood material:
  - Edge texture: 9 alternating dark/light stripes (plies)
  - Top/bottom: Wood grain procedural texture
- Set up scene lighting (central point light)
- Configure camera (position, rotation, fisheye lens option)
- Render and export images

#### `pdf_generator.py`
- Create technical cutting templates using matplotlib
- Draw shelf outlines with dimensions
- Add measurement annotations
- Include sinusoid parameters and level information
- Export as vector PDF (actual size for printing)

#### `utils.py`
- Coordinate system conversions
- Wall orientation helpers (E, S, W mappings)
- File naming conventions
- Logging and debugging utilities

## Scripts

### `generate_config.py`
Creates a new configuration file with random sinusoid offsets.

```bash
python scripts/generate_config.py --output configs/pantry_0000.json
python scripts/generate_config.py --base configs/pantry_0000.json --output configs/pantry_0001.json
```

### `render_3d.py`
Generates 3D renderings using Blender.

```bash
blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --camera-angle 45 --output output/renders/
# Alternative views:
blender --background --python scripts/render_3d.py -- configs/pantry_0000.json --fisheye --camera-height 48
```

### `render_2d.py`
Generates 2D PDF cutting templates.

```bash
python scripts/render_2d.py configs/pantry_0000.json --output output/cutting_templates/
```

### `modify_config.py`
Procedurally modify existing configurations (change offsets, depths, etc.).

```bash
python scripts/modify_config.py configs/pantry_0000.json --randomize-offsets --output configs/pantry_0001.json
python scripts/modify_config.py configs/pantry_0000.json --set-depth 8.0 --output configs/pantry_0002.json
```

## Workflow

1. **Generate initial configuration**:
   ```bash
   python scripts/generate_config.py --output configs/pantry_0000.json
   ```

2. **Review 2D layouts**:
   ```bash
   python scripts/render_2d.py configs/pantry_0000.json
   ```

3. **Generate 3D rendering**:
   ```bash
   blender --background --python scripts/render_3d.py -- configs/pantry_0000.json
   ```

4. **Iterate with new random offsets**:
   ```bash
   python scripts/generate_config.py --base configs/pantry_0000.json --output configs/pantry_0001.json
   python scripts/render_2d.py configs/pantry_0001.json
   blender --background --python scripts/render_3d.py -- configs/pantry_0001.json
   ```

5. **Fine-tune manually** by editing JSON, then re-render

## Coordinate System

- **Origin**: Northwest corner (door side, left when looking in)
- **X-axis**: West to East (left to right) [0 to 48"]
- **Y-axis**: North to South (door to back wall) [0 to 49"]
- **Z-axis**: Floor to ceiling (up) [0 to 105"]

### Wall Positions
- **North** (door wall): y = 0, x ∈ [0, 48]
- **East** (left wall): x = 0, y ∈ [0, 49]
- **South** (back wall): y = 49, x ∈ [0, 48]
- **West** (right wall): x = 48, y ∈ [0, 49]

## Shelf Level Heights

All heights are bottom-referenced (measured to the bottom of each 1" thick shelf).

### Main Shelves (Full L-shaped: Left + Back + Right walls)
Complete shelves that span all three walls (East, South, West):
- **19"** - shelf_L19, shelf_B19, shelf_R19
- **39"** - shelf_L39, shelf_B39, shelf_R39
- **59"** - shelf_L59, shelf_B59, shelf_R59
- **79"** - shelf_L79, shelf_B79, shelf_R79

### Left Intermediate Shelves (East wall only, 7" depth)
Additional shelves on the left/East wall only:
- **9"** - shelf_L9
- **29"** - shelf_L29
- **49"** - shelf_L49
- **69"** - shelf_L69

### Right Intermediate Shelves (West wall only, 4" depth)
Additional shelves on the right/West wall only:
- **5"** - shelf_R5
- **13"** - shelf_R13
- **26"** - shelf_R26
- **33"** - shelf_R33
- **46"** - shelf_R46
- **53"** - shelf_R53
- **66"** - shelf_R66
- **73"** - shelf_R73
- **86"** - shelf_R86

**Total**: 25 shelf pieces across 17 unique height levels

## Implementation Phases

### Phase 1: Core Geometry (MVP)
- [ ] JSON config structure and I/O
- [ ] Sinusoid edge calculation for straight sections
- [ ] Simple corner radiusing (circular arcs)
- [ ] Basic shelf footprint generation

### Phase 2: 2D Rendering
- [ ] Matplotlib technical drawings
- [ ] Dimension annotations
- [ ] PDF export with proper scaling
- [ ] Multiple views (one per shelf piece)

### Phase 3: 3D Rendering
- [ ] Blender geometry import
- [ ] 1" extrusion
- [ ] Basic materials and lighting
- [ ] Camera control system
- [ ] Render output

### Phase 4: Advanced Features
- [ ] Baltic birch texture (edge plies, grain)
- [ ] Advanced corner solving (tangent to sinusoids)
- [ ] Multiple camera presets
- [ ] Fisheye lens option
- [ ] Animation/turntable renders

### Phase 5: Utilities
- [ ] Configuration modification tools
- [ ] Batch rendering
- [ ] Comparison views
- [ ] Parameter exploration scripts

## Future Enhancements

- Interactive 3D viewer (web-based)
- Optimization for material usage
- Structural analysis (sagging, load capacity)
- CNC toolpath generation
- Cost estimation
- Alternative edge functions (splines, multiple harmonics)
