# Procedural Pantry Shelf Designer

A Python-based tool for designing and rendering procedurally-generated pantry shelves with sinusoidal edge perturbations and tangent circle corners.

## Project Status: ✅ FULLY FUNCTIONAL

This project generates custom pantry shelving with organic, sinusoidal edges. Each shelf's depth varies according to a sine wave with a random phase offset, creating unique, flowing shapes. The system produces:
- **Exact SVG cutting templates** for all 25 shelf pieces
- **Photorealistic 3D renderings** using Blender + Cycles
- **Life-size plywood layout PDFs** at 150 DPI for printing

---

## Current Pantry Configuration

### Pantry Specifications
- **Dimensions**: 48" wide × 49" deep × 105" tall (8' 9")
- **Layout**: Door on North wall, shelves on East (left), South (back), and West (right) walls
- **Door clearance**: 6" East side, 4" West side

### Shelf Design Parameters (CURRENT VALUES)
- **Base depth**:
  - **Left (East) wall**: 7"
  - **Back (South) wall**: 19"
  - **Right (West) wall**: 5" ⭐ (updated from 4")
- **Sinusoid perturbation**: **1" amplitude** ⭐ (updated from 2"), 24" period
- **Depth formula**: `depth(position) = base_depth + 1" × sin(2π × position / 24" + random_offset)`
- **Thickness**: 1" (extruded solid)
- **Corner radius**: 3" (interior corners at back wall)
- **Total shelves**: **25 pieces** across **17 unique height levels**

### Corner Treatment
- **East/West × South corners**: Smooth 3" radius arcs tangent to both sinusoidal edges
- **East/West × North corners**: Horizontal cut lines (door clearance)
- **Left/Right back corners**: Mathematically precise tangent circles solving for dual-sinusoid tangency

---

## Shelf Inventory

### Main Shelves (Full L-shaped: Left + Back + Right)
Complete shelves spanning all three walls:
- **19"** from floor: `shelf_L19.svg`, `shelf_B19.svg`, `shelf_R19.svg`
- **39"** from floor: `shelf_L39.svg`, `shelf_B39.svg`, `shelf_R39.svg`
- **59"** from floor: `shelf_L59.svg`, `shelf_B59.svg`, `shelf_R59.svg`
- **79"** from floor: `shelf_L79.svg`, `shelf_B79.svg`, `shelf_R79.svg`

### Left Intermediate Shelves (East wall only, 7" depth)
- **9"**, **29"**, **49"**, **69"** from floor: `shelf_L9.svg`, `shelf_L29.svg`, `shelf_L49.svg`, `shelf_L69.svg`

### Right Intermediate Shelves (West wall only, 5" depth)
- **5"**, **13"**, **26"**, **33"**, **46"**, **53"**, **66"**, **73"**, **86"** from floor
- 9 pieces: `shelf_R5.svg` through `shelf_R86.svg`

**Total**: 25 shelf pieces (12 main + 4 left + 9 right)

---

## Complete Workflow (CURRENT - January 2026)

### 1. Generate Exact Geometry & SVG Files
```bash
python scripts/extract_and_export_geometry.py
```

**Outputs:**
- `output/shelf_L*.svg`, `output/shelf_B*.svg`, `output/shelf_R*.svg` - 25 exact SVG files
- `output/exact_cutting_templates.pdf` - Complete PDF with:
  - Height-level visualization pages (17 pages)
  - Life-size plywood layout pages (5 sheets, 150 DPI)

**What it does:**
- Loads config from `configs/pantry_0002.json`
- Solves tangent circle geometry for all back corners
- Handles special case: B59 right arc direction (auto-detected)
- Generates exact polygon coordinates for each shelf
- Packs all pieces onto 5 plywood sheets (96" × 48")

### 2. Generate Photorealistic 3D Renderings
```bash
blender --background --python scripts/render_from_svg.py
```

**Outputs:**
- `output/renders/pantry_center_view.png` - Center doorway perspective
- `output/renders/pantry_angled_view.png` - Side angle perspective

**What it does:**
- Loads exact geometry from all 25 SVG files
- Creates 3D meshes by extruding polygons to 1" thickness
- Applies Baltic birch plywood materials (edge plies + wood grain)
- Sets up realistic lighting (point light + fill light)
- Renders at 1920×1080, 128 samples, Cycles engine
- Takes ~12-15 minutes per view

### 3. Configuration Management
Configuration is stored in `configs/pantry_NNNN.json` files:

```json
{
  "config_version": "0002",
  "pantry": {
    "width": 48.0,
    "depth": 49.0,
    "height": 105.0,
    "door_clearance_east": 6.0,
    "door_clearance_west": 4.0
  },
  "design_params": {
    "sinusoid_period": 24.0,
    "sinusoid_amplitude": 1.0,
    "shelf_base_depth_east": 7.0,
    "shelf_base_depth_south": 19.0,
    "shelf_base_depth_west": 5.0,
    "shelf_thickness": 1.0,
    "interior_corner_radius": 3.0,
    "door_corner_radius": 3.0
  },
  "shelves": [...]
}
```

**To modify parameters:**
1. Edit config file directly, OR
2. Update defaults in `src/config.py` (`_create_default_config()`)
3. Regenerate: `python scripts/extract_and_export_geometry.py`

---

## Kitchen Corner Shelves (Separate Project)

A secondary project for kitchen corner shelving using similar geometry.

### Specifications
- **Space**: 37" along wall × 12" depth (cabinet corner)
- **Shelves**: 3 shelves at 12", 24", 36" from ceiling
- **Thickness**: 1"
- **Sinusoid**: 1" amplitude, 24" period
- **Corner radius**: 4" (bottom-right corner)

### Generate Kitchen Corner Shelves
```bash
python scripts/generate_kitchen_corner_shelves.py
```

**Outputs:**
- `output/kitchen_corner/kitchen_corner_shelf_12in.svg` (and 24in, 36in)
- `output/kitchen_corner/kitchen_corner_shelf_12in.png` (visualizations)
- `output/kitchen_corner/kitchen_corner_cutting_templates.pdf`

**Geometry:**
- Cabinet corner at top-left (90° angle, no radius)
- Right edge: Sinusoidal (varying around 12" depth)
- Bottom edge: Straight
- Bottom-right corner: 4" radius arc tangent to both edges
- Uses controlled phase offsets for stable arc solutions

---

## Project Structure

```
pantry/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── configs/                       # JSON configuration files
│   ├── pantry_0000.json          # Initial config
│   ├── pantry_0002.json          # Current active config
│   └── pantry_NNNN.json          # Additional variants
├── src/                          # Core library modules
│   ├── config.py                 # Configuration management
│   ├── geometry.py               # Tangent circle solver, sinusoids
│   ├── blender_renderer.py       # Blender material/lighting utilities
│   └── utils.py                  # Coordinate conversions
├── scripts/                      # Executable scripts
│   ├── extract_and_export_geometry.py  # ⭐ MAIN: Generate exact SVGs & PDF
│   ├── render_from_svg.py              # ⭐ MAIN: Blender rendering from SVGs
│   ├── generate_kitchen_corner_shelves.py  # Kitchen corner variant
│   ├── generate_config.py        # Config file generator
│   ├── test_gp_curve.py          # GP/kriging curve tests
│   ├── test_tangent_circles.py   # Tangent circle geometry tests
│   ├── verify_geometry.py        # Geometry validation
│   └── old/                      # Deprecated scripts (pre-SVG workflow)
└── output/                       # Generated files
    ├── shelf_*.svg               # 25 exact SVG cutting templates
    ├── exact_cutting_templates.pdf  # Complete PDF guide
    ├── renders/                  # Blender 3D renderings
    │   ├── pantry_center_view.png
    │   └── pantry_angled_view.png
    └── kitchen_corner/           # Kitchen corner shelf outputs
        ├── kitchen_corner_shelf_*.svg
        └── kitchen_corner_cutting_templates.pdf
```

---

## Technical Details

### Coordinate System
- **Origin**: Northwest corner (door side, left when looking in)
- **X-axis**: West to East (left to right) [0 to 48"]
- **Y-axis**: North to South (door to back) [0 to 49"]
- **Z-axis**: Floor to ceiling (up) [0 to 105"]

### Wall Nomenclature
- **East (E)** = Left wall (x = 0)
- **South (S)** = Back wall (y = 49)
- **West (W)** = Right wall (x = 48)
- **North (N)** = Door wall (y = 0)

### Tangent Circle Solver (`geometry.py`)
Solves for circles tangent to two sinusoidal curves using:
- Scipy optimization (`minimize`)
- Gradient constraints for tangency
- Interior/exterior validation
- **Special handling**: Arc direction varies by level; B59 auto-detection

### SVG Export
- Exact coordinates from solver (no approximation)
- Polygon format: counter-clockwise vertex ordering
- Viewbox scaled to shelf dimensions
- Compatible with laser cutters, CNC routers

### Blender Materials
- **Edge texture**: 9 alternating plies (Baltic birch)
- **Top/bottom**: Procedural wood grain (Musgrave texture)
- **Lighting**: Point light + fill light for realistic shadows

---

## Key Achievements ✅

1. ✅ **Exact tangent circle geometry** - Dual-sinusoid corner solver working perfectly
2. ✅ **Arc direction auto-detection** - Handles B59 anomaly (fixed Jan 2026)
3. ✅ **25-piece full pantry** - All shelves generated with exact geometry
4. ✅ **SVG-based workflow** - Single source of truth for geometry
5. ✅ **Photorealistic rendering** - Blender + Cycles with realistic materials
6. ✅ **Plywood packing optimization** - 5 sheets for all 25 pieces
7. ✅ **Kitchen corner variant** - Separate project using similar techniques

---

## Experimental Features (Not Production-Ready)

### Gaussian Process (GP/Kriging) Curves
**Status**: ⚠️ NOT WORKING - Development paused

**Scripts:**
- `scripts/generate_kitchen_corner_shelves_kriging.py` - GP-based kitchen shelves
- `scripts/test_gp_curve.py` - GP curve testing and visualization

**Concept:**
- Replace sinusoids with Gaussian process random curves
- Kriging with hard observations at endpoints
- Derivative constraints for slope matching
- Smoothness regularization (standard deviation ≤ 1")

**Issues:**
- Intersection solving more complex than sinusoids
- Arc tangency calculation needs refinement
- Current implementation produces artifacts

**Files preserved for future development.**

---

## Dependencies

### Python Packages
```bash
pip install numpy scipy matplotlib
```

- **numpy**: Mathematical operations, sinusoid calculations
- **scipy**: Optimization for tangent circle solving
- **matplotlib**: 2D visualizations, PDF generation

### Blender
- **Version**: Blender 4.5+ (with Python 3.11)
- **Engine**: Cycles ray tracing
- **Required**: `bpy` (Blender Python API, included with Blender)

---

## Troubleshooting

### Issue: Arc points in wrong order (self-intersecting path)
**Symptom**: Back shelf SVG shows artifact, polygon crosses itself
**Cause**: Arc generated in opposite direction (point1→point2 vs point2→point1)
**Solution**: Fixed in `extract_and_export_geometry.py` line 432 - auto-detects arc direction
**Affected**: Only shelf B59 (level 2), now handled automatically

### Issue: Blender renders are all black
**Symptom**: Rendered images are pure black
**Cause**: Missing materials or incorrect lighting setup
**Solution**: Use `render_from_svg.py` (not old render scripts) - includes proper materials

### Issue: SVG files don't load in laser cutter software
**Symptom**: Software reports invalid path or won't import
**Cause**: Path not closed or self-intersecting polygon
**Solution**: Verify with `scripts/verify_geometry.py`, check polygon vertex order

---

## Known Limitations

1. **Fixed plywood size**: Packing assumes 96" × 48" sheets
2. **No kerf compensation**: SVG dimensions don't account for laser/saw blade width
3. **Single material**: Only Baltic birch texture implemented
4. **No structural analysis**: Load capacity, sagging not calculated
5. **Manual assembly**: No automated joinery or fastener placement

---

## Future Enhancements

### Short-term
- [ ] Kerf compensation parameter in config
- [ ] Alternative plywood sizes (48" × 96", 60" × 120")
- [ ] Dimension annotations on SVG files
- [ ] Assembly instructions PDF

### Long-term
- [ ] Web-based 3D viewer (Three.js)
- [ ] Structural FEA analysis (sagging simulation)
- [ ] CNC G-code generation
- [ ] Cost estimation with material pricing
- [ ] Joinery design (dados, rabbets)
- [ ] Multiple material options

### Research
- [ ] Fix GP/kriging curve implementation
- [ ] Multi-harmonic sinusoids (Fourier series)
- [ ] Bézier curve edges
- [ ] Topology optimization for weight reduction

---

## Recent Updates

### January 2026
- ✅ Fixed B59 back shelf arc direction anomaly
- ✅ Updated right wall depth: 4" → 5"
- ✅ Moved deprecated scripts to `scripts/old/`
- ✅ Consolidated workflow to SVG-based rendering
- ✅ Updated README with complete current documentation

### December 2025
- ✅ Implemented exact SVG export workflow
- ✅ Created kitchen corner shelf variant
- ✅ Added GP/kriging curve experimentation (incomplete)
- ✅ Optimized plywood packing (25 pieces → 5 sheets)

---

## Credits

**Design & Implementation**: Procedurally generated using Claude (Anthropic)
**Geometric Solver**: Custom tangent circle algorithm with scipy optimization
**Rendering**: Blender 4.5 + Cycles ray tracing engine
**Material Textures**: Procedural Baltic birch (custom shader nodes)

---

## License

This project is for personal use. The generated shelf designs are unique to your pantry dimensions.

---

**Last updated**: January 2, 2026
**Config version**: pantry_0002.json
**Status**: Production-ready ✅
