# Pantry Shelf Designer

Procedural custom pantry shelf system for a 48" × 49" × 105" walk-in pantry.
Shelves have sinusoidal edges with tangent-circle corners. Full pipeline: geometry
computation → SVG/DXF export → interactive nesting UI → plywood cut sheets.

---

## Quick Start

```bash
# 1. Generate all shelf geometry + SVGs/DXFs
python scripts/generate_from_patterns.py

# 2. Export geometry for nesting UI
python scripts/export_shelf_geometry.py        # → nesting_geometry.json

# 3. Open browser-based nesting tool
open nesting_ui.html                            # manually arrange shelves on sheets

# 4. Generate DXF cut sheets from nesting layout
python generate_dxf_from_layout.py             # reads nesting_layout.json
```

---

## Project Structure

```
pantry/
│
├── README.md                          # This file
├── CONSTRUCTION_PATTERNS.md           # Geometric primitive reference
├── requirements.txt
│
├── configs/
│   ├── shelf_level_patterns.json      # PRIMARY: all 25 shelves as construction sequences
│   ├── stud_positions.json            # Wall stud positions + bracket BOM (83 total)
│   ├── pantry_0002.json               # Current active pantry config (legacy)
│   └── pantry_000{0,1,3,4,5}.json    # Historical config variants (legacy)
│
├── src/                               # Core library
│   ├── geometry.py                    # Geometric solvers
│   │   ├── sinusoid_depth()           # Sine-wave depth at a position
│   │   ├── sinusoid_depth_derivative()
│   │   ├── generate_sinusoid_points() # Sample sinusoid into point array
│   │   ├── solve_interior_corner()    # Arc tangent to two sinusoids (scipy)
│   │   ├── solve_door_corner()        # Door-clearance notch arc
│   │   ├── CornerSolver               # Class: iterative tangent-circle solver
│   │   ├── solve_tangent_circle_two_sinusoids()       # Primary solver (scipy)
│   │   ├── solve_tangent_circle_two_sinusoids_newton() # Newton's method solver
│   │   ├── tangent_circle_two_sinusoids_offset_intersection()
│   │   ├── generate_circle_arc()      # Discretize an arc
│   │   ├── wall_to_pantry_coords()
│   │   ├── pantry_to_wall_coords()
│   │   ├── create_shelf_outline()     # Build full polygon for one shelf
│   │   ├── is_point_in_interior()
│   │   └── generate_interior_mask()
│   │
│   ├── config.py
│   │   └── ShelfConfig                # Dataclass: pantry dims + design params
│   │
│   ├── shelf_generator.py
│   │   ├── ShelfFootprint             # Dataclass: polygon + metadata
│   │   └── ShelfGenerator             # Class: generates all shelves from config
│   │
│   ├── pdf_generator.py               # PDF layout page rendering
│   └── blender_renderer.py            # Blender material/lighting helpers
│
├── scripts/                           # Runnable entry points
│   │
│   ├── generate_from_patterns.py      # ★ MAIN GENERATOR (construction-based)
│   │   ├── ReferenceResolver          # Resolves JSON cross-references
│   │   ├── solve_door_smoothing_alt9()# Rotation-based tangent arc solver
│   │   ├── BaseGeometrySolver         # Computes sinusoids, intersections, arcs per level
│   │   ├── ConstructionRenderer       # Walks construction sequences → polygon pts
│   │   ├── GeometryExporter           # SVG / DXF / PDF export
│   │   ├── apply_pipe_cutout()        # Circular cutout in polygon
│   │   ├── apply_outlet_cutout()      # Rectangular outlet notch
│   │   └── main()
│   │
│   ├── extract_and_export_geometry.py # Legacy generator (procedural, not pattern-based)
│   │   ├── find_brackets()            # Root-bracket finder
│   │   ├── bisect_root()              # Bisection root solver
│   │   ├── solve_tangent_circle_horizontal_sinusoid()
│   │   ├── solve_door_smoothing_radius()
│   │   ├── solve_door_smoothing_fixed_radius()
│   │   ├── generate_intermediate_shelf()
│   │   ├── extract_exact_shelf_geometries()
│   │   ├── export_polygon_to_svg()
│   │   ├── export_polygon_to_dxf()
│   │   ├── export_all_shelves_to_combined_dxf()
│   │   ├── simple_2d_pack()           # Greedy strip packer
│   │   └── main()
│   │
│   ├── generate_nested_layouts.py     # Shared nesting data model + DXF export
│   │   ├── ShelfGroup                 # Shapely polygon + bracket geometry for one shelf
│   │   ├── ShelfPair                  # Two shelves packed together
│   │   ├── Sheet                      # One 96"×48" plywood sheet
│   │   ├── export_sheet_to_dxf()      # Write Sheet → DXF via ezdxf
│   │   ├── load_shelves()             # Load all per-shelf DXFs → ShelfGroup list
│   │   ├── standardize_shelf()        # Mirror X, flip to nesting orientation
│   │   ├── create_tight_pair()        # Pack two shelves back-to-back
│   │   └── main()                     # Automated nesting (legacy)
│   │
│   ├── export_shelf_geometry.py       # Export geometry → nesting_geometry.json
│   │   ├── bbox_center()
│   │   ├── transform_pts()            # Re-center at (0,0), flip Y
│   │   ├── extract_linestring_coords()
│   │   └── main()
│   │
│   ├── generate_shelves_with_brackets.py  # Visualize shelves with bracket locations
│   │
│   ├── render_from_svg.py             # Blender rendering from SVG files
│   ├── render_blender.py              # Blender scene setup utilities
│   │
│   ├── generate_kitchen_corner_shelves.py          # Kitchen corner shelf variant
│   ├── generate_kitchen_corner_shelves_kriging.py  # GP/kriging variant (experimental)
│   │
│   ├── test_gp_curve.py               # Gaussian Process curve tests
│   ├── test_tangent_circles.py        # Tangent circle solver tests
│   ├── test_nesting.py                # Nesting algorithm tests
│   ├── test_bounds.py                 # Bounds validation tests
│   ├── verify_geometry.py             # Geometry sanity checks
│   ├── sin_sin_circle_test.py         # Sinusoid-circle intersection tests
│   ├── debug_overview.py              # Visual debug overview
│   ├── plot_back_left_level19_debug.py
│   │
│   ├── alt_{4,5,6,7,8,9}.py          # Door-smoothing solver iterations (alt_9 is current)
│   ├── alt_intersector{,_2,_3}.py     # Intersection solver iterations
│   ├── generate_config.py             # Config file generator
│   │
│   └── old/                           # Deprecated pre-SVG scripts
│       ├── generate_shelves.py
│       ├── generate_pdfs.py
│       ├── generate_cutting_pdf.py
│       ├── generate_final_templates.py
│       ├── render_2d.py / render_3d.py / render_complete_pantry.py / render_full_pantry.py
│       ├── apply_manual_layout_from_svg.py
│       └── test_{dxf,pair,svg_matching,svg_parser,svg_scale}.py
│
├── nesting_ui.html                    # ★ Browser-based interactive nesting tool
│   # (single-file HTML/CSS/JS, no build step)
│   # Features: pan/zoom SVG, drag shelves, R=rotate 90°, C=check gaps,
│   #           E=export JSON, snap-to-edge buttons, bracket BOM display
│   # Reads:  nesting_geometry.json
│   # Writes: nesting_layout.json  (via Export button)
│
├── generate_dxf_from_layout.py        # ★ Convert nesting JSON → per-sheet DXF
│   ├── rot_pt()                       # 90° CW rotation (SVG → CCW in DXF)
│   ├── transform_pts()                # Rotate + Y-flip + translate
│   ├── add_polyline()                 # Add lwpolyline to DXF modelspace
│   ├── export_sheet_dxf()             # One sheet → DXF file
│   └── main()
│
├── nest_from_inkscape.py              # Inkscape SVG → polished DXF (alternative workflow)
│   ├── parse_matrix()                 # Parse SVG matrix() transform
│   ├── build_nesting_affine()         # SVG matrix → Shapely affine coefficients
│   ├── apply_nesting_transform()      # Transform ShelfGroup geometry in-place
│   ├── parse_svg_shelves()            # Extract shelf positions from Inkscape SVG
│   ├── _sep_direction()               # Push direction (min-penetration axis)
│   ├── build_adjacency()              # Proximity graph of shelves
│   ├── bfs_order()                    # BFS traversal from anchor
│   ├── _clamp_to_sheet()              # Push shelf back inside sheet bounds
│   ├── polish_layout()                # Force-averaging gap enforcement (0.5" min)
│   ├── validate_layout()              # Report gap + bounds violations
│   └── main()
│
├── debug_sheet_order.py               # Debug: print sheet shelf ordering
├── debug_svg_coords.py                # Debug: print SVG coordinate values
├── dump_svg.py                        # Debug: dump SVG element tree
│
├── nesting_geometry.json              # Generated: shelf polygons centered at (0,0)
├── nesting_layout.json                # Generated: shelf placement per sheet (from UI export)
├── nesting_layout_old.json            # Backup of previous layout
│
├── Inkscape/                          # Manual Inkscape nesting layouts (SVG source)
│   ├── Panty_Sheet_1.svg
│   └── Panty_Sheet_2.svg
│
└── output/                            # All generated outputs
    ├── shelf_L{9,19,29,39,49,59,69,79}.{svg,dxf}   # Left wall shelves
    ├── shelf_B{19,39,59,79}.{svg,dxf}               # Back wall shelves
    ├── shelf_R{5,13,19,26,33,39,46,53,59,66,73,79,86}.{svg,dxf}  # Right wall shelves
    ├── shelf_*_exact.dxf              # DXF variants with exact suffix (legacy extract script)
    ├── pantry_layout.pdf              # Assembly guide (17 pages, color-coded by wall)
    ├── sheet_{1,2}_nesting.dxf        # ★ Final cut sheets (from nesting workflow)
    ├── renders/                       # Blender renders + SVG-based previews
    └── kitchen_corner_kriging/        # Kitchen corner GP experiment outputs
```

---

## Pantry Specifications

| Parameter | Value |
|---|---|
| Dimensions | 48" wide × 49" deep × 105" tall |
| Total shelves | 25 pieces across 17 height levels |
| Left wall depth | 7" base + 1" sine amplitude |
| Back wall depth | 19" base + 1" sine amplitude |
| Right wall depth | 5" base + 1" sine amplitude |
| Sine period | 24" |
| Corner radius | 3" (interior back corners) |
| Plywood sheets | 2 sheets @ 96" × 48" |

### Shelf Inventory

| Set | Heights | Count |
|---|---|---|
| Left (L) | 9, 19, 29, 39, 49, 59, 69, 79" | 8 |
| Back (B) | 19, 39, 59, 79" | 4 |
| Right (R) | 5, 13, 19, 26, 33, 39, 46, 53, 59, 66, 73, 79, 86" | 13 |

### Bracket BOM

| Tongue length | Count | Used for |
|---|---|---|
| 4" | 47 | Right wall (3 studs × 10 shelves) + back shelf corner brackets |
| 6" | 16 | Left wall (2 studs × 8 shelves) |
| 10" | 20 | Back wall (3 studs × 4 shelves) + back shelf side brackets |
| **Total** | **83** | |

---

## Coordinate Systems

Two coordinate systems are in use. Be careful at boundaries.

| Context | X origin | Y origin | Y direction |
|---|---|---|---|
| Pantry / DXF | NW corner (door-left) | same | South = +Y (door→back) |
| SVG / nesting UI | sheet top-left | same | Down = +Y |

**Pantry walls:**
- East = left wall (x = 0)
- West = right wall (x = 48)
- South = back wall (y = 49)
- North = door wall (y = 0)

---

## Key Algorithms

**Tangent circle at back corners** (`geometry.py`):
Finds a circle tangent to two sinusoidal curves simultaneously using scipy `minimize`
with gradient constraints. Falls back to Newton's method for degenerate cases.

**Door smoothing arc** (`alt_9.py` / `generate_from_patterns.py`):
Rotation-based solver — rotates the sinusoid until it's horizontal, solves for the
tangent arc analytically, then rotates back.

**Gap polishing** (`nest_from_inkscape.py:polish_layout`):
Freeze shelves with structural overlaps (area > 1 sq.in.). Detect "squeezed" free
shelves where opposing forces cancel (net < 20% of total). Apply symmetric force
vectors to remaining free shelves, capped at 0.75"/pass, for up to 200 passes.

---

## Dependencies

```bash
pip install numpy scipy matplotlib shapely ezdxf
```

Blender 4.5+ required only for photorealistic renders.

---

**Last updated**: March 2026
