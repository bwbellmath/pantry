# Architecture & Roadmap

This document describes the proposed repository reorganization, a migration checklist
for the current pantry project, and stubs for future woodworking projects.

---

## Proposed Repository Structure

The current repo is a single flat project. The proposed structure treats this as a
**woodworking design monorepo** — one repo, many rooms/spaces, shared tooling.

```
woodwork/
│
├── README.md                          # Repo overview + quickstart
├── ARCHITECTURE.md                    # This file
├── requirements.txt                   # Shared Python deps (numpy, scipy, shapely, ezdxf)
│
├── lib/                               # Shared geometry + export library
│   ├── geometry.py                    # Sinusoid solver, tangent circles, arc generation
│   ├── construction.py                # Construction-sequence interpreter (from generate_from_patterns.py)
│   ├── nesting.py                     # ShelfGroup, Sheet, gap polishing (from generate_nested_layouts.py)
│   ├── export.py                      # SVG, DXF, PDF export helpers
│   └── config.py                      # Base config dataclasses
│
├── tools/                             # Standalone CLI tools (project-agnostic)
│   ├── nesting_ui.html                # Interactive browser nesting tool
│   ├── generate_dxf_from_layout.py    # nesting_layout.json → per-sheet DXF
│   ├── nest_from_inkscape.py          # Inkscape SVG → polished DXF
│   └── export_shelf_geometry.py       # Any project → nesting_geometry.json
│
└── projects/
    ├── pantry/                        # ★ CURRENT PROJECT (mostly done)
    │   ├── project.json               # Project manifest (see format below)
    │   ├── configs/
    │   │   ├── shelf_level_patterns.json
    │   │   └── stud_positions.json
    │   ├── scripts/
    │   │   └── generate.py            # Entry point: python generate.py
    │   ├── Inkscape/                  # Manual nesting SVGs
    │   ├── nesting_geometry.json      # Generated
    │   ├── nesting_layout.json        # Saved UI layout
    │   └── output/                    # All generated files
    │
    ├── kitchen-corner/                # Kitchen corner shelves (stub)
    ├── dining-room/                   # L-shaped dining room shelves (stub)
    ├── laundry/                       # Laundry room shelves (stub)
    ├── yarn-shelf/                    # Free-standing yarn shelf (stub)
    ├── bessel-shelf/                  # Bessel function bookshelf (stub)
    ├── bessel-sculpture/              # Sculptural Bessel slices (stub)
    ├── boulder-shelf/                 # Organic boulder-form shelf (stub)
    └── math-bookshelf/                # Mathematical surface bookshelf (stub)
```

### Project Manifest Format (`project.json`)

Each project directory contains a `project.json` describing it:

```json
{
  "name": "pantry",
  "description": "Walk-in pantry with sinusoidal-edge shelves",
  "status": "complete",
  "space": {
    "width_in": 48,
    "depth_in": 49,
    "height_in": 105
  },
  "outputs": {
    "cut_sheets": ["output/sheet_1_nesting.dxf", "output/sheet_2_nesting.dxf"],
    "assembly_guide": "output/pantry_layout.pdf"
  },
  "dependencies": ["lib/geometry.py", "lib/construction.py", "lib/export.py"]
}
```

---

## Migration TODO (pantry → woodwork/projects/pantry)

These are the concrete steps to migrate the current pantry project into the proposed
monorepo structure. Do these when starting the next project so both can coexist.

### Phase 1 — Extract shared library

- [ ] Move `src/geometry.py` → `lib/geometry.py`
      Update all imports in scripts
- [ ] Extract construction-sequence logic from `scripts/generate_from_patterns.py`
      → `lib/construction.py` (classes: `ReferenceResolver`, `BaseGeometrySolver`,
      `ConstructionRenderer`)
- [ ] Extract `ShelfGroup`, `Sheet`, `load_shelves()`, `export_sheet_to_dxf()`
      from `scripts/generate_nested_layouts.py` → `lib/nesting.py`
- [ ] Extract SVG/DXF/PDF export helpers → `lib/export.py`
- [ ] Move `src/config.py` → `lib/config.py`; delete `src/` directory

### Phase 2 — Extract shared tools

- [ ] Move `nesting_ui.html` → `tools/nesting_ui.html`
      Update hard-coded paths/references if any
- [ ] Move `generate_dxf_from_layout.py` → `tools/generate_dxf_from_layout.py`
- [ ] Move `nest_from_inkscape.py` → `tools/nest_from_inkscape.py`
- [ ] Move `scripts/export_shelf_geometry.py` → `tools/export_shelf_geometry.py`

### Phase 3 — Reorganize pantry project

- [ ] Create `projects/pantry/` directory
- [ ] Move `configs/` → `projects/pantry/configs/`
- [ ] Move `Inkscape/` → `projects/pantry/Inkscape/`
- [ ] Move `output/` → `projects/pantry/output/`
- [ ] Move `nesting_geometry.json`, `nesting_layout.json` → `projects/pantry/`
- [ ] Reduce `scripts/generate_from_patterns.py` to thin entry point
      that imports from `lib/` → `projects/pantry/scripts/generate.py`
- [ ] Write `projects/pantry/project.json`

### Phase 4 — Cleanup

- [ ] Delete `src/blender_renderer.py`, `src/pdf_generator.py`,
      `src/shelf_generator.py` (superseded by construction-based workflow)
- [ ] Archive `scripts/old/` → `projects/pantry/scripts/old/`
- [ ] Archive `scripts/alt_{4-9}.py`, `scripts/alt_intersector*.py`
      → `projects/pantry/scripts/old/solver-iterations/`
- [ ] Delete debug scripts (`debug_sheet_order.py`, `debug_svg_coords.py`,
      `dump_svg.py`, `plot_back_left_level19_debug.py`)
- [ ] Delete legacy config files `pantry_0000.json` through `pantry_0005.json`
      (keep `pantry_0002.json` as reference, or fold into `project.json`)
- [ ] Remove duplicate `nesting_layout_old.json`
- [ ] Update top-level `requirements.txt` to cover all lib deps

---

## Future Project Stubs

### kitchen-corner
**Status:** Experimental code exists (`scripts/generate_kitchen_corner_shelves.py`)

Kitchen corner shelves — 3 shelves hanging from cabinet bottom in a 37" × 12" space.
Right angle at back, sinusoidal front edge with 4" radius corner at exposed end.
Existing GP/kriging variant is unfinished. Goal: clean up the working sinusoidal
version and migrate it into the monorepo project structure.

```
space:  37" along wall × 12" depth
shelves: 3 @ 12", 24", 36" from ceiling
thickness: 1"
```

---

### dining-room
**Status:** Stub only

L-shaped shelves for the dining room. The L follows a corner where two walls meet.
Sinusoidal or other organic edge treatment. Key difference from pantry: open room,
so shelves are visible from multiple angles — aesthetics matter more than maximizing
depth. Likely thinner (0.75") with a more dramatic sinusoid.

```
space: TBD
style: L-shaped, two-wall
```

---

### laundry
**Status:** Stub only

Utilitarian laundry room shelves. May not need sinusoidal edges — could be a simpler
straight-edge version using the same stud-bracket mounting system. Useful as a test
of how well the bracket system generalizes without organic geometry.

```
space: TBD
style: rectangular with bracket mounts
```

---

### yarn-shelf
**Status:** Stub only

Free-standing yarn storage shelf — not wall-mounted. Organic form, probably with
through-holes or cutouts for yarn skeins to be visible and accessible. No wall
brackets; needs feet or a base structure. First project requiring a structural
element beyond the shelf faces.

```
space: free-standing, TBD footprint
style: organic cutouts, display-oriented
mounting: floor feet or base frame
```

---

### bessel-shelf
**Status:** Stub only

Bookshelf where the shelf edge profiles are Bessel functions (J₀, J₁, etc.).
The undulating, damped oscillation of Bessel functions creates a natural-looking
but mathematically precise organic form. Could be wall-mounted like the pantry
using the same bracket system, or free-standing.

```
edge curve: J_n(x) scaled to shelf depth
style: bookshelf proportions (tall, narrow bays)
interest: Bessel zeros determine natural "shelf level" positions
```

---

### bessel-sculpture
**Status:** Stub only

A sculptural object (not functional shelving) built from cross-sectional slices
of a 3D surface defined by Bessel functions. Similar to the way a CNC router or
laser cutter can approximate a 3D form by stacking 2D cross-sections with spacers.
Each slice is a different cross-section of the Bessel surface — cut from plywood,
stacked vertically, the assembled piece reads as an organic 3D form.

```
form: stacked 2D cross-sections → implied 3D Bessel surface
material: plywood slices + spacers
output: series of DXFs, one per slice, labeled with assembly order
```

---

### boulder-shelf
**Status:** Stub only

Wall shelf that visually resembles a boulder or stone — very organic, rounded,
no straight edges. Gaussian process curves or splines rather than sinusoids.
The goal is to look like it was carved from stone, not milled from sheet goods.
Probably requires more complex geometry than sinusoids can easily provide —
good candidate for the GP/kriging approach once that's working.

```
style: organic, rounded, stone-like
edge curves: GP/kriging or splines
challenge: GP solver currently experimental
```

---

### math-bookshelf
**Status:** Stub only

A bookshelf where each shelf is defined by a different mathematical surface or
curve — each level is a different function. E.g., shelf 1 = sine, shelf 2 = Bessel,
shelf 3 = Weierstrass function, shelf 4 = Fibonacci spiral. Acts as a display piece
celebrating mathematical forms. Labels or engravings could identify each function.

```
style: bookshelf, each shelf = distinct mathematical curve
output: per-shelf SVG/DXF labeled with function name
interest: decorative + educational
```

---

*Last updated: March 2026*
