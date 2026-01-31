# Construction-Based Geometry Patterns

This document describes the construction-based development approach used in this pantry shelf system, where all geometry is specified as sequences of primitive operations in JSON configuration files.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Geometric Primitives](#geometric-primitives)
3. [Base Geometry](#base-geometry)
4. [Reference System](#reference-system)
5. [Solve Methods](#solve-methods)
6. [Complete Examples](#complete-examples)
7. [Adding New Features](#adding-new-features)

---

## Philosophy

**Every shelf is completely configured via JSON.** No shelf-specific geometry is hard-coded in Python. Instead:

1. Define **base geometry** (sinusoids, intersections, arcs) at each level
2. Describe each shelf as a **sequence of construction steps** using geometric primitives
3. Reference base geometry elements using a **path-based reference system**
4. Let the rendering engine interpret and execute the construction

This approach ensures:
- **Single source of truth**: All geometry lives in JSON
- **Maintainability**: Modify designs without touching code
- **Reusability**: Construction templates apply across multiple levels
- **Transparency**: Complete geometric intent visible in configuration

---

## Geometric Primitives

Each shelf's `construction` array is a sequence of primitive operations that form a closed path.

### 1. Point

Explicit coordinate to start or mark a position.

```json
{
  "step": 1,
  "type": "point",
  "description": "Start at wall/door corner",
  "coords": [48.0, 0]
}
```

**Fields:**
- `coords`: `[x, y]` coordinate (can use references)

### 2. Straight Line

Line segment from current position to a new point.

```json
{
  "step": 2,
  "type": "straight_line",
  "description": "Door edge to notch intersection",
  "to": [44.0, 0]
}
```

**Fields:**
- `to`: `[x, y]` destination (can use references)
- `close_path`: (optional) `true` if this line closes the polygon

### 3. Arc

Circular arc with explicit center, radius, and angle range.

```json
{
  "step": 3,
  "type": "arc",
  "description": "Door notch (concave quarter-circle)",
  "center": [44.0, -0.75],
  "radius": 0.75,
  "start_angle": 90,
  "end_angle": 180,
  "direction": "ccw"
}
```

**Fields:**
- `center`: `[x, y]` center of circle
- `radius`: Arc radius in inches
- `start_angle`: Starting angle in degrees (0° = east, 90° = north)
- `end_angle`: Ending angle in degrees
- `direction`: `"cw"` (clockwise) or `"ccw"` (counter-clockwise)

### 4. Sinusoid Segment

Portion of a sinusoidal curve defined in base geometry.

```json
{
  "step": 6,
  "type": "sinusoid_segment",
  "description": "Right edge sinusoid from door smoothing to back arc",
  "sinusoid": "right",
  "from": 3.2,
  "to": 28.5,
  "direction": "increasing_y"
}
```

**Fields:**
- `sinusoid`: Name of sinusoid from base geometry (e.g., `"right"`, `"left"`, `"back"`)
- `from`: Starting parameter value (y for vertical sinusoids, x for horizontal)
- `to`: Ending parameter value
- `direction`: (optional) `"increasing_y"`, `"decreasing_y"`, etc.

**Note:** The actual x-coordinate is computed from the sinusoid formula.

### 5. Arc Segment

Portion of a pre-computed tangent arc from base geometry.

```json
{
  "step": 7,
  "type": "arc_segment",
  "description": "Back corner arc from right sinusoid to arc midpoint",
  "arc": {"ref": "arcs.back_right_arc"},
  "from_tangent": "right",
  "to": {"ref": "arc_midpoints.back_right_arc_midpoint"}
}
```

**Fields:**
- `arc`: Reference to arc in base geometry
- `from_tangent`: (optional) Which sinusoid the arc starts tangent to
- `to_tangent`: (optional) Which sinusoid the arc ends tangent to
- `from`: (optional) Explicit starting point reference
- `to`: (optional) Explicit ending point reference

### 6. Solved Arc

Arc computed on-the-fly to satisfy tangency constraints.

```json
{
  "step": 5,
  "type": "arc",
  "description": "Door smoothing arc (convex, tangent to sinusoid)",
  "solve_method": "tangent_horizontal_line_and_sinusoid",
  "anchor_point": [6.9375, -0.75],
  "tangent_to_sinusoid": "left",
  "min_radius": 5.0,
  "direction": "ccw"
}
```

**Fields:**
- `solve_method`: Name of geometric solver to use
- Additional fields depend on the solver (anchor points, constraints, etc.)

---

## Base Geometry

Base geometry pre-computes shared elements used by multiple shelves at a given level.

### Sinusoids

Define sinusoidal curves along walls.

**Vertical Sinusoid** (along y-axis, varies in x):
```json
{
  "right": {
    "type": "vertical",
    "wall": "W",
    "x_base": 48.0,
    "offset_from_wall": 5.0,
    "y_start": 0,
    "y_end": 49.0,
    "period": 24.0,
    "amplitude": 1.0,
    "phase_offset": 2.667
  }
}
```

Formula: `x(y) = x_base - offset_from_wall - amplitude * sin(2π * y / period + phase_offset)`

**Horizontal Sinusoid** (along x-axis, varies in y):
```json
{
  "back": {
    "type": "horizontal",
    "wall": "S",
    "y_base": 49.0,
    "offset_from_wall": 19.0,
    "x_start": 0,
    "x_end": 48.0,
    "period": 24.0,
    "amplitude": 1.0,
    "phase_offset": 1.749
  }
}
```

Formula: `y(x) = y_base - offset_from_wall - amplitude * sin(2π * x / period + phase_offset)`

### Intersections

Points where two sinusoids meet.

```json
{
  "back_right": {
    "description": "Intersection of right and back sinusoids",
    "sinusoid_1": "right",
    "sinusoid_2": "back",
    "solve_method": "two_sinusoid_intersection"
  }
}
```

**Solver:** Finds `(x, y)` where both sinusoid equations are satisfied simultaneously.

### Arcs

Tangent circles computed to smoothly connect curves.

**Dual-Sinusoid Tangent Arc:**
```json
{
  "back_right_arc": {
    "description": "Slope-matching circle at right-back corner",
    "type": "tangent_circle",
    "radius": 3.0,
    "tangent_to": ["right", "back"],
    "intersection_point": {"ref": "intersections.back_right"},
    "solve_method": "slope_matching_two_sinusoids"
  }
}
```

**Solver:** Finds circle center such that:
1. Circle has the specified radius
2. Circle is tangent to both sinusoids (matches slopes)
3. Tangency points are near the intersection

**Horizontal Line + Sinusoid Tangent Arc:**
```json
{
  "back_arc": {
    "description": "Simple corner arc tangent to sinusoid and horizontal line",
    "type": "tangent_circle",
    "radius": 3.0,
    "tangent_to_sinusoid": "left",
    "tangent_to_line": "horizontal",
    "line_y": 29.0,
    "solve_method": "tangent_horizontal_and_sinusoid"
  }
}
```

**Solver:** Finds circle center such that:
1. Circle is tangent to horizontal line at `y = line_y`
2. Circle is tangent to the specified sinusoid
3. Circle has the specified radius

### Arc Midpoints

Computed points along arcs used to divide shelf pieces.

```json
{
  "back_right_arc_midpoint": {
    "description": "Point dividing right shelf from back shelf",
    "arc": {"ref": "arcs.back_right_arc"},
    "position": "midpoint"
  }
}
```

**Computation:** Finds the point halfway along the arc (by angle).

---

## Reference System

Values can reference other parts of the configuration using the `{"ref": "path"}` syntax.

### Reference Syntax

**Direct reference:**
```json
{"ref": "pantry_width"}           // → 48.0
{"ref": "design_params.sinusoid_amplitude"}  // → 1.0
```

**Nested reference:**
```json
{"ref": "arcs.back_right_arc.tangent_point_on_right.y"}
```

**Calculation syntax** (simple arithmetic):
```json
{"calc": "pantry_width - door_clearance_west"}  // → 48.0 - 4.0 = 44.0
{"calc": "-door_extension"}                      // → -0.75
```

### Common References

| Reference | Description |
|-----------|-------------|
| `pantry_width` | 48.0 (right wall x-coordinate) |
| `pantry_depth` | 49.0 (back wall y-coordinate) |
| `design_params.interior_corner_radius` | 3.0 |
| `arcs.back_right_arc` | Computed arc object |
| `intersections.back_right` | Computed intersection point |
| `arc_midpoints.back_right_arc_midpoint.x` | X-coordinate of midpoint |

---

## Solve Methods

Geometric solvers compute points/arcs that satisfy constraints.

### 1. `two_sinusoid_intersection`

**Purpose:** Find where two sinusoids intersect.

**Inputs:**
- `sinusoid_1`: Name of first sinusoid
- `sinusoid_2`: Name of second sinusoid

**Outputs:**
- `(x, y)`: Intersection point

**Algorithm:**
- Numerically solve system where both sinusoid equations equal
- Use bracketing + bisection for robustness

---

### 2. `slope_matching_two_sinusoids`

**Purpose:** Find a circle tangent to two sinusoids near their intersection.

**Inputs:**
- `radius`: Desired circle radius
- `tangent_to`: Array of two sinusoid names
- `intersection_point`: Reference to intersection

**Outputs:**
- Circle center `(cx, cy)`
- Tangency points on each sinusoid
- Arc angles

**Algorithm:**
- Start near intersection point
- Optimize circle center using gradient constraints
- Ensure tangency: circle normal = sinusoid normal at tangency points

**Implementation:** `solve_tangent_circle_two_sinusoids()` in `src/geometry.py`

---

### 3. `tangent_horizontal_and_sinusoid`

**Purpose:** Find a circle tangent to a horizontal line and a sinusoid.

**Inputs:**
- `radius`: Desired circle radius
- `tangent_to_sinusoid`: Sinusoid name
- `tangent_to_line`: "horizontal"
- `line_y`: Y-coordinate of horizontal line

**Outputs:**
- Circle center `(cx, cy)`
- Tangency point on sinusoid `(tx, ty)`

**Algorithm:**
- Center y-coordinate fixed: `cy = line_y - radius`
- Solve for y-tangency on sinusoid using constraint equation
- Center x-coordinate computed from tangency condition

**Implementation:** `solve_tangent_circle_horizontal_sinusoid()` in `scripts/extract_and_export_geometry.py`

---

### 4. `tangent_horizontal_line_and_sinusoid` (Door Smoothing)

**Purpose:** Find a circle with a fixed tangency point on a horizontal line, tangent to a sinusoid.

**Inputs:**
- `anchor_point`: `[x, y]` where circle touches horizontal line
- `tangent_to_sinusoid`: Sinusoid name
- `min_radius`: Minimum acceptable radius

**Outputs:**
- Circle center `(cx, cy)`
- Circle radius `r`
- Tangency point on sinusoid `(tx, ty)`

**Algorithm:**
- Center x-coordinate fixed at `anchor_point[0]`
- Center y-coordinate: `anchor_point[1] + radius`
- Find tangency on sinusoid by minimizing distance function
- Return smallest valid radius ≥ `min_radius`

**Implementation:** `solve_door_smoothing_fixed_radius()` in `scripts/extract_and_export_geometry.py`

---

## Complete Examples

### Example 1: Simple Left Intermediate Shelf

**Level:** 9.0" height, left wall only

**Base Geometry:**
- 1 sinusoid: `left` (vertical, 7" depth)
- 1 arc: `back_arc` (tangent to sinusoid and horizontal line at y=29")

**Construction:**
```json
{
  "construction": [
    {"type": "point", "coords": [0, 0]},
    {"type": "straight_line", "to": [0, 29.0]},
    {"type": "straight_line", "to": [{"ref": "arcs.back_arc.tangent_point_on_horizontal.x"}, 29.0]},
    {"type": "arc_segment", "arc": {"ref": "arcs.back_arc"}, "from_tangent": "horizontal", "to_tangent": "left"},
    {"type": "sinusoid_segment", "sinusoid": "left", "from": {"ref": "arcs.back_arc.tangent_point_on_left.y"}, "to": 0},
    {"type": "straight_line", "to": [0, 0], "close_path": true}
  ]
}
```

**Path Trace:**
1. Start at NW corner `(0, 0)`
2. Wall edge south to `(0, 29)`
3. Back edge east to arc tangency `(~7, 29)`
4. Arc from horizontal line to sinusoid tangency
5. Sinusoid edge north back to `(~7, 0)`
6. Door edge west back to origin

---

### Example 2: Main Right Shelf with Door Features

**Level:** 19.0" height, right wall (W)

**Base Geometry:**
- 3 sinusoids: `right`, `left`, `back`
- 2 intersections: `back_right`, `back_left`
- 2 arcs: `back_right_arc`, `back_left_arc`
- 2 arc midpoints

**Construction (simplified):**
```json
{
  "construction": [
    {"type": "point", "coords": [48, 0]},
    {"type": "straight_line", "to": [44, 0]},
    {"type": "arc", "center": [44, -0.75], "radius": 0.75, "start_angle": 90, "end_angle": 180},
    {"type": "straight_line", "to": [43.125, -0.75]},
    {"type": "arc", "solve_method": "tangent_horizontal_line_and_sinusoid", "anchor_point": [43.125, -0.75], "tangent_to_sinusoid": "right", "min_radius": 5.0},
    {"type": "sinusoid_segment", "sinusoid": "right", "from": "step_5.tangent_y", "to": "arcs.back_right_arc.tangent_on_right.y"},
    {"type": "arc_segment", "arc": "arcs.back_right_arc", "from_tangent": "right", "to": "arc_midpoints.back_right_arc_midpoint"},
    {"type": "straight_line", "to": [48, "arc_midpoints.back_right_arc_midpoint.y"]},
    {"type": "straight_line", "to": [48, 0], "close_path": true}
  ]
}
```

**Path Trace:**
1. Start at NE corner `(48, 0)`
2. Door edge to notch `(44, 0)`
3. Concave notch arc to `(43.25, -0.75)`
4. Door line to smoothing anchor `(43.125, -0.75)`
5. Convex smoothing arc to sinusoid `(~43, ~2)`
6. Sinusoid edge to back arc `(~43, ~28)`
7. Back arc to midpoint `(~45.5, ~29)`
8. Horizontal line to wall `(48, ~29)`
9. Wall edge back to origin

---

## Adding New Features

### Step 1: Define Base Geometry Element

If your feature requires a new geometric element (e.g., a new arc type):

**Add to `base_geometry`:**
```json
{
  "arcs": {
    "new_feature_arc": {
      "description": "Special arc for new feature",
      "type": "tangent_circle",
      "radius": 4.0,
      "solve_method": "my_new_solver",
      "constraint_1": "value1",
      "constraint_2": "value2"
    }
  }
}
```

### Step 2: Implement Solver (if needed)

If you defined a new `solve_method`, implement it in `src/geometry.py`:

```python
def solve_my_new_feature(constraint_1, constraint_2, radius):
    """
    Docstring explaining what this solver does.
    """
    # Solve for center, tangency points, etc.
    return center_x, center_y, tangent_point_x, tangent_point_y
```

### Step 3: Add to Construction Sequence

Reference the new element in a shelf's construction:

```json
{
  "construction": [
    ...
    {
      "step": 6,
      "type": "arc_segment",
      "description": "Use new feature arc",
      "arc": {"ref": "arcs.new_feature_arc"},
      "from": "point_a",
      "to": "point_b"
    },
    ...
  ]
}
```

### Step 4: Update Renderer

Modify `scripts/extract_and_export_geometry.py` to interpret the new primitive:

```python
def render_construction(construction, base_geometry):
    for step in construction:
        if step['type'] == 'arc_segment':
            arc = resolve_reference(step['arc'], base_geometry)
            # Generate arc points...
        # ... other cases
```

### Step 5: Test and Validate

```bash
python scripts/extract_and_export_geometry.py
# Check output SVG files for correct geometry
```

---

## Best Practices

1. **Always use references** instead of hard-coding dimensions
2. **Add clear descriptions** to every construction step
3. **Validate constraints** (e.g., arcs stay inside shelf bounds)
4. **Reuse construction templates** with `construction_template` references
5. **Document new solve methods** with algorithm descriptions
6. **Keep base geometry minimal** - only compute what's shared across shelves
7. **Test thoroughly** - small changes can affect multiple shelves

---

## Reference Implementation

See `configs/shelf_level_patterns.json` for the complete reference implementation of all 25 pantry shelves using this construction-based approach.

For solver implementations, see:
- `src/geometry.py` - Core geometric solvers
- `scripts/extract_and_export_geometry.py` - Construction rendering

---

**Last updated:** January 22, 2026
**Schema version:** pattern_v1
