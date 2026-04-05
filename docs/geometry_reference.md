# Geometry Reference: `nesting_geometry.json` Objects

Every shelf and bracket in `nesting_geometry.json` is described below, vertex-range by vertex-range.
Use this document to specify exactly which vertices to keep, remove, or extend in each output layer.

---

## Coordinate System

All coordinates are **local shelf coords** (re-centered at bbox center, Y-down SVG convention).
In `generate_dxf_from_layout.py`, the nesting transform applies rotation + Y-flip + sheet translation.

The shelf's **mount wall** is always on one edge of the bounding box:

| Shelf type | Wall side in local coords |
|---|---|
| R (right wall) | Negative X side |
| L (left wall) | Positive X side |
| B (back wall) | Negative Y side (back wall brackets); ±X sides (side + corner brackets) |

---

## Constants

| Name | Value | Description |
|---|---|---|
| `BASE_R` | 0.5" | Radius of wall-entry/exit arcs |
| `BASE_HEIGHT` | 0.2" | Base plate thickness (perpendicular to wall) |
| `BASE_WIDTH` | 3.2" | Base plate extent along wall |
| `STEM_WIDTH` | 1.6" | Tongue width |
| `DOGBONE_R` | 0.25" | Dogbone arc radius at tongue-tip inside corners |

---

## 1. `shelf_R` poly — Right-wall shelf (example: `shelf_R19`, 110 pts)

The mount wall is at **negative X**. The shelf outline traces CCW.

| Range | Feature | Approx seg | Notes |
|---|---|---|---|
| `[0]→[1]` | Door-side straight edge | 2.56" | Short straight from door notch to door arc |
| `[2]→[16]` | Door-notch arc | ≈0.084/seg, 15 pts | 0.5"-radius arc for door clearance |
| `[17]` | Duplicate of `[16]` | 0.0 | Zero-length step (polygon seam) |
| `[18]→[37]` | Sinusoidal front edge (door-half) | ≈0.276/seg, 20 pts | Sampled sine curve, shallower region |
| `[38]` | Duplicate of `[37]` | 0.0 | Zero-length step |
| `[39]→[87]` | Sinusoidal main edge | ≈0.46–0.48/seg, 49 pts | Primary sinusoidal front face |
| `[88]` | Duplicate of `[87]` | 0.0 | Zero-length step |
| `[89]→[107]` | Door-smoothing arc | ≈0.124/seg, 19 pts | Large arc blending sine into door wall |
| `[108]` | Bottom straight edge | 6.11" | Shelf back face (wall-side base, Y const) |
| `[109]→[0]` | **Wall edge** | 28.96" | Straight line along mount wall (X const) |

**Wall edge** `[109]→[0]`: this is the segment that runs along the right wall. Each bracket base slot will be cut into this edge.

---

## 2. `shelf_L` poly — Left-wall shelf (example: `shelf_L19`, 133 pts)

The mount wall is at **positive X**. L shelves have a D-shaped pipe cutout on the wall edge.

| Range | Feature | Approx seg | Notes |
|---|---|---|---|
| `[0]→[1]` | Top straight edge | 6.0" | Along wall edge above D-shape |
| `[2]→[16]` | Door-notch arc | ≈0.084/seg, 15 pts | 0.5"-radius door clearance arc |
| `[17]` | Duplicate | 0.0 | Zero-length step |
| `[18]→[37]` | Sinusoidal front edge (door-half) | ≈0.115/seg, 20 pts | Sampled sine, shallower region |
| `[38]` | Duplicate | 0.0 | Zero-length step |
| `[39]→[87]` | Sinusoidal main edge | ≈0.55/seg, 49 pts | Primary sinusoidal front face |
| `[88]` | Duplicate | 0.0 | Zero-length step |
| `[89]→[107]` | Door-smoothing arc | ≈0.122/seg, 19 pts | Large arc blending sine into door wall |
| `[107]→[108]` | Bottom straight edge | 7.73" | Shelf back face |
| `[108]→[109]` | **Wall edge (lower section)** | 23.46" | Straight along mount wall, below D-shape |
| `[109]→[110]` | Step to D-shape start | 3.75" | Along wall edge, to start of cutout |
| `[110]→[130]` | **D-shape pipe cutout arc** | ≈0.294/seg, 21 pts | ~1.875" radius semicircle for pipe clearance |
| `[130]→[131]` | Step past D-shape | 3.75" | Along wall edge, past cutout |
| `[131]→[132]` | **Wall edge (upper section)** | 2.625" | Straight along mount wall, above D-shape |
| `[132]→[0]` | Closing step | 0.0 | Coincident with `[0]` |

**Wall edge** = segments `[108]→[109]`, `[109]→[110]`, `[130]→[131]`, `[131]→[132]` — all at X = +4.5 (in shelf_L19). The D-shape arc `[110]→[130]` is a pre-existing cutout; bracket slots that fall in this region should be clipped against it rather than drawn.

---

## 3. `shelf_B` poly — Back-wall shelf (example: `shelf_B19`, 96 pts)

The mount wall is at **negative Y** (bottom). Two additional sinusoidal smoothing arcs appear on the left and right ends.

| Range | Feature | Approx seg | Notes |
|---|---|---|---|
| `[0]→[1]` | **Right wall edge** | 48.0" | Long straight along left side of bbox |
| `[1]→[2]` | **Back wall edge (left section)** | 20.04" | Straight along back wall (Y const) |
| `[2]→[3]` | Back face straight | 6.11" | Shelf back face (sinusoid peak) |
| `[4]→[23]` | Left door-smoothing arc | ≈0.124/seg, 20 pts | Arc blending into left side |
| `[23]` | Duplicate | 0.0 | Zero-length step |
| `[24]→[73]` | Sinusoidal front face | ≈0.62/seg, 50 pts | Primary sinusoidal edge |
| `[74]` | Duplicate | 0.0 | Zero-length step |
| `[74]→[93]` | Right door-smoothing arc | ≈0.122/seg, 20 pts | Arc blending into right side |
| `[93]→[94]` | Back face right section | 7.73" | Back face right of sinusoid |
| `[94]→[95]` | **Right wall edge** | 19.16" | Along right side of bbox |
| `[95]→[0]` | Closing step | 0.0 | Coincident with `[0]` |

**Wall edges** = `[0]→[1]` (right side of bbox, X = +24.0) and `[1]→[2]` (bottom, Y = -10.02). Back-wall bracket slots go into `[1]→[2]`; side and corner bracket slots go into `[0]→[1]` and its symmetric left counterpart.

---

## 4. Bracket polygon — all variants (always 87 pts)

Every bracket follows the same 87-point structure regardless of wall orientation:

```
[0]         wall-edge reference point (on wall surface)
[1]–[8]     entry arc — 8 arc segments (~0.058/seg)
              curves from wall face to base-plate inner face (BASE_R = 0.5")
[8]         corner at inner base face, far side of stem
[8]→[9]     straight along inner base face, len = (BASE_WIDTH−STEM_WIDTH)/2 = 0.8"
[9]         OUTER ARMPIT — corner where base plate meets stem edge (far side)
[9]→[10]    straight from outer armpit to inner armpit, len = tongue_length − BASE_HEIGHT
              (= 3.8" for 4" tongue, 5.8" for 6" tongue, 9.8" for 10" tongue)
[10]        INNER ARMPIT (far side) — first point of dogbone arc
[11]–[42]   FAR DOGBONE ARC — 33 pts (~0.0245/seg, radius = DOGBONE_R = 0.25")
[42]        end of far dogbone arc
[42]→[43]   across tongue tip, len = STEM_WIDTH − 2*chord_half ≈ 0.893"
[43]        start of near dogbone arc
[44]–[75]   NEAR DOGBONE ARC — 33 pts (~0.0245/seg)
[75]        INNER ARMPIT (near side) — last point of near dogbone arc
[75]→[76]   straight from inner armpit to outer armpit, len = tongue_length − BASE_HEIGHT
[76]        OUTER ARMPIT (near side)
[76]→[77]   straight along inner base face, len = 0.8"
[77]        corner at inner base face, near side of stem
[78]–[85]   exit arc — 8 arc segments (~0.058/seg), BASE_R = 0.5"
[85]        wall-edge reference point (other end)
[85]→[86]   CLOSING WALL-EDGE SEGMENT — straight along wall, len = BASE_WIDTH = 4.0"
              (this is the segment that lies on the shelf's mount wall line)
[86]→[0]    zero-length close (same point as [0])
```

**Key indices by semantic meaning:**

| Index | Semantic |
|---|---|
| `[0]` | Wall-edge start |
| `[8]` | Base plate inner face, far corner |
| `[9]` | Outer armpit (far) |
| `[10]` | Inner armpit (far) — start of dogbone |
| `[11]–[42]` | Far dogbone arc |
| `[42]→[43]` | Tongue tip flat |
| `[43]` | Start of near dogbone arc |
| `[44]–[75]` | Near dogbone arc |
| `[75]` | Inner armpit (near) — end of dogbone |
| `[76]` | Outer armpit (near) |
| `[77]` | Base plate inner face, near corner |
| `[85]` | Wall-edge end |
| `[85]→[86]→[0]` | Closing wall-edge segment (on wall line) |

### Bracket variants by wall

| Variant | Wall coord | Tongue direction | Tongue length | Example shelf |
|---|---|---|---|---|
| R-wall bracket | x = −3.5 (local) | +X (into shelf) | 4.0" | shelf_R19, brackets[0–2] |
| L-wall bracket | x = +4.5 (local) | −X (into shelf) | 6.0" | shelf_L19, brackets[0–1] |
| B-wall bracket | y = −10.02 (local) | +Y (into shelf) | 10.0" | shelf_B19, brackets[0–2] |
| B side-right bracket | x = +24.0 (local) | −X (into shelf) | 10.0" | shelf_B19, bracket[3] |
| B side-left bracket | x = −24.0 (local) | +X (into shelf) | 10.0" | shelf_B19, bracket[4] |
| B corner-right bracket | x = +24.0 (local) | −X (into shelf) | 4.0" | shelf_B19, bracket[5] |
| B corner-left bracket | x = −24.0 (local) | +X (into shelf) | 4.0" | shelf_B19, bracket[6] |

---

## 5. Cut line

Each bracket has a corresponding cut line (`cuts` array), a **2-point line segment**:

```
[ [inner_armpit_near], [inner_armpit_far] ]
```

These are the two **inner armpits** (`[75]` and `[10]` in the bracket polygon). The cut line lies at:

- R-wall brackets: x = wall_x + BASE_HEIGHT (e.g., x = −3.3)
- L-wall brackets: x = wall_x − BASE_HEIGHT (e.g., x = +4.3)
- B-wall brackets: y = wall_y + BASE_HEIGHT (e.g., y = −9.82)
- B side/corner brackets: x = ±(24.0 − 0.2) = ±23.8

The cut line **divides the base-plate region from the tongue/pocket region**. It is the boundary between CONTOUR layer geometry and POCKET layer geometry.

---

## Summary of which vertices belong to which region

For a bracket polygon, the regions are:

```
WALL LINE (on shelf wall): segments [85]→[86]→[0]
BASE PLATE EXTERIOR: entry arc [0]→[8], segments [8]→[9], [76]→[77], exit arc [77]→[85]
BASE PLATE INTERIOR EDGE: segment [9]→[10] and [75]→[76]   ← cut line location
TONGUE / POCKET: inner armpits [10], [75], dogbones [11]–[75], tongue tip [42]→[43]
```

The cut line (`cuts` entry) connects `[10]` to `[75]` — the inner armpits — completing the closed
boundary between base and tongue regions.
