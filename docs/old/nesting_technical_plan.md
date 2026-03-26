# Technical Plan: CNC Plywood Nesting Algorithm for Pantry Shelves

## Overview
This document outlines the technical approach to implement a custom nesting algorithm for the pantry shelf pieces. The regular layout script (`generate_shelves_with_brackets.py`) visually organizes shelves by vertical level. This new script will extend that logic to spatially pack the shelves onto 4'x8' (48" x 96") plywood sheet boundaries for CNC machining, minimizing wood waste through specific pairing heuristics.

## 1. Script Architecture
- **Filename**: `scripts/generate_nested_layouts.py` (or similar)
- **Input**:
  - Automatically loads shelf DXFs (`shelf_*.dxf`) from `output/`.
  - Parses `configs/stud_positions.json` to calculate and attach the bracket geometry to each shelf before nesting.
- **Output**:
  - `sheet_1_layouts_with_brackets.dxf`, `sheet_2_layouts_with_brackets.dxf`, etc.
  - Each DXF will contain a 48" x 96" boundary rectangle, representing the stock plywood, along with the nested shelf layouts (including the bracket cut and outline geometries translated along with the shelf paths).

## 2. Plywood Stock Configuration
- **Coordinates Mapping**: 
  - Y-axis represents the 4' (48") span.
  - X-axis represents the 8' (96") span.
- **Gap Requirements**: 
  - Margin between parts: **0.5 inches** (1/2").
  - (Optional) margin between parts and the plywood boundary (e.g., 0.5" or 1.0" for clamping/edge holding).

## 3. General Constraints & Entity Handling
- **Grouping**: A "shelf" is treated as a rigid group comprising its outer boundary (LWPOLYLINE) and all associated bracket geometries (cut lines and outlines). Any translation or rotation applied to a shelf boundary must be identically applied to its associated bracket features.
- **Rotation**: Shelves may only be rotated in orthogonal 90-degree increments (0°, 90°, 180°, 270°) to keep the long sides parallel to the X or Y axis.
- **Bounding Boxes**: For initial placement, every shelf uses an axis-aligned bounding box (AABB) calculated from its path extents.

## 4. Part 1: Nesting the 4 Back Shelves
The back shelves are long and have a distinct sinusoidal front edge. They require a specialized vertical clustering strategy:
- **Orientation**: Rotate the back shelves so their long axis is parallel to the Y-axis (48" span).
- **Placement Logic**: Place all four back shelves sequentially, alternating their facing direction:
  1. **Shelf 1**: Straight back against the left edge (X = 0) of the plywood sheet.
  2. **Shelf 2**: Rotated 180° relative to Shelf 1 so its sinusoidal curve faces the curve of Shelf 1. Packed tightly.
  3. **Shelf 3**: Rotated back to the orientation of Shelf 1. Its straight back will face the straight back of Shelf 2, separated by exactly 0.5".
  4. **Shelf 4**: Rotated 180° relative to Shelf 1 (curve facing Shelf 3). Its straight back should end up at the opposite edge of the plywood or closely packed against Shelf 3.
- **Filling the Gaps**: Due to the interlocking sinusoidal curves, there will be empty spatial gaps. We will iterate through the intermediate shelves (Left/Right thin shelves) and identify shelves whose sinusoidal period lines up with the gaps. These will be slotted into the voids left by the back shelves. This will be the first segment placed on `sheet_1`.

## 5. Part 2: Iterative Sinusoid-Paired Nesting (Remaining Shelves)
For the remaining left, right, and any unplaced intermediate shelves, we will use a pairwise interlocking strategy:

### A. Pairing Heuristic
- Only add the remaining shelves oriented horizontally (long axis parallel to the X-axis / 96" span).
- Identify pairs of shelves whose sinusoidal profiles are closest to being 1/2 out of phase.
- Align the X-coordinates of the pair so that their sinusoids exactly offset each other to allow maximum Y-compression.
- Since the X-translation is locked by the phase-matching, the pair is then compressed entirely along the Y-axis until the shortest line connecting any point on Shelf A to any point on Shelf B is exactly **0.5"**. 

### B. Sheet Placement
We will add these pre-calculated pairs (or individual shelves if they cannot be nicely paired) to the active sheet one by one:
1. **Initial Spot Calculation**: Calculate bounding rectangles for the pair.
2. **Translation Strategy**: 
   - Start at the top-left available coordinate.
   - Sweep "DOWN" the Y-axis (the 48" span).
   - Once the Y-axis space is exhausted, step "RIGHT" along the X-axis (the 96" span) and resume sweeping down.
3. **Collision Checking**: Verify that the bounding rectangle of the new pair does not intersect the bounding rectangle of any previously placed shelves on the current sheet.
4. **Fine-Tuning (Bracketing Algorithm)**: 
   - Once a non-intersecting AABB placement is found, establish the minimum physical distance between the true geometry of the new shelf pair and the adjacent placed shelves.
   - Use a 1D bracketing/binary search algorithm (translating strictly on the Y-axis or X-axis depending on the approach vector) to slide the new shelf closer to the cluster until the exact minimum distance to its neighbor reaches **0.5"**.

## 6. Iteration & Pagination (Multiple Sheets)
- If a shelf (or a shelf pair) cannot find a valid non-intersecting AABB placement on the current active sheet, it signals that the sheet is full.
- The script will finalize the current `sheet_N_layouts_with_brackets.dxf` and instantiate a new virtual 48"x96" boundary for `sheet_N+1`.
- The placement logic resumes at the top-left of the new sheet.
- This process repeats until the queue of all shelves (and their associated brackets) is empty.

## 7. Libraries and Tools Required
- **`ezdxf`**: Existing dependency. Used for generating the DXF files, reading the shelf lines, and grouping geometries.
- **`numpy`**: Existing dependency. Essential for rigid body translations, rotations, and vector calculations (e.g., finding the shortest distance between two discrete polyline sets). 
- **`shapely`** (Recommended new dependency): While `numpy` can do point-to-point brute force distance calculations, using `shapely`'s `Polygon` and `distance()` methods will drastically simplify the AABB collision checks, true geometry intersection checks, and the 0.5" offset calculations required for the fine-tuning/bracketing phase.
