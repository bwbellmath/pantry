#!/usr/bin/env python3
"""
Generate 3D renderings using exact geometry from DXF shelf files.

Two-stage process:
  1. Extract shelf vertices from DXF files (using system Python + ezdxf)
  2. Launch Blender with the extracted geometry as JSON

Usage:
    python scripts/render_from_svg.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import ezdxf
import numpy as np


# ---------------------------------------------------------------------------
# Pantry dimensions (from shelf_level_patterns.json)
# ---------------------------------------------------------------------------
PANTRY_WIDTH = 48.0
PANTRY_DEPTH = 49.0
PANTRY_HEIGHT = 105.0
SHELF_THICKNESS = 1.0

# Shelf definitions
MAIN_SHELF_HEIGHTS = [19, 39, 59, 79]
LEFT_INTERMEDIATE_HEIGHTS = [9, 29, 49, 69]
RIGHT_INTERMEDIATE_HEIGHTS = [5, 13, 26, 33, 46, 53, 66, 73, 86]


def load_shelf_from_dxf(dxf_path):
    """Load shelf outline vertices from a DXF file (LWPolyline)."""
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            verts = [(v[0], v[1]) for v in entity.get_points(format='xy')]
            if len(verts) > 1 and np.allclose(verts[0], verts[-1], atol=1e-6):
                verts = verts[:-1]
            return verts

    raise ValueError(f"No LWPolyline found in DXF file: {dxf_path}")


def extract_all_shelves(output_dir):
    """Extract vertices from all shelf DXF files."""
    shelves = []

    def try_load(prefix, height):
        dxf_path = output_dir / f'shelf_{prefix}{height}.dxf'
        if not dxf_path.exists():
            print(f"  WARNING: {dxf_path.name} not found")
            return
        verts = load_shelf_from_dxf(dxf_path)
        shelves.append({
            'name': f'shelf_{prefix}{height}',
            'height': float(height),
            'vertices': verts,
        })
        print(f"  {dxf_path.name}: {len(verts)} vertices")

    print("Extracting shelf geometry from DXF files...")

    print("  Main shelves:")
    for h in MAIN_SHELF_HEIGHTS:
        for prefix in ['L', 'B', 'R']:
            try_load(prefix, h)

    print("  Left intermediate:")
    for h in LEFT_INTERMEDIATE_HEIGHTS:
        try_load('L', h)

    print("  Right intermediate:")
    for h in RIGHT_INTERMEDIATE_HEIGHTS:
        try_load('R', h)

    print(f"Total: {len(shelves)} shelves extracted")
    return shelves


def main():
    output_dir = Path('output')
    render_dir = output_dir / 'renders'
    render_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Extract DXF geometry
    shelves = extract_all_shelves(output_dir)

    # Write to temp JSON for Blender to consume
    geometry_data = {
        'pantry_width': PANTRY_WIDTH,
        'pantry_depth': PANTRY_DEPTH,
        'pantry_height': PANTRY_HEIGHT,
        'shelf_thickness': SHELF_THICKNESS,
        'shelves': shelves,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(geometry_data, f)
        geometry_path = f.name

    print(f"\nGeometry written to: {geometry_path}")

    # Stage 2: Launch Blender
    blender_script = Path(__file__).parent / 'render_blender.py'
    print(f"\nLaunching Blender...")
    result = subprocess.run(
        ['blender', '--background', '--python', str(blender_script), '--', geometry_path],
        cwd=str(Path(__file__).parent.parent),
    )

    # Clean up temp file
    Path(geometry_path).unlink(missing_ok=True)

    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
