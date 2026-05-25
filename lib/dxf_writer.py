"""
DXF contour writers — legacy (polyline) and analytic (bulge-encoded arcs).

Legacy path:  write_contour_legacy(msp, pts, layer)
  - Single closed LWPOLYLINE from flat (x, y) point list.  Existing behaviour.

Analytic path: write_contour_analytic(msp, vertices, layer)
  - Single closed LWPOLYLINE from 5-tuple (x, y, sw, ew, bulge) vertex list.
  - Non-zero bulge values encode circular arcs exactly; no sampling required.
  - Spline sections (sinusoidal edges) can be added as separate SPLINE entities
    alongside the LWPOLYLINE — future extension, not yet implemented here.

Switch between paths via the "analytic_dxf" key in the project config.
"""

import ezdxf
from ezdxf import units


def write_contour_legacy(msp, pts, layer, dxfattribs=None):
    """Write a closed contour as a plain LWPOLYLINE from (x, y) point list."""
    attrs = {'layer': layer}
    if dxfattribs:
        attrs.update(dxfattribs)
    closed_pts = list(pts) + [pts[0]]
    msp.add_lwpolyline(
        [(x, y, 0) for x, y in closed_pts],
        close=True,
        dxfattribs=attrs,
    )


def write_contour_analytic(msp, vertices, layer, dxfattribs=None):
    """Write a closed contour as an LWPOLYLINE with bulge-encoded arcs.

    `vertices` is a list of 5-tuples (x, y, start_width, end_width, bulge).
    bulge=0  → straight segment to next vertex.
    bulge≠0  → circular arc to next vertex (tan(included_angle/4)).
    """
    attrs = {'layer': layer}
    if dxfattribs:
        attrs.update(dxfattribs)
    msp.add_lwpolyline(vertices, close=True, dxfattribs=attrs)


def new_dxf_doc():
    """Create a fresh R2010 DXF document with inch units."""
    doc = ezdxf.new(dxfversion='R2010')
    doc.units = units.IN
    return doc
