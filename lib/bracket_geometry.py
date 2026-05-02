"""
Bracket geometry lookups from a project's stud_positions.json.

The JSON shape this module expects:

    "bracket_geometry": {
        "base_height":        0.2,
        "stem_width":         1.615,
        "base_corner_radius": 0.5,
        "dogbone_radius":     0.25,
        "screw_hole_inset":   0.375,
        "by_tongue_length": {
            "4":  { "base_width": 3.215 },
            "6":  { "base_width": 3.215 },
            "10": { "base_width": 3.975 }
        }
    }

Per-wall configs use a `stud_offset_sign` (+1, -1, or 0). The bracket center
is computed as:

    bracket_center = stud_center + sign * (base_width/2 - screw_hole_inset)

Sign 0 means the bracket is centered on the stud (no shift). Corner brackets
follow a different rule (flush against the wall) and use bracket_dims directly
without the screw-hole formula.
"""

import json
from pathlib import Path


def load_bracket_config(project_dir):
    """Load stud_positions.json from a project directory."""
    cfg_path = Path(project_dir) / 'configs' / 'stud_positions.json'
    with open(cfg_path) as f:
        return json.load(f)


def bracket_dims(cfg, tongue_length):
    """
    Return all bracket dimensions for a given tongue length.

    Returns a dict with keys: base_width, base_height, stem_width,
    base_corner_radius, dogbone_radius, screw_hole_inset.
    """
    g = cfg['bracket_geometry']
    spec = g['by_tongue_length'][str(int(tongue_length))]
    return {
        'base_width':         spec['base_width'],
        'base_height':        g['base_height'],
        'stem_width':         g['stem_width'],
        'base_corner_radius': g['base_corner_radius'],
        'dogbone_radius':     g['dogbone_radius'],
        'screw_hole_inset':   g['screw_hole_inset'],
    }


def stud_to_bracket_offset(cfg, tongue_length, sign):
    """
    Distance to shift bracket center off the stud center, signed.

        offset = sign * (base_width/2 - screw_hole_inset)

    `sign` is typically +1, -1, or 0 (don't shift). Returns 0.0 when sign is 0.
    """
    if not sign:
        return 0.0
    d = bracket_dims(cfg, tongue_length)
    return sign * (d['base_width'] / 2.0 - d['screw_hole_inset'])


def corner_bracket_offset(cfg, tongue_length):
    """
    For corner brackets: distance from the perpendicular wall the bracket sits
    flush against, equal to half its base width. Always positive.
    """
    return bracket_dims(cfg, tongue_length)['base_width'] / 2.0
