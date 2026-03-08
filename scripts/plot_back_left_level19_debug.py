#!/usr/bin/env python3
"""
Debug plot for level 19 back-left tangent circle.

Renders:
- Left (vertical) sinusoid in pantry coordinates
- Back (horizontal) sinusoid in pantry coordinates
- Tangent circle center and radius
- Tangency points (left/back)
- Arc between tangency points
- Arc midpoint
"""

import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry import (
    tangent_circle_two_sinusoids_offset_intersection,
    _short_arc_span,
    _wrap_to_pi,
)


def main() -> None:
    # Level 19 parameters from configs/shelf_level_patterns.json
    pantry_width = 48.0
    pantry_depth = 49.0
    period = 24.0
    amplitude = 1.0
    radius = 3.0

    left_depth = 8.0
    left_offset = 3.4143139461097722

    back_depth = 19.0
    back_offset = 1.7490464303899562

    # Map to solver coordinates (same as solve_tangent_circle_two_sinusoids_newton)
    x_pos = left_depth
    Av = amplitude
    y_off_v = -left_offset * period / (2.0 * math.pi)

    y_pos = pantry_depth - back_depth
    Ah = -amplitude
    x_off_h = -back_offset * period / (2.0 * math.pi)

    quadrant = "br"

    result = tangent_circle_two_sinusoids_offset_intersection(
        Av=Av,
        x_pos=x_pos,
        y_off_v=y_off_v,
        Ah=Ah,
        y_pos=y_pos,
        x_off_h=x_off_h,
        r=radius,
        quadrant=quadrant,
        period_v=period,
        period_h=period,
    )

    center = np.array(result["center"])
    p_vert = np.array(result["raw_points"]["vertical"])
    p_horz = np.array(result["raw_points"]["horizontal"])
    p_ordered = [np.array(p) for p in result["points"]]

    # Build the actual sinusoids in pantry coordinates
    # Left wall (E): x = base + amp * sin(2π*y/period + phase)
    y_vals = np.linspace(0.0, pantry_depth, 600)
    x_left = left_depth + amplitude * np.sin(2.0 * math.pi * y_vals / period + left_offset)

    # Back wall (S): y = pantry_depth - back_depth - amp * sin(2π*x/period + phase)
    x_vals = np.linspace(0.0, pantry_width, 600)
    y_back = pantry_depth - back_depth - amplitude * np.sin(2.0 * math.pi * x_vals / period + back_offset)

    # Arc between tangency points (short arc)
    theta1 = math.atan2(p_ordered[0][1] - center[1], p_ordered[0][0] - center[0])
    theta2 = math.atan2(p_ordered[1][1] - center[1], p_ordered[1][0] - center[0])
    dtheta = _wrap_to_pi(theta2 - theta1)
    n_arc = 200
    arc_thetas = theta1 + np.linspace(0.0, dtheta, n_arc)
    arc_x = center[0] + radius * np.cos(arc_thetas)
    arc_y = center[1] + radius * np.sin(arc_thetas)

    # Midpoint along the short arc
    theta_mid = theta1 + 0.5 * dtheta
    mid_pt = np.array([center[0] + radius * math.cos(theta_mid),
                       center[1] + radius * math.sin(theta_mid)])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")

    # Plot sinusoids
    ax.plot(x_left, y_vals, color="#2f6fdd", lw=2.0, label="Left sinusoid (vertical)")
    ax.plot(x_vals, y_back, color="#2d8f3a", lw=2.0, label="Back sinusoid (horizontal)")

    # Plot circle and key points
    circle = plt.Circle((center[0], center[1]), radius, fill=False, ls="--", lw=1.5, color="#444444")
    ax.add_patch(circle)
    ax.scatter([center[0]], [center[1]], color="#111111", s=30, zorder=5, label="Circle center")

    ax.scatter([p_vert[0]], [p_vert[1]], color="#ff8800", s=60, zorder=6, label="Tangent on left")
    ax.scatter([p_horz[0]], [p_horz[1]], color="#e6007a", s=60, zorder=6, label="Tangent on back")

    # Arc and midpoint
    ax.plot(arc_x, arc_y, color="#cc0000", lw=2.5, label="Short arc (tangent)")
    ax.scatter([mid_pt[0]], [mid_pt[1]], color="#cc0000", s=60, zorder=7, label="Arc midpoint")

    # Annotate tangency order
    ax.text(p_ordered[0][0] + 0.25, p_ordered[0][1] + 0.25, "P1", color="#cc0000")
    ax.text(p_ordered[1][0] + 0.25, p_ordered[1][1] + 0.25, "P2", color="#cc0000")

    ax.set_title("Level 19 Back-Left Tangent Circle Debug")
    ax.set_xlabel("x (in)")
    ax.set_ylabel("y (in)")
    ax.set_xlim(-1, pantry_width + 1)
    ax.set_ylim(-1, pantry_depth + 1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    out_path = Path("output") / "back_left_level19_debug.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
