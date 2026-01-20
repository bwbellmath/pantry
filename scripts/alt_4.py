#!/usr/bin/env python3
"""
corner_sine_fillet.py  (UPDATED: choose interior arc, enforce center-y feasibility)

Shelf coordinate system:
  - wall is y = 0
  - front edge is sine: y(x) = o + a*sin(2*pi*x/p)
  - shelf spans x in [0, L]
  - vertical sides at x=0 (left) and x=L (right)

Corner rounding arc of radius r that is tangent to:
  1) the vertical side line at the chosen end (x=0 or x=L),
  2) the sine curve y(x).

Fixed center-x constraint (inside the shelf):
  - left end:  x_c = r
  - right end: x_c = L - r

Unknowns solved:
  - sine tangency x_t
  - circle center y_c

Also outputs:
  - side vertical segment: (x_end, 0) -> (x_end, y_c)
  - arc metadata that specifically selects the *interior* arc (contained in shelf)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import argparse
import matplotlib.pyplot as plt


# ----------------------------
# Sine curve (no phase shift)
# ----------------------------

def sine_y(x: float, p: float, a: float, o: float) -> float:
    k = 2.0 * math.pi / p
    return o + a * math.sin(k * x)

def sine_dy_dx(x: float, p: float, a: float) -> float:
    k = 2.0 * math.pi / p
    return a * k * math.cos(k * x)

def wrap_angle_pi(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return theta

def angle_of_point(cx: float, cy: float, px: float, py: float) -> float:
    return math.atan2(py - cy, px - cx)


# ----------------------------
# Root finding (scan + bisection)
# ----------------------------

def bisect_root(f, lo: float, hi: float, *, max_iter: int = 90, tol: float = 1e-13) -> Optional[float]:
    flo = f(lo)
    fhi = f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        return None
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return None

    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if not math.isfinite(fm):
            return None
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

def find_brackets(f, x0: float, x1: float, *, samples: int = 12000) -> List[Tuple[float, float]]:
    if x1 < x0:
        x0, x1 = x1, x0
    xs = [x0 + (x1 - x0) * i / samples for i in range(samples + 1)]
    fs = [f(x) for x in xs]

    brackets: List[Tuple[float, float]] = []
    for i in range(samples):
        f0, f1 = fs[i], fs[i + 1]
        if not (math.isfinite(f0) and math.isfinite(f1)):
            continue
        if f0 == 0.0:
            eps = (x1 - x0) / samples
            brackets.append((max(x0, xs[i] - eps), min(x1, xs[i] + eps)))
        elif f0 * f1 < 0.0:
            brackets.append((xs[i], xs[i + 1]))

    # merge overlaps
    brackets.sort()
    merged: List[Tuple[float, float]] = []
    for lo, hi in brackets:
        if not merged:
            merged.append((lo, hi))
        else:
            plo, phi = merged[-1]
            if lo <= phi:
                merged[-1] = (plo, max(phi, hi))
            else:
                merged.append((lo, hi))
    return merged


# ----------------------------
# Result
# ----------------------------

@dataclass(frozen=True)
class CornerFillet:
    end: str
    L: float
    r: float
    p: float
    a: float
    o: float

    x_end: float
    center: Tuple[float, float]          # (x_c, y_c)
    tangent_side: Tuple[float, float]    # (x_end, y_c)
    tangent_sine: Tuple[float, float]    # (x_t, y(x_t))
    vertical_segment: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x_end,0),(x_end,y_c))

    theta_side: float
    theta_sine: float
    arc_direction: str                   # "CCW" or "CW"
    arc_sweep: float                     # signed sweep (chosen arc), can be long or short

    dist_err: float
    slope_err: float                     # |m_sine - m_circle|


# ----------------------------
# Arc sampling + containment test
# ----------------------------

def point_on_arc(cx: float, cy: float, r: float, theta0: float, sweep: float, t: float) -> Tuple[float, float]:
    ang = theta0 + sweep * t
    return (cx + r * math.cos(ang), cy + r * math.sin(ang))

def arc_is_inside_shelf(
    *,
    cx: float, cy: float, r: float,
    theta0: float, sweep: float,
    L: float, p: float, a: float, o: float,
    samples: int = 40,
    eps: float = 1e-9
) -> bool:
    """
    Check if points along the arc satisfy 0 <= y <= sine_y(x).
    We also require x to remain within [0, L] (within numeric eps).
    """
    for i in range(1, samples):  # skip endpoints (they're on boundary by construction)
        t = i / samples
        x, y = point_on_arc(cx, cy, r, theta0, sweep, t)
        if x < -eps or x > L + eps:
            return False
        if y < -eps:
            return False
        yb = sine_y(x, p, a, o)
        if y > yb + eps:
            return False
    return True


# ----------------------------
# Solver
# ----------------------------

def solve_corner_fillet(
    *,
    L: float,
    r: float,
    p: float,
    a: float,
    o: float,
    end: str,
    samples: int = 15000,
    eps_slope: float = 1e-12
) -> CornerFillet:
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    x_end = 0.0 if end == "left" else L
    x_c = r if end == "left" else (L - r)

    # Tangency x must lie within the circle's x-span near the side.
    x_min = 0.0 if end == "left" else max(0.0, L - 2.0 * r)
    x_max = min(L, 2.0 * r) if end == "left" else L

    def F(x: float) -> float:
        # Derived from tangency to sine + circle with fixed x_c and unknown y_c
        m = sine_dy_dx(x, p, a)
        dx = x - x_c
        return (dx * dx) * (1.0 + m * m) - (r * r) * (m * m)

    brackets = find_brackets(F, x_min, x_max, samples=samples)
    if not brackets:
        raise RuntimeError(
            f"No tangency root found in [{x_min}, {x_max}]. "
            f"Try smaller r or different (a,o,p), or increase --samples."
        )

    # Feasibility bounds for *inside shelf*
    y_end = sine_y(x_end, p, a, o)
    y_c_low = r
    y_c_high = min(o + abs(a) - r, y_end)  # your requested bound plus the side-height constraint
    if y_c_high < y_c_low:
        raise RuntimeError(
            f"Radius r={r} cannot fit inside shelf at the {end} end: "
            f"need y_c in [{y_c_low:.6g}, {y_c_high:.6g}], impossible. "
            f"Use a smaller radius."
        )

    candidates: List[Tuple[float, float]] = []  # (x_t, y_c)

    for lo, hi in brackets:
        x_t = bisect_root(F, lo, hi)
        if x_t is None:
            continue

        y_t = sine_y(x_t, p, a, o)
        m_t = sine_dy_dx(x_t, p, a)
        dx = x_t - x_c

        if abs(m_t) < eps_slope:
            # near-horizontal tangent: y_c would blow up unless dx≈0
            continue

        y_c = y_t + dx / m_t

        if not math.isfinite(y_c):
            continue

        # Enforce inside-shelf center bounds
        if y_c < y_c_low - 1e-9 or y_c > y_c_high + 1e-9:
            continue

        # de-dup x roots
        if all(abs(x_t - xt0) > 1e-7 for xt0, _ in candidates):
            candidates.append((x_t, y_c))

    if not candidates:
        raise RuntimeError(
            f"Found tangency roots, but none yield an interior circle. "
            f"This usually means r is too large for the local shelf geometry at the {end} end. "
            f"Try a smaller radius."
        )

    # Choose candidate closest to the side (typical corner rounding)
    x_t, y_c = (min(candidates, key=lambda t: t[0]) if end == "left"
                else max(candidates, key=lambda t: t[0]))

    y_t = sine_y(x_t, p, a, o)
    m_t = sine_dy_dx(x_t, p, a)

    center = (x_c, y_c)
    tangent_side = (x_end, y_c)
    tangent_sine = (x_t, y_t)
    vertical_segment = ((x_end, 0.0), (x_end, y_c))

    # Diagnostics
    dist = math.hypot(x_t - x_c, y_t - y_c)
    dist_err = abs(dist - r)

    denom = (y_t - y_c)
    m_circle = float("inf") if abs(denom) < 1e-12 else -(x_t - x_c) / denom
    slope_err = abs(m_t - m_circle) if math.isfinite(m_circle) else abs(x_t - x_c)

    # Arc selection:
    theta_side = angle_of_point(x_c, y_c, tangent_side[0], tangent_side[1])
    theta_sine = angle_of_point(x_c, y_c, tangent_sine[0], tangent_sine[1])

    d_short = wrap_angle_pi(theta_sine - theta_side)
    # long sweep: go the other way around
    d_long = d_short - 2.0 * math.pi if d_short > 0 else d_short + 2.0 * math.pi

    short_ok = arc_is_inside_shelf(
        cx=x_c, cy=y_c, r=r, theta0=theta_side, sweep=d_short,
        L=L, p=p, a=a, o=o
    )
    long_ok = arc_is_inside_shelf(
        cx=x_c, cy=y_c, r=r, theta0=theta_side, sweep=d_long,
        L=L, p=p, a=a, o=o
    )

    if short_ok and not long_ok:
        arc_sweep = d_short
    elif long_ok and not short_ok:
        arc_sweep = d_long
    elif short_ok and long_ok:
        # rare/numerical: prefer shorter magnitude
        arc_sweep = d_short if abs(d_short) <= abs(d_long) else d_long
    else:
        raise RuntimeError(
            "Solved tangency, but neither arc lies fully inside the shelf. "
            "This likely indicates the radius is too large, or the shelf sine is too low/steep near the end."
        )

    arc_direction = "CCW" if arc_sweep >= 0 else "CW"

    return CornerFillet(
        end=end, L=L, r=r, p=p, a=a, o=o,
        x_end=x_end, center=center,
        tangent_side=tangent_side, tangent_sine=tangent_sine,
        vertical_segment=vertical_segment,
        theta_side=theta_side, theta_sine=theta_sine,
        arc_direction=arc_direction, arc_sweep=arc_sweep,
        dist_err=dist_err, slope_err=slope_err
    )


# ----------------------------
# Plotting
# ----------------------------

def sample_arc(res: CornerFillet, n: int = 240) -> List[Tuple[float, float]]:
    cx, cy = res.center
    pts: List[Tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        pts.append(point_on_arc(cx, cy, res.r, res.theta_side, res.arc_sweep, t))
    return pts

def plot_solution(res: CornerFillet, *, sine_samples: int = 2200) -> None:
    L = res.L
    r = res.r
    p, a, o = res.p, res.a, res.o
    cx, cy = res.center

    xs = [L * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    arc = sample_arc(res, n=260)
    arc_x = [u for u, v in arc]
    arc_y = [v for u, v in arc]

    (x0, y0), (x1, y1) = res.vertical_segment

    plt.figure()
    plt.plot(xs, ys, label="front edge: sine y(x)")
    plt.axhline(0.0, linestyle="--", label="wall: y=0")
    plt.axvline(res.x_end, linestyle="--", label=f"side: x={res.x_end:g}")

    plt.plot(circ_x, circ_y, alpha=0.35, label="circle (full)")
    plt.plot(arc_x, arc_y, linewidth=3.0, label="chosen interior arc")

    plt.plot([x0, x1], [y0, y1], linewidth=2.5, label="vertical segment to circle")

    tv = res.tangent_side
    ts = res.tangent_sine
    plt.scatter([cx], [cy], marker="x", s=90, label="center")
    plt.scatter([tv[0], ts[0]], [tv[1], ts[1]], s=70, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"Corner fillet end={res.end}  dir={res.arc_direction}  sweep={res.arc_sweep:.4f} rad\n"
        f"dist_err={res.dist_err:.2e}, slope_err={res.slope_err:.2e}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True, help="shelf length (x in [0,L])")
    ap.add_argument("--r", type=float, required=True, help="corner radius")
    ap.add_argument("--p", type=float, required=True, help="sine period")
    ap.add_argument("--a", type=float, required=True, help="sine amplitude")
    ap.add_argument("--o", type=float, required=True, help="sine vertical offset above wall")
    ap.add_argument("--end", choices=["left", "right"], required=True, help="which end corner to round")
    ap.add_argument("--samples", type=int, default=15000, help="root bracketing samples")
    ap.add_argument("--plot", action="store_true", help="plot the result")
    args = ap.parse_args()

    res = solve_corner_fillet(L=args.L, r=args.r, p=args.p, a=args.a, o=args.o,
                             end=args.end, samples=args.samples)

    print("=== Corner fillet (INTERIOR) result ===")
    print(f"end            : {res.end}")
    print(f"side line      : x = {res.x_end}")
    print(f"center         : {res.center}")
    print(f"tangent_side   : {res.tangent_side}")
    print(f"tangent_sine   : {res.tangent_sine}")
    print(f"vertical_seg   : {res.vertical_segment[0]} -> {res.vertical_segment[1]}")
    print(f"theta_side     : {res.theta_side:.12f} rad")
    print(f"theta_sine     : {res.theta_sine:.12f} rad")
    print(f"arc_direction  : {res.arc_direction}")
    print(f"arc_sweep      : {res.arc_sweep:.12f} rad  (chosen interior arc)")
    print(f"dist_err       : {res.dist_err:.3e}")
    print(f"slope_err      : {res.slope_err:.3e}")

    if args.plot:
        plot_solution(res)

if __name__ == "__main__":
    main()
