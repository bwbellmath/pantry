#!/usr/bin/env python3
"""
corner_sine_fillet.py

Shelf coordinate system:
  - wall is the x-axis: y = 0
  - front edge is a sine: y(x) = o + a*sin(2*pi*x/p)
  - shelf spans x in [0, L]
  - left side vertical edge is x=0, right side is x=L

We build an "inside" corner rounding arc of radius r that is tangent to:
  1) the vertical side line at the chosen end (x=0 or x=L),
  2) the sine curve y(x).

Additional fixed-center-x constraint (your request):
  - left end:  circle center x_c = r
  - right end: circle center x_c = L - r

Unknowns solved:
  - tangency x on the sine (x_t)
  - circle center y_c
Then:
  - side-line tangency point is (x_end, y_c)
  - vertical segment from (x_end,0) to (x_end, y_c) is added for convenience
  - arc metadata (center, radius, angles, short-arc direction) is reported

Run example:
  python corner_sine_fillet.py --L 29 --r 4 --p 12 --a 1 --o 7 --end left --plot
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
# Root finding (scan brackets + bisection)
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

def find_brackets(f, x0: float, x1: float, *, samples: int = 8000) -> List[Tuple[float, float]]:
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
# Results
# ----------------------------

@dataclass(frozen=True)
class CornerFillet:
    end: str
    L: float
    r: float
    p: float
    a: float
    o: float

    x_end: float                # 0 or L
    center: Tuple[float, float] # (x_c, y_c)

    tangent_side: Tuple[float, float]  # (x_end, y_c) (tangent to vertical line)
    tangent_sine: Tuple[float, float]  # (x_t, y(x_t))

    vertical_segment: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x_end,0),(x_end,y_c))

    theta_side: float
    theta_sine: float
    arc_direction: str          # "CCW" or "CW"
    arc_sweep: float            # signed sweep, magnitude <= pi (short arc)

    dist_err: float
    slope_err: float            # |m_sine - m_circle| at tangency


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
    samples: int = 12000,
    eps_slope: float = 1e-12
) -> CornerFillet:
    """
    Solve for an inside corner fillet at the left or right end.

    Constraints:
      - left:  x_end=0,  x_c = r
      - right: x_end=L,  x_c = L - r
    Unknown:
      - sine tangency x_t in a local window of width 2r from the side
      - circle center y_c

    Tangency condition (eliminating y_c) for vertical-side tangent circle:
      F(x) = (x-x_c)^2 * (1 + m(x)^2) - r^2 * m(x)^2 = 0
      where m(x) = y'(x)

    Then recover y_c from perpendicularity:
      (x-x_c) + (y-y_c)*m = 0  =>  y_c = y + (x-x_c)/m   (m != 0)
    """
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    x_end = 0.0 if end == "left" else L
    x_c = r if end == "left" else (L - r)

    # Tangency point must lie on the circle's x-span near the side:
    # left:  circle spans [0, 2r]
    # right: circle spans [L-2r, L]
    x_min = 0.0 if end == "left" else max(0.0, L - 2.0 * r)
    x_max = min(L, 2.0 * r) if end == "left" else L

    def F(x: float) -> float:
        m = sine_dy_dx(x, p, a)
        dx = x - x_c
        return (dx * dx) * (1.0 + m * m) - (r * r) * (m * m)

    brackets = find_brackets(F, x_min, x_max, samples=samples)
    if not brackets:
        raise RuntimeError(
            f"No tangency root found in [{x_min}, {x_max}]. "
            f"Try different r, o, a, p, or increase --samples."
        )

    roots: List[float] = []
    for lo, hi in brackets:
        rt = bisect_root(F, lo, hi)
        if rt is None:
            continue
        if all(abs(rt - r0) > 1e-7 for r0 in roots):
            roots.append(rt)

    if not roots:
        raise RuntimeError("Brackets found, but solving failed; increase --samples.")

    # Pick the tangency x closest to the side (typical corner-rounding):
    x_t = min(roots) if end == "left" else max(roots)

    y_t = sine_y(x_t, p, a, o)
    m_t = sine_dy_dx(x_t, p, a)
    dx = x_t - x_c

    # Recover y_c
    if abs(m_t) < eps_slope:
        # m ~ 0 => tangency requires dx ~ 0. If so, y_c = y_t - r makes the circle sit "below" the sine.
        if abs(dx) > 1e-6:
            raise RuntimeError("Found near-horizontal slope but dx != 0; numerical instability.")
        y_c = y_t - r
    else:
        y_c = y_t + dx / m_t

    center = (x_c, y_c)
    tangent_side = (x_end, y_c)
    tangent_sine = (x_t, y_t)
    vertical_segment = ((x_end, 0.0), (x_end, y_c))

    # Check circle membership and slope match at the tangency point
    dist = math.hypot(x_t - x_c, y_t - y_c)
    dist_err = abs(dist - r)

    # circle slope at (x_t,y_t): from implicit circle
    # y' = -(x-x_c)/(y-y_c)
    denom = (y_t - y_c)
    if abs(denom) < 1e-12:
        m_circle = float("inf")
    else:
        m_circle = -(x_t - x_c) / denom
    slope_err = abs(m_t - m_circle) if math.isfinite(m_circle) else abs(x_t - x_c)  # crude when vertical

    # Arc metadata (short arc from side tangency to sine tangency)
    theta_side = angle_of_point(x_c, y_c, tangent_side[0], tangent_side[1])
    theta_sine = angle_of_point(x_c, y_c, tangent_sine[0], tangent_sine[1])
    d = wrap_angle_pi(theta_sine - theta_side)
    arc_direction = "CCW" if d >= 0 else "CW"
    arc_sweep = d

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
# Arc sampling (for stitching into polylines)
# ----------------------------

def sample_short_arc(res: CornerFillet, n: int = 240) -> List[Tuple[float, float]]:
    cx, cy = res.center
    r = res.r
    a0 = res.theta_side
    sweep = res.arc_sweep
    pts: List[Tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        ang = a0 + sweep * t
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


# ----------------------------
# Plotting
# ----------------------------

def plot_solution(res: CornerFillet, *, sine_samples: int = 2200) -> None:
    L = res.L
    r = res.r
    p, a, o = res.p, res.a, res.o
    cx, cy = res.center

    xs = [L * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    # Full circle for viz
    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    arc = sample_short_arc(res, n=240)
    arc_x = [u for u, v in arc]
    arc_y = [v for u, v in arc]

    # Vertical segment at end
    (x0, y0), (x1, y1) = res.vertical_segment

    plt.figure()
    plt.plot(xs, ys, label="front edge: sine y(x)")
    plt.axhline(0.0, linestyle="--", label="wall: y=0")
    plt.axvline(res.x_end, linestyle="--", label=f"side: x={res.x_end:g}")

    plt.plot(circ_x, circ_y, label="circle (full)")
    plt.plot(arc_x, arc_y, linewidth=3.0, label="chosen short arc")

    plt.plot([x0, x1], [y0, y1], linewidth=2.5, label="vertical segment to circle")

    # Mark points
    ts = res.tangent_sine
    tv = res.tangent_side
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
    ap.add_argument("--L", type=float, required=True, help="shelf length (domain x in [0,L])")
    ap.add_argument("--r", type=float, required=True, help="corner fillet radius")
    ap.add_argument("--p", type=float, required=True, help="sine period")
    ap.add_argument("--a", type=float, required=True, help="sine amplitude")
    ap.add_argument("--o", type=float, required=True, help="sine vertical offset (above wall)")
    ap.add_argument("--end", choices=["left", "right"], required=True, help="which end corner to round")
    ap.add_argument("--samples", type=int, default=12000, help="bracketing samples")
    ap.add_argument("--plot", action="store_true", help="show plot")
    args = ap.parse_args()

    res = solve_corner_fillet(
        L=args.L, r=args.r, p=args.p, a=args.a, o=args.o,
        end=args.end, samples=args.samples
    )

    print("=== Corner fillet result ===")
    print(f"end            : {res.end}")
    print(f"shelf length L : {res.L}")
    print(f"radius r       : {res.r}")
    print(f"side line      : x = {res.x_end}")
    print(f"center         : {res.center}")
    print(f"tangent_side   : {res.tangent_side}  (circle tangent to vertical side)")
    print(f"tangent_sine   : {res.tangent_sine}  (circle tangent to sine)")
    print(f"vertical_seg   : {res.vertical_segment[0]} -> {res.vertical_segment[1]}")
    print(f"theta_side     : {res.theta_side:.12f} rad")
    print(f"theta_sine     : {res.theta_sine:.12f} rad")
    print(f"arc_direction  : {res.arc_direction}")
    print(f"arc_sweep      : {res.arc_sweep:.12f} rad (short arc, signed)")
    print(f"dist_err       : {res.dist_err:.3e}")
    print(f"slope_err      : {res.slope_err:.3e}")
    print()
    print("Arc construction (unambiguous):")
    print("  start_angle = theta_side")
    print("  end_angle   = theta_side + arc_sweep")
    print("  direction   = CCW if arc_sweep >= 0 else CW")
    print("  Use center, radius, and (start, sweep) to build the short arc segment.")

    if args.plot:
        plot_solution(res)

if __name__ == "__main__":
    main()
