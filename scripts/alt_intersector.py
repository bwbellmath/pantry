#!/usr/bin/env python3
"""
circle_sine_fillet.py

Compute a radius-r circle that is tangent to:
  1) the vertical line x = x_v
  2) the sine curve y(x) = o + a*sin(2*pi*x/p)

Returns the two join points (line tangency and sine tangency) and arc metadata
(center, radius, angles, and CW/CCW direction for the short arc).

Also includes a plotting test harness.

Usage example:
  python circle_sine_fillet.py --r 3 --p 18 --a 0.75 --o 2.0 --side right --xv 29 --plot
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import argparse

# matplotlib only used in the tester/plot path
import matplotlib.pyplot as plt


# ----------------------------
# Core curve definitions
# ----------------------------

def sine_y(x: float, p: float, a: float, o: float) -> float:
    k = 2.0 * math.pi / p
    return o + a * math.sin(k * x)

def sine_dy_dx(x: float, p: float, a: float) -> float:
    k = 2.0 * math.pi / p
    return a * k * math.cos(k * x)

def wrap_angle_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a

def angle_of_point(cx: float, cy: float, px: float, py: float) -> float:
    return math.atan2(py - cy, px - cx)


# ----------------------------
# Root finding helpers
# ----------------------------

def bisect_root(f, lo: float, hi: float, *, max_iter: int = 80, tol: float = 1e-12) -> Optional[float]:
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
            # try shrinking a bit
            m = math.nextafter(m, a)
            fm = f(m)
            if not math.isfinite(fm):
                return None

        if abs(fm) < tol or (b - a) < tol:
            return m

        # keep the bracket
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)

def find_bracketed_roots(
    f,
    x0: float,
    x1: float,
    *,
    samples: int = 4000,
    near_zero: float = 1e-8
) -> List[Tuple[float, float]]:
    """
    Scan [x0, x1] and return a list of (lo, hi) brackets where f changes sign,
    plus tiny brackets around points where |f| is very small.
    """
    if x1 < x0:
        x0, x1 = x1, x0

    brackets: List[Tuple[float, float]] = []
    xs = [x0 + (x1 - x0) * i / samples for i in range(samples + 1)]
    fs = []
    for x in xs:
        fx = f(x)
        fs.append(fx)

    # sign-change brackets
    for i in range(samples):
        f0, f1 = fs[i], fs[i + 1]
        if not (math.isfinite(f0) and math.isfinite(f1)):
            continue
        if f0 == 0.0:
            # make a tiny bracket around the exact sample
            eps = (x1 - x0) / samples
            brackets.append((max(x0, xs[i] - eps), min(x1, xs[i] + eps)))
            continue
        if f0 * f1 < 0.0:
            brackets.append((xs[i], xs[i + 1]))

    # near-zero points (can help if it just kisses the axis numerically)
    for i in range(1, samples):
        fprev, fcur, fnext = fs[i - 1], fs[i], fs[i + 1]
        if not (math.isfinite(fprev) and math.isfinite(fcur) and math.isfinite(fnext)):
            continue
        if abs(fcur) < near_zero and abs(fcur) <= abs(fprev) and abs(fcur) <= abs(fnext):
            eps = (x1 - x0) / samples
            brackets.append((max(x0, xs[i] - eps), min(x1, xs[i] + eps)))

    # de-dup / merge overlaps
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
# Result structure
# ----------------------------

@dataclass(frozen=True)
class FilletResult:
    side: str
    r: float
    p: float
    a: float
    o: float
    x_v: float

    center: Tuple[float, float]          # (x_c, y_c)
    tangent_line: Tuple[float, float]    # (x_v, y_c)
    tangent_sine: Tuple[float, float]    # (x_t, y(x_t))

    theta_line: float                    # angle at tangent_line around center
    theta_sine: float                    # angle at tangent_sine around center

    # short-arc choice
    arc_direction: str                   # "CCW" or "CW"
    arc_sweep: float                     # signed sweep angle (radians) in chosen direction, magnitude <= pi

    # diagnostics
    dist_err: float                      # |distance(center, tangent_sine) - r|
    perp_err: float                      # |dot(radius_vec, tangent_vec)|


# ----------------------------
# Main solver
# ----------------------------

def solve_circle_fillet(
    *,
    r: float,
    p: float,
    a: float,
    o: float,
    side: str,
    x_v: float,
    samples: int = 6000,
    prefer_near_line: bool = True,
    eps_slope: float = 1e-12
) -> FilletResult:
    """
    Solve for a circle of radius r tangent to:
      - vertical line x = x_v
      - sine y = o + a*sin(2*pi*x/p)

    side: "left" or "right" indicating which side of the vertical line the circle bulges into.
    """

    side = side.lower().strip()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    # Circle center x-coordinate fixed by tangency to vertical line.
    # Right: center is r to the left of the line. Left: center is r to the right.
    x_c = (x_v - r) if side == "right" else (x_v + r)

    # Search interval where sine tangency point must lie on that circle near the line:
    # On the right side, circle spans [x_v-2r, x_v]. On the left, [x_v, x_v+2r].
    x_min = x_v - 2.0 * r if side == "right" else x_v
    x_max = x_v if side == "right" else x_v + 2.0 * r

    def F(x: float) -> float:
        m = sine_dy_dx(x, p, a)
        dx = x - x_c
        # F(x) = dx^2*(1+m^2) - r^2*m^2
        return (dx * dx) * (1.0 + m * m) - (r * r) * (m * m)

    brackets = find_bracketed_roots(F, x_min, x_max, samples=samples)
    if not brackets:
        raise RuntimeError(
            "No root brackets found in the expected interval. "
            "Try increasing samples or check if a tangent fillet exists for these parameters."
        )

    roots: List[float] = []
    for lo, hi in brackets:
        rt = bisect_root(F, lo, hi)
        if rt is None:
            continue
        # de-dup close roots
        if all(abs(rt - r0) > 1e-7 for r0 in roots):
            roots.append(rt)

    if not roots:
        raise RuntimeError(
            "Brackets found, but root solving failed. "
            "Try adjusting sampling density or tolerances."
        )

    # Choose which tangency point to use.
    # Typical CAD fillet wants the join nearest to the vertical line (smallest modification).
    if prefer_near_line:
        # Right side: prefer largest x (closest to x_v). Left side: prefer smallest x.
        x_t = max(roots) if side == "right" else min(roots)
    else:
        x_t = roots[0]

    y_t = sine_y(x_t, p, a, o)
    m = sine_dy_dx(x_t, p, a)

    # Compute y_c from perpendicularity condition:
    # y_c = y(x) + (x - x_c)/m , provided m != 0
    if abs(m) < eps_slope:
        # Special case: nearly horizontal sine tangent.
        # Then perpendicularity implies x_t == x_c (or very close). Center is vertically offset by ±r.
        # Pick the center that yields a "short arc" to the line point on the same y-level.
        # We'll choose y_c = y_t + r by default.
        y_c = y_t + r
    else:
        y_c = y_t + (x_t - x_c) / m

    # Points:
    center = (x_c, y_c)
    tangent_line = (x_v, y_c)
    tangent_sine = (x_t, y_t)

    # Diagnostics for correctness:
    dist = math.hypot(x_t - x_c, y_t - y_c)
    dist_err = abs(dist - r)

    # Perpendicularity check: radius_vec dot tangent_vec should be 0
    # tangent_vec = (1, m)
    perp_err = abs((x_t - x_c) * 1.0 + (y_t - y_c) * m)

    # Arc angles:
    theta_line = angle_of_point(x_c, y_c, tangent_line[0], tangent_line[1])
    theta_sine = angle_of_point(x_c, y_c, tangent_sine[0], tangent_sine[1])

    # Choose the short arc direction.
    # Compute signed delta in (-pi, pi]:
    d = wrap_angle_pi(theta_sine - theta_line)
    # If d is positive, CCW from line->sine is short. If negative, CW is short.
    arc_direction = "CCW" if d >= 0 else "CW"
    arc_sweep = d  # signed, magnitude <= pi

    return FilletResult(
        side=side,
        r=r, p=p, a=a, o=o, x_v=x_v,
        center=center,
        tangent_line=tangent_line,
        tangent_sine=tangent_sine,
        theta_line=theta_line,
        theta_sine=theta_sine,
        arc_direction=arc_direction,
        arc_sweep=arc_sweep,
        dist_err=dist_err,
        perp_err=perp_err
    )


# ----------------------------
# Geometry helper: sample the chosen short arc
# ----------------------------

def sample_short_arc(result: FilletResult, n: int = 120) -> List[Tuple[float, float]]:
    """
    Return polyline points along the chosen short arc from tangent_line to tangent_sine.
    """
    x_c, y_c = result.center
    r = result.r
    a0 = result.theta_line
    sweep = result.arc_sweep

    pts: List[Tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        ang = a0 + sweep * t
        pts.append((x_c + r * math.cos(ang), y_c + r * math.sin(ang)))
    return pts


# ----------------------------
# Tester / plot
# ----------------------------

def plot_solution(result: FilletResult, *, margin: float = 1.2, sine_samples: int = 2000) -> None:
    x_c, y_c = result.center
    x_v = result.x_v
    r = result.r
    p, a, o = result.p, result.a, result.o

    # Plot window around the circle and vertical line
    if result.side == "right":
        x0 = x_v - 2.0 * r - margin * r
        x1 = x_v + margin * r
    else:
        x0 = x_v - margin * r
        x1 = x_v + 2.0 * r + margin * r

    xs = [x0 + (x1 - x0) * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    # Circle (full, for visualization)
    ths = [2.0 * math.pi * i / 800 for i in range(801)]
    cx = [x_c + r * math.cos(t) for t in ths]
    cy = [y_c + r * math.sin(t) for t in ths]

    # Arc polyline (short arc)
    arc_pts = sample_short_arc(result, n=200)
    ax = [u for u, v in arc_pts]
    ay = [v for u, v in arc_pts]

    plt.figure()
    plt.plot(xs, ys, label="sine y(x)")
    plt.plot(cx, cy, label="circle (full)")
    plt.plot(ax, ay, linewidth=3.0, label="chosen short arc")

    # Vertical line
    plt.axvline(x=x_v, linestyle="--", label=f"vertical line x={x_v:g}")

    # Mark points
    tl = result.tangent_line
    ts = result.tangent_sine
    plt.scatter([x_c], [y_c], marker="x", s=80, label="center")
    plt.scatter([tl[0], ts[0]], [tl[1], ts[1]], s=60, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"Fillet (side={result.side})  dir={result.arc_direction}  sweep={result.arc_sweep:.4f} rad\n"
        f"dist_err={result.dist_err:.2e}, perp_err={result.perp_err:.2e}"
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, required=True, help="circle radius")
    ap.add_argument("--p", type=float, required=True, help="sine period")
    ap.add_argument("--a", type=float, required=True, help="sine amplitude")
    ap.add_argument("--o", type=float, required=True, help="sine vertical offset")
    ap.add_argument("--side", type=str, choices=["left", "right"], required=True, help="which side of the vertical line")
    ap.add_argument("--xv", type=float, required=True, help="vertical line position x = xv")
    ap.add_argument("--plot", action="store_true", help="show matplotlib plot")
    ap.add_argument("--samples", type=int, default=6000, help="sampling density for bracketing roots")
    args = ap.parse_args()

    res = solve_circle_fillet(
        r=args.r, p=args.p, a=args.a, o=args.o,
        side=args.side, x_v=args.xv,
        samples=args.samples
    )

    print("=== Fillet result ===")
    print(f"side          : {res.side}")
    print(f"r             : {res.r}")
    print(f"vertical line : x = {res.x_v}")
    print(f"center        : {res.center}")
    print(f"tangent_line  : {res.tangent_line}   (circle tangent to vertical line)")
    print(f"tangent_sine  : {res.tangent_sine}   (circle tangent to sine)")
    print(f"theta_line    : {res.theta_line:.12f} rad")
    print(f"theta_sine    : {res.theta_sine:.12f} rad")
    print(f"arc_direction : {res.arc_direction}")
    print(f"arc_sweep     : {res.arc_sweep:.12f} rad   (short arc, signed)")
    print(f"dist_err      : {res.dist_err:.3e}")
    print(f"perp_err      : {res.perp_err:.3e}")
    print()
    print("Arc construction hints (unambiguous):")
    print("  Center (cx,cy), radius r, start angle = theta_line, end angle = theta_line + arc_sweep.")
    print("  Direction is CCW if arc_sweep >= 0 else CW. Magnitude <= pi gives the short arc.")
    print()

    if args.plot:
        plot_solution(res)

if __name__ == "__main__":
    main()
