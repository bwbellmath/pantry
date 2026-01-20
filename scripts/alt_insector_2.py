#!/usr/bin/env python3
"""
wall_sine_fillet.py

Compute a radius-r circle that is tangent to:
  1) the wall y = 0 (x-axis), from above
  2) the sine curve y(x) = o + a*sin(2*pi*x/p)

Then choose the "left" or "right" solution within x in [0, L]
by picking the tangency point closest to x=0 or x=L.

Outputs:
  - circle center
  - wall tangency point
  - sine tangency point
  - arc metadata: start angle, sweep (signed, short arc), direction

Also includes a plotting tester.

Example:
  python wall_sine_fillet.py --r 0.75 --p 12 --a 1 --o 2.5 --L 29 --end left --plot
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import argparse
import matplotlib.pyplot as plt


# ----------------------------
# Sine curve
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
# Root finding (bracket scan + bisection)
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
            return None

        if abs(fm) < tol or (b - a) < tol:
            return m

        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    return 0.5 * (a + b)

def find_bracketed_roots(f, x0: float, x1: float, *, samples: int = 6000) -> List[Tuple[float, float]]:
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
            # tiny bracket around exact sample
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
# Result structure
# ----------------------------

@dataclass(frozen=True)
class FilletResult:
    r: float
    p: float
    a: float
    o: float
    L: float
    end: str  # "left" or "right"

    center: Tuple[float, float]         # (c, r)
    tangent_wall: Tuple[float, float]   # (c, 0)
    tangent_sine: Tuple[float, float]   # (x_t, y(x_t))

    theta_wall: float
    theta_sine: float

    arc_direction: str                  # "CCW" or "CW"
    arc_sweep: float                    # signed short sweep in [-pi, pi]

    dist_err: float
    perp_err: float


# ----------------------------
# Main solver
# ----------------------------

def solve_wall_sine_fillet(
    *,
    r: float,
    p: float,
    a: float,
    o: float,
    L: float,
    end: str,
    samples: int = 8000
) -> FilletResult:
    """
    Find a circle of radius r tangent to wall y=0 and sine y(x) (above the wall),
    picking either the leftmost or rightmost solution over x in [0, L].

    end:
      - "left": choose tangency x closest to 0
      - "right": choose tangency x closest to L
    """
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    # Scalar equation in x derived from tangency + wall constraint y_c=r:
    # G(x) = (y-r)^2 * (y')^2 - y*(2r - y) = 0
    def G(x: float) -> float:
        y = sine_y(x, p, a, o)
        yp = sine_dy_dx(x, p, a)
        return (y - r) * (y - r) * (yp * yp) - y * (2.0 * r - y)

    brackets = find_bracketed_roots(G, 0.0, L, samples=samples)
    if not brackets:
        raise RuntimeError(
            "No roots found in [0, L]. This likely means no radius-r circle can be tangent "
            "to both y=0 and the sine curve with these parameters (or sampling too low)."
        )

    roots: List[float] = []
    for lo, hi in brackets:
        rt = bisect_root(G, lo, hi)
        if rt is None:
            continue
        if all(abs(rt - r0) > 1e-7 for r0 in roots):
            roots.append(rt)

    if not roots:
        raise RuntimeError("Root bracketing succeeded but solving failed; try higher --samples.")

    # Pick the root nearest the requested end
    if end == "left":
        x_t = min(roots, key=lambda x: abs(x - 0.0))
    else:
        x_t = min(roots, key=lambda x: abs(x - L))

    y_t = sine_y(x_t, p, a, o)
    yp = sine_dy_dx(x_t, p, a)

    # Center x-coordinate in closed form:
    # c = x + (y - r)*y'
    c = x_t + (y_t - r) * yp
    center = (c, r)

    tangent_wall = (c, 0.0)
    tangent_sine = (x_t, y_t)

    # Diagnostics:
    dist = math.hypot(x_t - c, y_t - r)
    dist_err = abs(dist - r)

    # Perpendicularity: radius dot tangent = 0, tangent=(1, y')
    perp_err = abs((x_t - c) * 1.0 + (y_t - r) * yp)

    # Angles for arc metadata
    theta_wall = angle_of_point(c, r, tangent_wall[0], tangent_wall[1])
    theta_sine = angle_of_point(c, r, tangent_sine[0], tangent_sine[1])

    d = wrap_angle_pi(theta_sine - theta_wall)
    arc_direction = "CCW" if d >= 0 else "CW"
    arc_sweep = d

    return FilletResult(
        r=r, p=p, a=a, o=o, L=L, end=end,
        center=center,
        tangent_wall=tangent_wall,
        tangent_sine=tangent_sine,
        theta_wall=theta_wall,
        theta_sine=theta_sine,
        arc_direction=arc_direction,
        arc_sweep=arc_sweep,
        dist_err=dist_err,
        perp_err=perp_err
    )


# ----------------------------
# Sampling helper for stitching into geometry
# ----------------------------

def sample_short_arc(res: FilletResult, n: int = 200) -> List[Tuple[float, float]]:
    c, cy = res.center
    r = res.r
    a0 = res.theta_wall
    sweep = res.arc_sweep
    pts: List[Tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        ang = a0 + sweep * t
        pts.append((c + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


# ----------------------------
# Plot tester
# ----------------------------

def plot_solution(res: FilletResult, *, sine_samples: int = 2000) -> None:
    r = res.r
    p, a, o = res.p, res.a, res.o
    L = res.L
    c, cy = res.center

    xs = [L * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    # Circle (full)
    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    cx = [c + r * math.cos(t) for t in ths]
    cyy = [cy + r * math.sin(t) for t in ths]

    # Chosen short arc
    arc = sample_short_arc(res, n=250)
    ax = [u for u, v in arc]
    ay = [v for u, v in arc]

    plt.figure()
    plt.plot(xs, ys, label="front edge: sine y(x)")
    plt.axhline(0.0, linestyle="--", label="wall: y=0")
    plt.plot(cx, cyy, label="circle (full)")
    plt.plot(ax, ay, linewidth=3.0, label="chosen short arc")

    # Mark tangencies + center
    tw = res.tangent_wall
    ts = res.tangent_sine
    plt.scatter([c], [cy], marker="x", s=80, label="center")
    plt.scatter([tw[0], ts[0]], [tw[1], ts[1]], s=70, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"Wall–sine fillet (end={res.end}) dir={res.arc_direction} sweep={res.arc_sweep:.4f} rad\n"
        f"dist_err={res.dist_err:.2e}, perp_err={res.perp_err:.2e}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, required=True, help="fillet radius")
    ap.add_argument("--p", type=float, required=True, help="sine period")
    ap.add_argument("--a", type=float, required=True, help="sine amplitude")
    ap.add_argument("--o", type=float, required=True, help="sine vertical offset (above wall)")
    ap.add_argument("--L", type=float, required=True, help="shelf length domain x in [0, L]")
    ap.add_argument("--end", choices=["left", "right"], required=True, help="which end to choose")
    ap.add_argument("--samples", type=int, default=8000, help="root bracketing samples")
    ap.add_argument("--plot", action="store_true", help="plot the result")
    args = ap.parse_args()

    res = solve_wall_sine_fillet(
        r=args.r, p=args.p, a=args.a, o=args.o, L=args.L,
        end=args.end, samples=args.samples
    )

    print("=== Wall–sine fillet result ===")
    print(f"end          : {res.end}")
    print(f"r            : {res.r}")
    print(f"domain       : x in [0, {res.L}]")
    print(f"center       : {res.center}")
    print(f"tangent_wall : {res.tangent_wall}   (circle tangent to y=0)")
    print(f"tangent_sine : {res.tangent_sine}   (circle tangent to sine)")
    print(f"theta_wall   : {res.theta_wall:.12f} rad")
    print(f"theta_sine   : {res.theta_sine:.12f} rad")
    print(f"arc_dir      : {res.arc_direction}")
    print(f"arc_sweep    : {res.arc_sweep:.12f} rad (short arc, signed)")
    print(f"dist_err     : {res.dist_err:.3e}")
    print(f"perp_err     : {res.perp_err:.3e}")
    print()
    print("Arc construction (unambiguous):")
    print("  start_angle = theta_wall")
    print("  end_angle   = theta_wall + arc_sweep")
    print("  direction   = CCW if arc_sweep >= 0 else CW")
    print("  short arc guaranteed because arc_sweep is wrapped to [-pi, pi].")

    if args.plot:
        plot_solution(res)


if __name__ == "__main__":
    main()
    
