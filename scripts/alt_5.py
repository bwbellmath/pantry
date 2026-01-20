#!/usr/bin/env python3
"""
corner_sine_fillet_height.py

Shelf:
  - wall is y=0
  - front edge is sine: y(x) = o + a*sin(2*pi*x/p)
  - domain x in [0, L]
  - left side x=0, right side x=L

Goal:
  Find a circle (fillet) that is:
    - tangent to the vertical side at the chosen end (x=0 or x=L)
    - tangent to the sine curve
    - has its side tangency point at exact height h above the wall

Key constraint:
  side tangency is at (x_end, y_c), so forcing that height means y_c = h.

We solve for x_t (sine tangency x) using a scalar equation H(x)=0, then
recover r directly. Finally choose the *interior* arc by sampling.

Example (your new numbers):
  offset o=5", height h=3"
  python corner_sine_fillet_height.py --L 29 --p 12 --a 1 --o 5 --h 3 --end left --plot
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

def wrap_angle_pi(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return theta

def angle_of_point(cx: float, cy: float, px: float, py: float) -> float:
    return math.atan2(py - cy, px - cx)


# ----------------------------
# Root finding
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

def find_brackets(f, x0: float, x1: float, *, samples: int = 20000) -> List[Tuple[float, float]]:
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
# Arc sampling + containment
# ----------------------------

def point_on_arc(cx: float, cy: float, r: float, theta0: float, sweep: float, t: float) -> Tuple[float, float]:
    ang = theta0 + sweep * t
    return (cx + r * math.cos(ang), cy + r * math.sin(ang))

def arc_is_inside_shelf(
    *,
    cx: float, cy: float, r: float,
    theta0: float, sweep: float,
    L: float, p: float, a: float, o: float,
    samples: int = 60,
    eps: float = 1e-9
) -> bool:
    for i in range(1, samples):
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
# Result
# ----------------------------

@dataclass(frozen=True)
class CornerFilletHeight:
    end: str
    L: float
    p: float
    a: float
    o: float
    h: float

    r: float
    x_end: float
    center: Tuple[float, float]

    tangent_side: Tuple[float, float]
    tangent_sine: Tuple[float, float]

    vertical_segment: Tuple[Tuple[float, float], Tuple[float, float]]

    theta_side: float
    theta_sine: float
    arc_direction: str
    arc_sweep: float

    dist_err: float
    slope_err: float


# ----------------------------
# Solver
# ----------------------------

def solve_corner_fillet_with_height(
    *,
    L: float,
    p: float,
    a: float,
    o: float,
    h: float,
    end: str,
    samples: int = 25000,
    eps_slope: float = 1e-12
) -> CornerFilletHeight:
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    if h < 0:
        raise ValueError("h must be >= 0")

    # Your “avoid unsolvable” guidance:
    # ensure h is below the minimum possible shelf height o - |a| (global)
    y_min_global = o - abs(a)
    if h > y_min_global + 1e-9:
        raise RuntimeError(
            f"Requested height h={h} exceeds o-|a|={y_min_global}. "
            f"Choose h <= {y_min_global} to avoid guaranteed unsolvability."
        )

    x_end = 0.0 if end == "left" else L
    y_end = sine_y(x_end, p, a, o)
    if h > y_end + 1e-9:
        raise RuntimeError(
            f"Requested height h={h} is above the shelf edge at the {end} end "
            f"(y_end={y_end}). Choose a smaller h."
        )

    # Since y_c=h and we need circle bottom >= 0: r <= h (otherwise it dips below wall).
    # Also, near the end the circle can’t extend further than 2r horizontally from the side,
    # so x_t must lie in [0, 2r] (left) or [L-2r, L] (right). With r<=h, we can safely scan:
    x_min = 0.0 if end == "left" else max(0.0, L - 2.0 * h)
    x_max = min(L, 2.0 * h) if end == "left" else L

    def H(x: float) -> float:
        y = sine_y(x, p, a, o)
        m = sine_dy_dx(x, p, a)
        if end == "left":
            r_expr = x + (y - h) * m
        else:
            r_expr = L - x - (y - h) * m
        return (y - h) * (y - h) * (1.0 + m * m) - (r_expr * r_expr)

    brackets = find_brackets(H, x_min, x_max, samples=samples)
    if not brackets:
        raise RuntimeError(
            f"No solution bracket found for x in [{x_min}, {x_max}]. "
            f"Try a different h (or smaller amplitude/period changes), or increase --samples."
        )

    candidates: List[Tuple[float, float]] = []  # (x_t, r)

    for lo, hi in brackets:
        x_t = bisect_root(H, lo, hi)
        if x_t is None:
            continue

        y = sine_y(x_t, p, a, o)
        m = sine_dy_dx(x_t, p, a)

        if abs(m) < eps_slope:
            continue

        r = (x_t + (y - h) * m) if end == "left" else (L - x_t - (y - h) * m)
        if not math.isfinite(r):
            continue
        if r <= 1e-9:
            continue
        if r > h + 1e-9:
            # would dip below wall since center is at y=h
            continue

        candidates.append((x_t, r))

    if not candidates:
        raise RuntimeError(
            "Found roots, but none yield a feasible radius (r>0 and r<=h). "
            "Try a different height h (or accept a smaller arc by lowering h)."
        )

    # Choose solution closest to the side (typical corner rounding)
    x_t, r = (min(candidates, key=lambda t: t[0]) if end == "left"
              else max(candidates, key=lambda t: t[0]))

    # Construct center from r and h
    x_c = r if end == "left" else (L - r)
    y_c = h
    center = (x_c, y_c)

    tangent_side = (x_end, y_c)
    tangent_sine = (x_t, sine_y(x_t, p, a, o))
    vertical_segment = ((x_end, 0.0), (x_end, y_c))

    # Diagnostics
    dist = math.hypot(tangent_sine[0] - x_c, tangent_sine[1] - y_c)
    dist_err = abs(dist - r)

    m_t = sine_dy_dx(x_t, p, a)
    denom = (tangent_sine[1] - y_c)
    m_circle = float("inf") if abs(denom) < 1e-12 else -(tangent_sine[0] - x_c) / denom
    slope_err = abs(m_t - m_circle) if math.isfinite(m_circle) else abs(tangent_sine[0] - x_c)

    # Arc choice: pick the arc that lies inside shelf
    theta_side = angle_of_point(x_c, y_c, tangent_side[0], tangent_side[1])
    theta_sine = angle_of_point(x_c, y_c, tangent_sine[0], tangent_sine[1])

    d_short = wrap_angle_pi(theta_sine - theta_side)
    d_long = d_short - 2.0 * math.pi if d_short > 0 else d_short + 2.0 * math.pi

    short_ok = arc_is_inside_shelf(cx=x_c, cy=y_c, r=r, theta0=theta_side, sweep=d_short,
                                   L=L, p=p, a=a, o=o)
    long_ok = arc_is_inside_shelf(cx=x_c, cy=y_c, r=r, theta0=theta_side, sweep=d_long,
                                  L=L, p=p, a=a, o=o)

    if short_ok and not long_ok:
        arc_sweep = d_short
    elif long_ok and not short_ok:
        arc_sweep = d_long
    elif short_ok and long_ok:
        arc_sweep = d_short if abs(d_short) <= abs(d_long) else d_long
    else:
        raise RuntimeError(
            "Solved tangency, but neither arc lies fully inside the shelf. "
            "Try a smaller h (or adjust sine parameters)."
        )

    arc_direction = "CCW" if arc_sweep >= 0 else "CW"

    return CornerFilletHeight(
        end=end, L=L, p=p, a=a, o=o, h=h,
        r=r, x_end=x_end, center=center,
        tangent_side=tangent_side,
        tangent_sine=tangent_sine,
        vertical_segment=vertical_segment,
        theta_side=theta_side,
        theta_sine=theta_sine,
        arc_direction=arc_direction,
        arc_sweep=arc_sweep,
        dist_err=dist_err,
        slope_err=slope_err
    )


# ----------------------------
# Plotting
# ----------------------------

def sample_arc(res: CornerFilletHeight, n: int = 260) -> List[Tuple[float, float]]:
    cx, cy = res.center
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append(point_on_arc(cx, cy, res.r, res.theta_side, res.arc_sweep, t))
    return pts

def plot_solution(res: CornerFilletHeight, *, sine_samples: int = 2400) -> None:
    L = res.L
    p, a, o = res.p, res.a, res.o
    cx, cy = res.center
    r = res.r

    xs = [L * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    arc = sample_arc(res, n=280)
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

    ts = res.tangent_sine
    tv = res.tangent_side
    plt.scatter([cx], [cy], marker="x", s=90, label="center")
    plt.scatter([tv[0], ts[0]], [tv[1], ts[1]], s=70, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"end={res.end}, h={res.h:g}, r={res.r:g}  dir={res.arc_direction} sweep={res.arc_sweep:.4f}\n"
        f"dist_err={res.dist_err:.2e}, slope_err={res.slope_err:.2e}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--o", type=float, required=True)
    ap.add_argument("--h", type=float, required=True, help="target height above wall for the side tangency (y_c)")
    ap.add_argument("--end", choices=["left", "right"], required=True)
    ap.add_argument("--samples", type=int, default=25000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    res = solve_corner_fillet_with_height(
        L=args.L, p=args.p, a=args.a, o=args.o, h=args.h,
        end=args.end, samples=args.samples
    )

    print("=== Corner fillet with fixed height ===")
    print(f"end            : {res.end}")
    print(f"L              : {res.L}")
    print(f"sine (o,a,p)   : ({res.o}, {res.a}, {res.p})")
    print(f"target height h: {res.h}")
    print(f"solved radius r: {res.r}")
    print(f"center         : {res.center}")
    print(f"tangent_side   : {res.tangent_side}")
    print(f"tangent_sine   : {res.tangent_sine}")
    print(f"vertical_seg   : {res.vertical_segment[0]} -> {res.vertical_segment[1]}")
    print(f"arc_direction  : {res.arc_direction}")
    print(f"arc_sweep      : {res.arc_sweep:.12f} rad")
    print(f"dist_err       : {res.dist_err:.3e}")
    print(f"slope_err      : {res.slope_err:.3e}")

    if args.plot:
        plot_solution(res)

if __name__ == "__main__":
    main()
