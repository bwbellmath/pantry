#!/usr/bin/env python3
"""
corner_rotate_center.py

Given:
  - shelf sine edge y(x)=o + a*sin(2*pi*x/p)
  - shelf domain x in [0, L]
  - anchor point A on side line at height h: A=(0,h) for left, (L,h) for right
  - circle radius r (default 4)

We rotate the circle center C(theta) around A with |C-A|=r and search theta
so that the circle is tangent to the sine curve.

For a fixed theta, define:
  q(x;theta) = (x-Cx)^2 + (y_sine(x)-Cy)^2 - r^2
Intersections correspond to roots q=0 on x in [Cx-r, Cx+r] intersected with [0,L].

Bracketing signal:
  min_q(theta) = min_x q(x;theta)
  - min_q > 0 => 0 intersections (circle too "far" from curve)
  - min_q < 0 => 2 intersections (circle crosses the curve)
  - min_q == 0 => tangency (double root)

We find theta* where min_q(theta*)=0 by scanning for a sign change and then bisection.

Defaults:
  p=24 for aesthetics
  theta range is [-pi/2, +pi/2] which keeps center inside shelf in x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

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


# ----------------------------
# Center parameterization
# ----------------------------

def center_from_theta(*, theta: float, end: str, L: float, h: float, r: float) -> Tuple[float, float]:
    """
    end='left':  A=(0,h),  C=A + r*(cosθ, sinθ)
    end='right': A=(L,h),  C=A + r*(-cosθ, sinθ)  (θ=0 points inward)
    """
    end = end.lower()
    if end == "left":
        ax = 0.0
        return (ax + r * math.cos(theta), h + r * math.sin(theta))
    elif end == "right":
        ax = L
        return (ax - r * math.cos(theta), h + r * math.sin(theta))
    else:
        raise ValueError("end must be 'left' or 'right'")


# ----------------------------
# q(x;theta) and search interval
# ----------------------------

def q_of_x(
    x: float,
    *,
    cx: float, cy: float,
    r: float,
    p: float, a: float, o: float
) -> float:
    y = sine_y(x, p, a, o)
    dx = x - cx
    dy = y - cy
    return dx * dx + dy * dy - r * r

def circle_x_interval(cx: float, r: float, L: float) -> Tuple[float, float]:
    return (max(0.0, cx - r), min(L, cx + r))


# ----------------------------
# 1D minimization over x (sample + local refine)
# ----------------------------

def golden_section_min(f: Callable[[float], float], a: float, b: float, *, iters: int = 60) -> Tuple[float, float]:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if not math.isfinite(fc):
            fc = float("inf")
        if not math.isfinite(fd):
            fd = float("inf")
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)

def min_q_over_x(
    *,
    cx: float, cy: float,
    r: float,
    L: float,
    p: float, a: float, o: float,
    samples: int = 1200
) -> Tuple[float, float]:
    """
    Returns (x_min, min_q) by:
      - coarse sampling to find a good basin
      - local golden-section refine in a small neighborhood
    """
    x0, x1 = circle_x_interval(cx, r, L)
    if x1 <= x0:
        return (x0, float("inf"))

    f = lambda x: q_of_x(x, cx=cx, cy=cy, r=r, p=p, a=a, o=o)

    # coarse sample
    best_x = x0
    best_v = f(x0)
    for i in range(1, samples + 1):
        x = x0 + (x1 - x0) * i / samples
        v = f(x)
        if math.isfinite(v) and v < best_v:
            best_v = v
            best_x = x

    # refine locally around best_x
    width = (x1 - x0) / samples
    lo = max(x0, best_x - 5.0 * width)
    hi = min(x1, best_x + 5.0 * width)
    xr, vr = golden_section_min(f, lo, hi, iters=70)
    return (xr, vr)


# ----------------------------
# Solve for theta: min_q(theta)=0
# ----------------------------

def bisect_root(g: Callable[[float], float], lo: float, hi: float, *, max_iter: int = 70, tol: float = 1e-9) -> float:
    glo = g(lo)
    ghi = g(hi)
    if not (math.isfinite(glo) and math.isfinite(ghi)):
        raise RuntimeError("Non-finite g at bracket endpoints.")
    if glo == 0.0:
        return lo
    if ghi == 0.0:
        return hi
    if glo * ghi > 0:
        raise RuntimeError(f"Root not bracketed: g(lo)={glo}, g(hi)={ghi}")

    a, b = lo, hi
    fa, fb = glo, ghi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = g(m)
        if not math.isfinite(fm):
            # nudge
            m = math.nextafter(m, a)
            fm = g(m)
            if not math.isfinite(fm):
                break
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

def find_theta_bracket(
    *,
    end: str, L: float, h: float, r: float,
    p: float, a: float, o: float,
    theta_lo: float, theta_hi: float,
    theta_samples: int = 200
) -> Tuple[float, float]:
    """
    Scan theta interval to find a sign change in min_q(theta).
    """
    def g(theta: float) -> float:
        cx, cy = center_from_theta(theta=theta, end=end, L=L, h=h, r=r)
        _, mq = min_q_over_x(cx=cx, cy=cy, r=r, L=L, p=p, a=a, o=o)
        return mq

    thetas = [theta_lo + (theta_hi - theta_lo) * i / theta_samples for i in range(theta_samples + 1)]
    vals = [g(t) for t in thetas]

    # If we already hit ~0 somewhere, bracket around it
    best_i = min(range(len(vals)), key=lambda i: abs(vals[i]) if math.isfinite(vals[i]) else float("inf"))
    if abs(vals[best_i]) < 1e-7:
        i = best_i
        lo = thetas[max(0, i - 1)]
        hi = thetas[min(theta_samples, i + 1)]
        return (lo, hi)

    for i in range(theta_samples):
        v0, v1 = vals[i], vals[i + 1]
        if not (math.isfinite(v0) and math.isfinite(v1)):
            continue
        if v0 * v1 < 0:
            return (thetas[i], thetas[i + 1])

    raise RuntimeError(
        "No sign change found in min_q(theta) over the theta range. "
        "Try expanding theta range or adjusting r/h/(a,o,p)."
    )


@dataclass(frozen=True)
class Solution:
    end: str
    L: float
    p: float
    a: float
    o: float
    h: float
    r: float
    theta: float
    center: Tuple[float, float]
    x_touch: float
    y_touch: float
    slope_err: float


def solve_theta_for_tangency(
    *,
    end: str,
    L: float,
    h: float,
    r: float,
    p: float,
    a: float,
    o: float,
    theta_lo: float,
    theta_hi: float
) -> Solution:

    def g(theta: float) -> float:
        cx, cy = center_from_theta(theta=theta, end=end, L=L, h=h, r=r)
        _, mq = min_q_over_x(cx=cx, cy=cy, r=r, L=L, p=p, a=a, o=o)
        return mq

    lo, hi = find_theta_bracket(
        end=end, L=L, h=h, r=r, p=p, a=a, o=o,
        theta_lo=theta_lo, theta_hi=theta_hi,
        theta_samples=220
    )

    theta_star = bisect_root(g, lo, hi, max_iter=75, tol=1e-9)
    cx, cy = center_from_theta(theta=theta_star, end=end, L=L, h=h, r=r)
    x_touch, mq = min_q_over_x(cx=cx, cy=cy, r=r, L=L, p=p, a=a, o=o, samples=2000)
    y_touch = sine_y(x_touch, p, a, o)

    # slope check at touch point (should be ~0 residual)
    # circle implicit slope: y' = -(x-cx)/(y-cy)
    denom = (y_touch - cy)
    m_circle = float("inf") if abs(denom) < 1e-12 else -(x_touch - cx) / denom
    m_sine = sine_dy_dx(x_touch, p, a)
    slope_err = abs(m_sine - m_circle) if math.isfinite(m_circle) else abs(x_touch - cx)

    if abs(mq) > 1e-6:
        raise RuntimeError(f"Converged theta but min_q residual seems large: {mq}")

    return Solution(
        end=end, L=L, p=p, a=a, o=o, h=h, r=r,
        theta=theta_star,
        center=(cx, cy),
        x_touch=x_touch,
        y_touch=y_touch,
        slope_err=slope_err
    )


# ----------------------------
# Plotting
# ----------------------------

def plot_solution(sol: Solution) -> None:
    L, p, a, o = sol.L, sol.p, sol.a, sol.o
    cx, cy = sol.center
    r = sol.r
    end = sol.end.lower()
    ax = 0.0 if end == "left" else L
    ay = sol.h

    xs = [L * i / 2400 for i in range(2401)]
    ys = [sine_y(x, p, a, o) for x in xs]

    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    # Draw the portion of circle near the side
    x0, x1 = circle_x_interval(cx, r, L)
    arc_x = []
    arc_y = []
    for i in range(600):
        x = x0 + (x1 - x0) * i / 599
        # pick the point on the circle closest to the sine (just for visualization)
        inside = r*r - (x - cx)**2
        if inside < 0:
            continue
        y_up = cy + math.sqrt(inside)
        y_dn = cy - math.sqrt(inside)
        y_s = sine_y(x, p, a, o)
        # choose branch closer to sine
        y = y_up if abs(y_s - y_up) <= abs(y_s - y_dn) else y_dn
        arc_x.append(x); arc_y.append(y)

    plt.figure()
    plt.plot(xs, ys, label="sine edge")
    plt.axhline(0.0, linestyle="--", label="wall y=0")
    plt.axvline(ax, linestyle="--", label=f"side x={ax:g}")

    plt.plot(circ_x, circ_y, alpha=0.25, label="circle (full)")
    plt.plot(arc_x, arc_y, linewidth=3.0, label="circle near side")

    # anchor and center
    plt.scatter([ax], [ay], s=70, label="anchor (on side)")
    plt.scatter([cx], [cy], marker="x", s=90, label="center")
    plt.scatter([sol.x_touch], [sol.y_touch], s=70, label="tangent point on sine")

    # line from anchor to center (rotation radius)
    plt.plot([ax, cx], [ay, cy], linewidth=2.0, label="center rotation radius")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"end={sol.end}, r={r:g}, theta={sol.theta:.4f} rad, slope_err={sol.slope_err:.2e}")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--o", type=float, required=True)
    ap.add_argument("--h", type=float, required=True, help="anchor height on side line (y)")
    ap.add_argument("--r", type=float, default=4.0, help="circle radius (default 4)")
    ap.add_argument("--end", choices=["left", "right"], required=True)
    ap.add_argument("--p", type=float, default=24.0, help="sine period (default 24)")
    ap.add_argument("--theta_lo", type=float, default=-math.pi/2, help="theta lower bound (default -pi/2)")
    ap.add_argument("--theta_hi", type=float, default= math.pi/2, help="theta upper bound (default +pi/2)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    sol = solve_theta_for_tangency(
        end=args.end, L=args.L, h=args.h, r=args.r,
        p=args.p, a=args.a, o=args.o,
        theta_lo=args.theta_lo, theta_hi=args.theta_hi
    )

    print("=== Rotate-center solve (bracket on theta) ===")
    print(f"end        : {sol.end}")
    print(f"p          : {sol.p}")
    print(f"(o,a)      : ({sol.o}, {sol.a})")
    print(f"anchor     : ({0.0 if sol.end=='left' else sol.L}, {sol.h})")
    print(f"r          : {sol.r}")
    print(f"theta      : {sol.theta}")
    print(f"center     : {sol.center}")
    print(f"touch      : ({sol.x_touch}, {sol.y_touch})")
    print(f"slope_err  : {sol.slope_err:.3e}")

    if args.plot:
        plot_solution(sol)


if __name__ == "__main__":
    main()
