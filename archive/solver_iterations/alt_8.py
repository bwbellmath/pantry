#!/usr/bin/env python3
"""
corner_sine_single_r.py

Single-parameter solve:
  - Fix circle center y = h_target
  - Fix circle center x = r (left) or x = L - r (right)
  - Find r such that the circle's upper semicircle is tangent to the sine curve.

We solve M(r)=0 where:
  phi(x;r) = y_sine(x) - y_circle_upper(x;r)
  M(r) = min_{x in domain(r)} phi(x;r)

Interpretation:
  - If M(r) > 0: sine is everywhere above the circle (no intersection).
  - If M(r) < 0: sine dips below circle somewhere (two intersections).
  - Tangency occurs at M(r) = 0.

Default period is 24 (aesthetic preference).

Example:
  python corner_sine_single_r.py --L 29 --a 1 --o 5 --h 3 --end left --plot
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List

import argparse
import matplotlib.pyplot as plt


# ---- sine ----

def sine_y(x: float, p: float, a: float, o: float) -> float:
    k = 2.0 * math.pi / p
    return o + a * math.sin(k * x)

# ---- circle upper semicircle y(x) for given r, end ----

def circle_upper_y(x: float, *, r: float, h: float, end: str, L: float) -> float:
    end = end.lower()
    cx = r if end == "left" else (L - r)
    dx = x - cx
    inside = r * r - dx * dx
    if inside < 0:
        return float("nan")
    return h + math.sqrt(inside)

def phi(x: float, *, r: float, h: float, end: str, L: float, p: float, a: float, o: float) -> float:
    yc = circle_upper_y(x, r=r, h=h, end=end, L=L)
    if not math.isfinite(yc):
        return float("nan")
    return sine_y(x, p, a, o) - yc


# ---- 1D minimization: golden-section search ----

def golden_section_min(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    iters: int = 70
) -> Tuple[float, float]:
    """
    Minimize f over [a,b] assuming unimodal-ish behavior.
    Returns (x_min, f(x_min)).
    """
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
    fx = f(x)
    return x, fx


# ---- bracketing + bisection in r ----

def bisect_root(
    g: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    max_iter: int = 70,
    tol: float = 1e-10
) -> float:
    glo = g(lo)
    ghi = g(hi)
    if not (math.isfinite(glo) and math.isfinite(ghi)):
        raise RuntimeError("Non-finite g(lo) or g(hi).")
    if glo == 0.0:
        return lo
    if ghi == 0.0:
        return hi
    if glo * ghi > 0.0:
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


# ---- Solve M(r)=0 ----

@dataclass(frozen=True)
class Solution:
    end: str
    L: float
    p: float
    a: float
    o: float
    h: float
    r: float
    x_touch: float
    y_touch: float


def solve_radius_single(
    *,
    L: float,
    p: float,
    a: float,
    o: float,
    h: float,
    end: str,
    r_lo: float,
    r_hi: float
) -> Solution:
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be left or right")

    # domain of x for the circle arc near the side:
    # left: x in [0, 2r]
    # right: x in [L-2r, L]
    def M(r: float) -> float:
        if r <= 0:
            return float("inf")
        x0 = 0.0 if end == "left" else (L - 2.0 * r)
        x1 = 2.0 * r if end == "left" else L
        # clamp to [0,L]
        x0 = max(0.0, x0)
        x1 = min(L, x1)
        if x1 <= x0:
            return float("inf")

        f = lambda x: phi(x, r=r, h=h, end=end, L=L, p=p, a=a, o=o)
        _, fmin = golden_section_min(f, x0, x1, iters=70)
        return fmin

    # We want M(r)=0
    r_star = bisect_root(M, r_lo, r_hi, max_iter=75, tol=1e-9)

    # Recover touch point x by minimizing phi at r_star
    x0 = 0.0 if end == "left" else max(0.0, L - 2.0 * r_star)
    x1 = min(L, 2.0 * r_star) if end == "left" else L
    f = lambda x: phi(x, r=r_star, h=h, end=end, L=L, p=p, a=a, o=o)
    x_touch, fmin = golden_section_min(f, x0, x1, iters=90)
    y_touch = sine_y(x_touch, p, a, o)

    # fmin should be ~0 at tangency
    if abs(fmin) > 1e-5:
        raise RuntimeError(f"Converged r but tangency residual seems large: min phi={fmin}")

    return Solution(end=end, L=L, p=p, a=a, o=o, h=h, r=r_star, x_touch=x_touch, y_touch=y_touch)


# ---- plotting ----

def plot_solution(sol: Solution) -> None:
    L, p, a, o, h, r, end = sol.L, sol.p, sol.a, sol.o, sol.h, sol.r, sol.end
    cx = r if end == "left" else (L - r)

    xs = [L * i / 2400 for i in range(2401)]
    ys = [sine_y(x, p, a, o) for x in xs]

    # circle
    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [h + r * math.sin(t) for t in ths]

    # relevant arc window
    arc_x = []
    arc_y = []
    if end == "left":
        for x in [0.0 + (2.0 * r) * i / 500 for i in range(501)]:
            y = circle_upper_y(x, r=r, h=h, end=end, L=L)
            if math.isfinite(y):
                arc_x.append(x); arc_y.append(y)
    else:
        for x in [max(0.0, L - 2.0 * r) + (min(L, L) - max(0.0, L - 2.0 * r)) * i / 500 for i in range(501)]:
            y = circle_upper_y(x, r=r, h=h, end=end, L=L)
            if math.isfinite(y):
                arc_x.append(x); arc_y.append(y)

    x_end = 0.0 if end == "left" else L

    plt.figure()
    plt.plot(xs, ys, label="sine edge")
    plt.axhline(0.0, linestyle="--", label="wall y=0")
    plt.axvline(x_end, linestyle="--", label=f"side x={x_end:g}")

    plt.plot(circ_x, circ_y, alpha=0.25, label="circle (full)")
    plt.plot(arc_x, arc_y, linewidth=3.0, label="circle upper arc near side")

    # vertical segment up to center height (side tangency point)
    plt.plot([x_end, x_end], [0.0, h], linewidth=2.5, label="vertical segment to circle side tangency")

    plt.scatter([cx], [h], marker="x", s=90, label="center")
    plt.scatter([sol.x_touch], [sol.y_touch], s=70, label="tangent point on sine")
    plt.scatter([x_end], [h], s=70, label="tangent point on side")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"end={end}, p={p}, r={r:.6g}, center=({cx:.6g},{h:.6g})")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--o", type=float, required=True)
    ap.add_argument("--h", type=float, required=True, help="target center height (side tangency height)")
    ap.add_argument("--end", choices=["left", "right"], required=True)
    ap.add_argument("--p", type=float, default=24.0, help="sine period (default 24)")
    ap.add_argument("--rlo", type=float, default=1e-4)
    ap.add_argument("--rhi", type=float, default=None, help="upper bound on r (default min(p/2, L/2, o+|a|-h))")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # Default r upper bound:
    # - p/2: avoid touching next period region (your request)
    # - L/2: geometry at one end can't exceed this anyway
    # - o+|a|-h: ensure circle top doesn't exceed global sine max (optional but helpful)
    if args.rhi is None:
        args.rhi = min(args.p / 2.0, args.L / 2.0, max(1e-4, (args.o + abs(args.a) - args.h)))

    sol = solve_radius_single(
        L=args.L, p=args.p, a=args.a, o=args.o, h=args.h,
        end=args.end, r_lo=args.rlo, r_hi=args.rhi
    )

    print("=== Single-parameter solve (bracket on r only) ===")
    print(f"end            : {sol.end}")
    print(f"period p       : {sol.p}")
    print(f"sine (o,a)     : ({sol.o}, {sol.a})")
    print(f"target height h: {sol.h}")
    print(f"solved r       : {sol.r}")
    cx = sol.r if sol.end == "left" else (sol.L - sol.r)
    print(f"center         : ({cx}, {sol.h})")
    print(f"side tangency  : ({0.0 if sol.end=='left' else sol.L}, {sol.h})")
    print(f"sine tangency  : ({sol.x_touch}, {sol.y_touch})")

    if args.plot:
        plot_solution(sol)

if __name__ == "__main__":
    main()
