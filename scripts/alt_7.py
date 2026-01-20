#!/usr/bin/env python3
"""
corner_sine_bilevel2.py

Bi-level solve with:
  - inner solve: bracket in x only, NO y-feasibility pruning
  - outer solve: bracket in r to hit y_c(r) == h_target
  - after convergence: validate "circle contained inside shelf" and choose the interior arc.

Shelf:
  wall y=0
  front edge y(x)=o + a*sin(2*pi*x/p)
  x in [0,L]
  left corner uses x_c=r, right corner uses x_c=L-r

Run (your example):
  python corner_sine_bilevel2.py --L 29 --p 12 --a 1 --o 5 --h 3 --end left --plot
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import argparse
import matplotlib.pyplot as plt


# ---- sine ----

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


# ---- bracketing helpers ----

def bisect_root(f: Callable[[float], float], lo: float, hi: float, *,
                max_iter: int = 90, tol: float = 1e-13) -> Optional[float]:
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

def find_brackets_with_nearzeros(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    samples: int = 20000,
    near_zero: float = 1e-8
) -> List[Tuple[float, float]]:
    """
    Sign-change brackets + tiny brackets around local minima where |f| is small.
    Helps with grazing/double roots.
    """
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

    # near-zero local minima in |f|
    for i in range(1, samples):
        fprev, fcur, fnext = fs[i - 1], fs[i], fs[i + 1]
        if not (math.isfinite(fprev) and math.isfinite(fcur) and math.isfinite(fnext)):
            continue
        if abs(fcur) < near_zero and abs(fcur) <= abs(fprev) and abs(fcur) <= abs(fnext):
            eps = (x1 - x0) / samples
            brackets.append((max(x0, xs[i] - eps), min(x1, xs[i] + eps)))

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


# ---- arc containment ----

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
        if y > sine_y(x, p, a, o) + eps:
            return False
    return True


# ---- results ----

@dataclass(frozen=True)
class Candidate:
    x_t: float
    y_c: float

@dataclass(frozen=True)
class InnerSolution:
    r: float
    end: str
    x_end: float
    center: Tuple[float, float]
    tangent_side: Tuple[float, float]
    tangent_sine: Tuple[float, float]
    theta_side: float
    theta_sine: float
    arc_direction: str
    arc_sweep: float
    dist_err: float
    slope_err: float

@dataclass(frozen=True)
class OuterSolution:
    r: float
    inner: InnerSolution


# ---- inner solve: for given r, return all (x_t,y_c) candidates ----

def inner_candidates_given_r(
    *,
    L: float, p: float, a: float, o: float,
    r: float,
    end: str,
    x_samples: int
) -> List[Candidate]:
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")
    if r <= 0:
        return []

    x_end = 0.0 if end == "left" else L
    x_c = r if end == "left" else (L - r)

    # Tangency x must lie within circle x-span near the side.
    # (still purely x-geometry)
    x_min = 0.0 if end == "left" else max(0.0, L - 2.0 * r)
    x_max = min(L, 2.0 * r) if end == "left" else L
    if x_max <= x_min:
        return []

    def F(x: float) -> float:
        m = sine_dy_dx(x, p, a)
        dx = x - x_c
        return (dx * dx) * (1.0 + m * m) - (r * r) * (m * m)

    brackets = find_brackets_with_nearzeros(F, x_min, x_max, samples=x_samples, near_zero=1e-8)
    cands: List[Candidate] = []

    for lo, hi in brackets:
        x_t = bisect_root(F, lo, hi)
        if x_t is None:
            continue
        y_t = sine_y(x_t, p, a, o)
        m_t = sine_dy_dx(x_t, p, a)
        dx = x_t - x_c
        if abs(m_t) < 1e-12:
            continue
        y_c = y_t + dx / m_t
        if math.isfinite(y_c):
            # de-dup x roots
            if all(abs(x_t - c.x_t) > 1e-7 for c in cands):
                cands.append(Candidate(x_t=x_t, y_c=y_c))

    return cands


def build_inner_solution_from_candidate(
    *,
    L: float, p: float, a: float, o: float,
    r: float,
    end: str,
    cand: Candidate,
    require_contained: bool = True
) -> Optional[InnerSolution]:
    end = end.lower().strip()
    x_end = 0.0 if end == "left" else L
    x_c = r if end == "left" else (L - r)

    y_c = cand.y_c
    x_t = cand.x_t
    y_t = sine_y(x_t, p, a, o)
    m_t = sine_dy_dx(x_t, p, a)

    # Optional containment checks ONLY HERE (not during bracketing):
    if require_contained:
        # above wall
        if y_c - r < -1e-9:
            return None
        # below sine everywhere is hard; we enforce via interior-arc test later.
        # Ensure side tangency is within shelf height at end:
        if y_c > sine_y(x_end, p, a, o) + 1e-9:
            return None

    center = (x_c, y_c)
    tangent_side = (x_end, y_c)
    tangent_sine = (x_t, y_t)

    # Diagnostics
    dist = math.hypot(x_t - x_c, y_t - y_c)
    dist_err = abs(dist - r)

    denom = (y_t - y_c)
    m_circle = float("inf") if abs(denom) < 1e-12 else -(x_t - x_c) / denom
    slope_err = abs(m_t - m_circle) if math.isfinite(m_circle) else abs(x_t - x_c)

    theta_side = angle_of_point(x_c, y_c, tangent_side[0], tangent_side[1])
    theta_sine = angle_of_point(x_c, y_c, tangent_sine[0], tangent_sine[1])

    # pick interior arc
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
        return None

    arc_direction = "CCW" if arc_sweep >= 0 else "CW"

    return InnerSolution(
        r=r, end=end, x_end=x_end,
        center=center,
        tangent_side=tangent_side,
        tangent_sine=tangent_sine,
        theta_side=theta_side,
        theta_sine=theta_sine,
        arc_direction=arc_direction,
        arc_sweep=arc_sweep,
        dist_err=dist_err,
        slope_err=slope_err
    )


# ---- outer solve: choose r so that y_c(r)=h_target ----

def solve_bilevel(
    *,
    L: float, p: float, a: float, o: float,
    h_target: float,
    end: str,
    rmax: Optional[float],
    r_samples: int,
    x_samples: int
) -> OuterSolution:

    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    x_end = 0.0 if end == "left" else L
    y_end = sine_y(x_end, p, a, o)
    if h_target > y_end + 1e-9:
        raise RuntimeError(f"h_target={h_target} is above shelf height at the {end} end (y_end={y_end}).")

    # Default rmax per your suggestion; allow override.
    if rmax is None:
        rmax = min(L / 2.0, max(1e-6, (o + abs(a) - h_target)))
    if rmax <= 0:
        raise RuntimeError("rmax <= 0; choose different parameters or pass --rmax.")

    r_lo = 1e-6
    r_hi = rmax

    # For each r, compute candidates; pick candidate with y_c closest to h_target
    def best_g_at_r(r: float) -> Tuple[float, Optional[Candidate]]:
        cands = inner_candidates_given_r(L=L, p=p, a=a, o=o, r=r, end=end, x_samples=x_samples)
        if not cands:
            return (float("nan"), None)
        best = min(cands, key=lambda c: abs(c.y_c - h_target))
        return (best.y_c - h_target, best)

    # Sample r to find a sign change in g(r)
    samples = r_samples
    rs = [r_lo + (r_hi - r_lo) * i / samples for i in range(samples + 1)]
    kept: List[Tuple[float, float, Candidate]] = []
    for r in rs:
        g, cand = best_g_at_r(r)
        if cand is None or not math.isfinite(g):
            continue
        kept.append((r, g, cand))

    if len(kept) < 2:
        raise RuntimeError(
            "Could not evaluate g(r) for enough radii. Increase --r_samples or --x_samples, "
            "or pass a larger --rmax."
        )

    kept.sort(key=lambda t: t[0])

    # Find bracket
    bracket = None
    best = min(kept, key=lambda t: abs(t[1]))
    if abs(best[1]) < 1e-8:
        r_star, _, cand_star = best
        inner = build_inner_solution_from_candidate(
            L=L, p=p, a=a, o=o, r=r_star, end=end, cand=cand_star, require_contained=True
        )
        if inner is None:
            raise RuntimeError("Hit y-target but final containment/arc test failed; try bigger --rmax.")
        return OuterSolution(r=r_star, inner=inner)

    for i in range(len(kept) - 1):
        r0, g0, c0 = kept[i]
        r1, g1, c1 = kept[i + 1]
        if g0 * g1 < 0.0:
            bracket = (r0, g0, c0, r1, g1, c1)
            break

    if bracket is None:
        r_best, g_best, _ = best
        raise RuntimeError(
            "No sign change found in g(r)=y_c(r)-h_target on sampled radii. "
            f"Closest: r={r_best:.6g} gives error {g_best:.6g}. "
            "Try larger --rmax or increase --r_samples."
        )

    rA, gA, cA, rB, gB, cB = bracket

    # Outer bisection: always recompute best candidate at midpoint
    a_r, a_g, a_c = rA, gA, cA
    b_r, b_g, b_c = rB, gB, cB

    for _ in range(70):
        m_r = 0.5 * (a_r + b_r)
        m_g, m_c = best_g_at_r(m_r)
        if m_c is None or not math.isfinite(m_g):
            # shrink toward a_r
            m_r = math.nextafter(m_r, a_r)
            m_g, m_c = best_g_at_r(m_r)
            if m_c is None or not math.isfinite(m_g):
                break

        if abs(m_g) < 1e-10 or (b_r - a_r) < 1e-10:
            inner = build_inner_solution_from_candidate(
                L=L, p=p, a=a, o=o, r=m_r, end=end, cand=m_c, require_contained=True
            )
            if inner is None:
                raise RuntimeError("Converged in r, but final containment/arc test failed. Try larger --rmax.")
            return OuterSolution(r=m_r, inner=inner)

        if a_g * m_g <= 0:
            b_r, b_g, b_c = m_r, m_g, m_c
        else:
            a_r, a_g, a_c = m_r, m_g, m_c

    # fallback: pick the closer endpoint and validate
    candidates = [(abs(a_g), a_r, a_c), (abs(b_g), b_r, b_c)]
    _, r_star, cand_star = min(candidates, key=lambda t: t[0])
    inner = build_inner_solution_from_candidate(
        L=L, p=p, a=a, o=o, r=r_star, end=end, cand=cand_star, require_contained=True
    )
    if inner is None:
        raise RuntimeError("Outer solve fallback failed containment/arc test; try larger --rmax.")
    return OuterSolution(r=r_star, inner=inner)


# ---- plotting ----

def plot_solution(L: float, p: float, a: float, o: float, out: OuterSolution) -> None:
    inner = out.inner
    cx, cy = inner.center
    r = inner.r

    xs = [L * i / 2400 for i in range(2401)]
    ys = [sine_y(x, p, a, o) for x in xs]

    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    arc_pts = [point_on_arc(cx, cy, r, inner.theta_side, inner.arc_sweep, i / 280) for i in range(281)]
    ax = [u for u, v in arc_pts]
    ay = [v for u, v in arc_pts]

    x_end = inner.x_end

    plt.figure()
    plt.plot(xs, ys, label="sine edge")
    plt.axhline(0.0, linestyle="--", label="wall y=0")
    plt.axvline(x_end, linestyle="--", label=f"side x={x_end:g}")

    plt.plot(circ_x, circ_y, alpha=0.35, label="circle (full)")
    plt.plot(ax, ay, linewidth=3.0, label="chosen interior arc")
    plt.plot([x_end, x_end], [0.0, cy], linewidth=2.5, label="vertical segment to circle")

    tv = inner.tangent_side
    ts = inner.tangent_sine
    plt.scatter([cx], [cy], marker="x", s=90, label="center")
    plt.scatter([tv[0], ts[0]], [tv[1], ts[1]], s=70, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"r={out.r:.6g}, y_c={cy:.6g}, arc={inner.arc_direction}, dist_err={inner.dist_err:.2e}")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--o", type=float, required=True)
    ap.add_argument("--h", type=float, required=True, help="target center height y_c (side tangency height)")
    ap.add_argument("--end", choices=["left", "right"], required=True)
    ap.add_argument("--rmax", type=float, default=None, help="override upper bound on r")
    ap.add_argument("--r_samples", type=int, default=260)
    ap.add_argument("--x_samples", type=int, default=22000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out = solve_bilevel(
        L=args.L, p=args.p, a=args.a, o=args.o,
        h_target=args.h,
        end=args.end,
        rmax=args.rmax,
        r_samples=args.r_samples,
        x_samples=args.x_samples
    )

    inner = out.inner
    print("=== Bi-level solution (x-only inner bracketing) ===")
    print(f"end            : {args.end}")
    print(f"target y_c=h   : {args.h}")
    print(f"solved r       : {out.r}")
    print(f"center         : {inner.center}")
    print(f"tangent_side   : {inner.tangent_side}")
    print(f"tangent_sine   : {inner.tangent_sine}")
    print(f"vertical_seg   : ({inner.x_end},0) -> ({inner.x_end},{inner.center[1]})")
    print(f"arc_direction  : {inner.arc_direction}")
    print(f"arc_sweep      : {inner.arc_sweep:.12f} rad")
    print(f"dist_err       : {inner.dist_err:.3e}")
    print(f"slope_err      : {inner.slope_err:.3e}")

    if args.plot:
        plot_solution(args.L, args.p, args.a, args.o, out)

if __name__ == "__main__":
    main()
