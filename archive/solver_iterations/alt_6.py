#!/usr/bin/env python3
"""
corner_sine_bilevel.py

Bi-level solve for an inside-corner fillet between:
  - a vertical side line at x=0 (left) or x=L (right),
  - a sine "front" edge y(x) = o + a*sin(2*pi*x/p),
  - with wall y=0 below, and the fillet circle contained inside the shelf region.

Goal:
  Find radius r such that the resulting tangent circle has center height y_c == h_target
  (equivalently, the side tangency point is at height h_target).

Inner solve (given r):
  - Fix x_c = r (left) or x_c = L-r (right).
  - Solve for a tangency point x_t on sine satisfying tangency + circle membership.
  - Recover y_c from perpendicularity.
  - Reject solutions that don't fit inside shelf.
  - Choose the interior arc (not merely the short arc).

Outer solve:
  - Solve g(r) = y_c(r) - h_target = 0 by bracketing + bisection over r.

Example (your requested demo):
  shelf offset o=5", amplitude a=1", target height h=3"
  python corner_sine_bilevel.py --L 29 --p 12 --a 1 --o 5 --h 3 --end left --plot
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

def wrap_angle_pi(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return theta

def angle_of_point(cx: float, cy: float, px: float, py: float) -> float:
    return math.atan2(py - cy, px - cx)


# ----------------------------
# Root helpers (robust)
# ----------------------------

def bisect_root(f: Callable[[float], float], lo: float, hi: float, *, max_iter: int = 90, tol: float = 1e-13) -> Optional[float]:
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
    Returns brackets [lo,hi] for sign changes AND tiny brackets around local minima
    where |f| is very small (helps when you have a double-root / grazing tangency).
    """
    if x1 < x0:
        x0, x1 = x1, x0

    xs = [x0 + (x1 - x0) * i / samples for i in range(samples + 1)]
    fs = [f(x) for x in xs]

    brackets: List[Tuple[float, float]] = []

    # sign changes
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
# Arc containment test
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
        if y > sine_y(x, p, a, o) + eps:
            return False
    return True


# ----------------------------
# Results
# ----------------------------

@dataclass(frozen=True)
class InnerSolution:
    r: float
    end: str
    x_end: float

    center: Tuple[float, float]        # (x_c, y_c)
    tangent_side: Tuple[float, float]  # (x_end, y_c)
    tangent_sine: Tuple[float, float]  # (x_t, y(x_t))

    theta_side: float
    theta_sine: float
    arc_direction: str
    arc_sweep: float                   # chosen interior arc sweep (signed)

    dist_err: float
    slope_err: float


@dataclass(frozen=True)
class OuterSolution:
    end: str
    L: float
    p: float
    a: float
    o: float
    h_target: float

    r: float
    inner: InnerSolution


# ----------------------------
# Inner solve: given r -> compute y_c and arc
# ----------------------------

def solve_inner_given_r(
    *,
    L: float, p: float, a: float, o: float,
    r: float,
    end: str,
    samples_x: int = 20000
) -> Optional[InnerSolution]:
    """
    For a fixed radius r, find a tangent circle that:
      - is tangent to vertical side at x=0 (left) or x=L (right),
      - is tangent to sine y(x),
      - and is contained inside the shelf (above wall, below sine).
    Returns None if no feasible solution for this r.
    """
    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")
    if r <= 0:
        return None

    x_end = 0.0 if end == "left" else L
    x_c = r if end == "left" else (L - r)

    # tangency x must lie within circle horizontal span near the side
    x_min = 0.0 if end == "left" else max(0.0, L - 2.0 * r)
    x_max = min(L, 2.0 * r) if end == "left" else L
    if x_max <= x_min:
        return None

    def F(x: float) -> float:
        # Tangency equation eliminating y_c:
        # F(x) = (x-x_c)^2*(1+m^2) - r^2*m^2 = 0
        m = sine_dy_dx(x, p, a)
        dx = x - x_c
        return (dx * dx) * (1.0 + m * m) - (r * r) * (m * m)

    brackets = find_brackets_with_nearzeros(F, x_min, x_max, samples=samples_x, near_zero=1e-8)
    if not brackets:
        return None

    candidates: List[Tuple[float, float]] = []  # (x_t, y_c)

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
        if not math.isfinite(y_c):
            continue

        # Containment constraints:
        # - circle above wall: y_c - r >= 0
        # - circle top not above global max of sine: y_c + r <= o+|a|
        # - side tangency point must be within shelf height at end: y_c <= y(x_end)
        if y_c < r - 1e-9:
            continue
        if y_c + r > (o + abs(a)) + 1e-9:
            continue
        if y_c > sine_y(x_end, p, a, o) + 1e-9:
            continue

        # de-dup x
        if all(abs(x_t - xt0) > 1e-7 for xt0, _ in candidates):
            candidates.append((x_t, y_c))

    if not candidates:
        return None

    # pick tangency closest to the side (consistent selection for outer continuity)
    x_t, y_c = (min(candidates, key=lambda t: t[0]) if end == "left"
                else max(candidates, key=lambda t: t[0]))

    y_t = sine_y(x_t, p, a, o)
    center = (x_c, y_c)
    tangent_side = (x_end, y_c)
    tangent_sine = (x_t, y_t)

    # diagnostics
    dist = math.hypot(x_t - x_c, y_t - y_c)
    dist_err = abs(dist - r)
    m_t = sine_dy_dx(x_t, p, a)
    denom = (y_t - y_c)
    m_circle = float("inf") if abs(denom) < 1e-12 else -(x_t - x_c) / denom
    slope_err = abs(m_t - m_circle) if math.isfinite(m_circle) else abs(x_t - x_c)

    # choose interior arc (short vs long)
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


# ----------------------------
# Outer solve: find r so that y_c(r)=h_target
# ----------------------------

def solve_radius_for_target_height(
    *,
    L: float, p: float, a: float, o: float,
    h_target: float,
    end: str,
    r_samples: int = 200,
    samples_x: int = 20000
) -> OuterSolution:
    """
    Outer solve: find r in a bounded interval such that inner solution exists and
    y_c(r) == h_target.
    """

    if h_target < 0:
        raise ValueError("h_target must be >= 0")

    end = end.lower().strip()
    if end not in ("left", "right"):
        raise ValueError("end must be 'left' or 'right'")

    # Conservative feasibility bounds for r (discussed above)
    r_hi = min(h_target, o + abs(a) - h_target)
    if r_hi <= 0:
        raise RuntimeError(
            f"No feasible radius possible: r_hi=min(h, o+|a|-h) <= 0. "
            f"Given h={h_target}, o={o}, a={a}."
        )

    # Additional constraint: side height at end must be >= h_target
    x_end = 0.0 if end == "left" else L
    y_end = sine_y(x_end, p, a, o)
    if h_target > y_end + 1e-9:
        raise RuntimeError(
            f"Target height h={h_target} is above shelf front edge at the {end} end (y_end={y_end})."
        )

    r_lo = 1e-6  # allow approaching 0 but avoid degenerate radius

    # Build samples of r and look for a sign change in g(r)=y_c(r)-h_target
    rs = [r_lo + (r_hi - r_lo) * i / r_samples for i in range(r_samples + 1)]

    vals: List[Tuple[float, float, InnerSolution]] = []  # (r, g, inner)

    for r in rs:
        inner = solve_inner_given_r(L=L, p=p, a=a, o=o, r=r, end=end, samples_x=samples_x)
        if inner is None:
            continue
        g = inner.center[1] - h_target
        vals.append((r, g, inner))

    if len(vals) < 2:
        raise RuntimeError(
            "Could not find enough feasible radii in the search interval to bracket a solution. "
            "Try changing h_target or increasing r_samples."
        )

    # Find a bracketing pair with sign change or near-zero
    vals.sort(key=lambda t: t[0])

    # near-zero shortcut
    best = min(vals, key=lambda t: abs(t[1]))
    if abs(best[1]) < 1e-8:
        return OuterSolution(end=end, L=L, p=p, a=a, o=o, h_target=h_target, r=best[0], inner=best[2])

    bracket: Optional[Tuple[Tuple[float, float, InnerSolution], Tuple[float, float, InnerSolution]]] = None
    for i in range(len(vals) - 1):
        r0, g0, in0 = vals[i]
        r1, g1, in1 = vals[i + 1]
        if g0 == 0.0:
            return OuterSolution(end=end, L=L, p=p, a=a, o=o, h_target=h_target, r=r0, inner=in0)
        if g0 * g1 < 0.0:
            bracket = (vals[i], vals[i + 1])
            break

    if bracket is None:
        # No sign change; report best achievable and suggest what happened
        r_best, g_best, in_best = best
        raise RuntimeError(
            "No sign change found in g(r)=y_c(r)-h_target across feasible r. "
            f"Closest found: r={r_best:.6g} gives y_c={in_best.center[1]:.6g} (error {g_best:.6g}). "
            "This can happen if y_c(r) does not cross h_target for this geometry."
        )

    (ra, ga, ina), (rb, gb, inb) = bracket

    # Outer bisection
    def g_of_r(r: float) -> Tuple[float, Optional[InnerSolution]]:
        inner = solve_inner_given_r(L=L, p=p, a=a, o=o, r=r, end=end, samples_x=samples_x)
        if inner is None:
            return (float("nan"), None)
        return (inner.center[1] - h_target, inner)

    a_r, b_r = ra, rb
    a_g, a_in = ga, ina
    b_g, b_in = gb, inb

    for _ in range(70):
        m_r = 0.5 * (a_r + b_r)
        m_g, m_in = g_of_r(m_r)
        if not math.isfinite(m_g) or m_in is None:
            # If inner solve fails at midpoint, shrink toward the endpoint that worked.
            m_r = math.nextafter(m_r, a_r)
            m_g, m_in = g_of_r(m_r)
            if not math.isfinite(m_g) or m_in is None:
                break

        if abs(m_g) < 1e-10 or (b_r - a_r) < 1e-10:
            return OuterSolution(end=end, L=L, p=p, a=a, o=o, h_target=h_target, r=m_r, inner=m_in)

        if a_g * m_g <= 0:
            b_r, b_g, b_in = m_r, m_g, m_in
        else:
            a_r, a_g, a_in = m_r, m_g, m_in

    # fallback: return best of endpoints
    cand = []
    if a_in is not None and math.isfinite(a_g):
        cand.append((abs(a_g), a_r, a_in))
    if b_in is not None and math.isfinite(b_g):
        cand.append((abs(b_g), b_r, b_in))
    if cand:
        _, r_best, in_best = min(cand, key=lambda t: t[0])
        return OuterSolution(end=end, L=L, p=p, a=a, o=o, h_target=h_target, r=r_best, inner=in_best)

    raise RuntimeError("Outer bisection failed unexpectedly.")


# ----------------------------
# Plotting
# ----------------------------

def sample_arc(inner: InnerSolution, n: int = 260) -> List[Tuple[float, float]]:
    cx, cy = inner.center
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append(point_on_arc(cx, cy, inner.r, inner.theta_side, inner.arc_sweep, t))
    return pts

def plot_solution(outer: OuterSolution, *, sine_samples: int = 2400) -> None:
    inner = outer.inner
    L, p, a, o = outer.L, outer.p, outer.a, outer.o

    xs = [L * i / sine_samples for i in range(sine_samples + 1)]
    ys = [sine_y(x, p, a, o) for x in xs]

    cx, cy = inner.center
    r = inner.r

    ths = [2.0 * math.pi * i / 900 for i in range(901)]
    circ_x = [cx + r * math.cos(t) for t in ths]
    circ_y = [cy + r * math.sin(t) for t in ths]

    arc = sample_arc(inner, n=280)
    arc_x = [u for u, v in arc]
    arc_y = [v for u, v in arc]

    x_end = inner.x_end
    # vertical segment up to circle-side tangency point
    plt.figure()
    plt.plot(xs, ys, label="front edge: sine y(x)")
    plt.axhline(0.0, linestyle="--", label="wall: y=0")
    plt.axvline(x_end, linestyle="--", label=f"side: x={x_end:g}")

    plt.plot(circ_x, circ_y, alpha=0.35, label="circle (full)")
    plt.plot(arc_x, arc_y, linewidth=3.0, label="chosen interior arc")

    # vertical segment
    plt.plot([x_end, x_end], [0.0, cy], linewidth=2.5, label="vertical segment to circle")

    tv = inner.tangent_side
    ts = inner.tangent_sine
    plt.scatter([cx], [cy], marker="x", s=90, label="center")
    plt.scatter([tv[0], ts[0]], [tv[1], ts[1]], s=70, label="tangency points")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(
        f"end={outer.end}, target h={outer.h_target:g} => solved r={outer.r:.6g}\n"
        f"y_c={cy:.6g}, dist_err={inner.dist_err:.2e}, slope_err={inner.slope_err:.2e}, arc={inner.arc_direction}"
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
    ap.add_argument("--h", type=float, required=True, help="target side tangency height above wall")
    ap.add_argument("--end", choices=["left", "right"], required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--r_samples", type=int, default=220, help="outer bracketing samples in r")
    ap.add_argument("--x_samples", type=int, default=20000, help="inner bracketing samples in x")
    args = ap.parse_args()

    out = solve_radius_for_target_height(
        L=args.L, p=args.p, a=args.a, o=args.o,
        h_target=args.h,
        end=args.end,
        r_samples=args.r_samples,
        samples_x=args.x_samples
    )

    inner = out.inner
    print("=== Bi-level corner solve ===")
    print(f"end            : {out.end}")
    print(f"L              : {out.L}")
    print(f"sine (o,a,p)   : ({out.o}, {out.a}, {out.p})")
    print(f"target height h: {out.h_target}")
    print(f"solved radius r: {out.r}")
    print(f"center         : {inner.center}")
    print(f"tangent_side   : {inner.tangent_side}")
    print(f"tangent_sine   : {inner.tangent_sine}")
    print(f"vertical_seg   : ({inner.x_end},0) -> ({inner.x_end},{inner.center[1]})")
    print(f"arc_direction  : {inner.arc_direction}")
    print(f"arc_sweep      : {inner.arc_sweep:.12f} rad")
    print(f"dist_err       : {inner.dist_err:.3e}")
    print(f"slope_err      : {inner.slope_err:.3e}")

    if args.plot:
        plot_solution(out)

if __name__ == "__main__":
    main()
