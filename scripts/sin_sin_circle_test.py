import math
import os
from typing import Dict, Tuple, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _short_arc_span(theta_a: float, theta_b: float) -> float:
    return abs(_wrap_to_pi(theta_b - theta_a))


def _newton_2d(
    F, x0: Tuple[float, float],
    tol: float = 1e-12,
    max_iter: int = 60,
    fd_eps: float = 1e-7,
) -> Optional[Tuple[float, float]]:
    """2D Newton with finite-difference Jacobian + backtracking."""
    x, y = x0

    def norm2(v):
        return math.hypot(v[0], v[1])

    fx, fy = F(x, y)
    fn = norm2((fx, fy))

    for _ in range(max_iter):
        if fn < tol:
            return (x, y)

        # FD Jacobian J = [[dFx/dx, dFx/dy],[dFy/dx, dFy/dy]]
        hx = fd_eps * (1.0 + abs(x))
        hy = fd_eps * (1.0 + abs(y))

        fpx = F(x + hx, y)
        fmx = F(x - hx, y)
        dFdx = ((fpx[0] - fmx[0]) / (2 * hx), (fpx[1] - fmx[1]) / (2 * hx))

        fpy = F(x, y + hy)
        fmy = F(x, y - hy)
        dFdy = ((fpy[0] - fmy[0]) / (2 * hy), (fpy[1] - fmy[1]) / (2 * hy))

        a, c = dFdx  # dFx/dx, dFy/dx
        b, d = dFdy  # dFx/dy, dFy/dy

        det = a * d - b * c
        if abs(det) < 1e-18:
            return None

        # Solve J * [dx, dy]^T = -F via 2x2 inverse
        dx = (-fx * d + b * fy) / det
        dy = (c * fx - a * fy) / det

        # Backtracking line search
        alpha = 1.0
        accepted = False
        for _ls in range(20):
            xt = x + alpha * dx
            yt = y + alpha * dy
            g0, g1 = F(xt, yt)
            gn = norm2((g0, g1))
            if gn < fn:
                x, y = xt, yt
                fx, fy = g0, g1
                fn = gn
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return None

    return None



def tangent_circle_two_sinusoids_offset_intersection(
    Av: float, x_pos: float, y_off_v: float,
    Ah: float, y_pos: float, x_off_h: float,
    r: float,
    quadrant: str,
    *,
    tol: float = 1e-12,
) -> Dict[str, object]:
    """
    Horizontal sinusoid: y = y_pos + Ah * sin(x - x_off_h)
    Vertical   sinusoid: x = x_pos + Av * sin(y - y_off_v)

    Returns:
      - center: (cx, cy)
      - points: ordered tangency points per quadrant rule
      - short_arc_span: radians of the shorter arc between them
    """

    quadrant = quadrant.lower().strip()
    if quadrant not in {"br", "tr", "tl", "bl"}:
        raise ValueError("quadrant must be one of {'br','tr','tl','bl'}")
    if r <= 0:
        raise ValueError("r must be positive")

    # Side selection from quadrant
    # sh: +1 above horizontal curve, -1 below
    # sv: +1 right of vertical curve, -1 left
    sh = +1.0 if quadrant in {"tr", "tl"} else -1.0
    sv = +1.0 if quadrant in {"tr", "br"} else -1.0

    # Horizontal curve point + unit normal
    def horiz_point_and_unit_normal(x: float):
        y = y_pos + Ah * math.sin(x - x_off_h)
        yp = Ah * math.cos(x - x_off_h)  # dy/dx
        # normal direction (-yp, 1)
        inv = 1.0 / math.sqrt(1.0 + yp * yp)
        nx, ny = (-yp * inv, 1.0 * inv)
        return (x, y), (nx, ny), yp

    # Vertical curve point + unit normal
    def vert_point_and_unit_normal(y: float):
        x = x_pos + Av * math.sin(y - y_off_v)
        xp = Av * math.cos(y - y_off_v)  # dx/dy
        # normal direction (1, -xp)
        inv = 1.0 / math.sqrt(1.0 + xp * xp)
        mx, my = (1.0 * inv, -xp * inv)
        return (x, y), (mx, my), xp

    # Offset-center curves:
    def Ch(x: float):
        P, n, _ = horiz_point_and_unit_normal(x)
        return (P[0] + sh * r * n[0], P[1] + sh * r * n[1])

    def Cv(y: float):
        P, m, _ = vert_point_and_unit_normal(y)
        return (P[0] + sv * r * m[0], P[1] + sv * r * m[1])

    # Solve Ch(xh) - Cv(yv) = 0
    def F(xh: float, yv: float):
        cxh, cyh = Ch(xh)
        cxv, cyv = Cv(yv)
        return (cxh - cxv, cyh - cyv)

    # Initial guess: start at extrema in the selected quadrant.
    xh0 = x_off_h + (math.pi / 2.0 if sh > 0 else -math.pi / 2.0)
    yv0 = y_off_v + (math.pi / 2.0 if sv > 0 else -math.pi / 2.0)
    cx0 = x_pos + sv * (r + Av)
    cy0 = y_pos + sh * (r + Ah)

    # Try a few 2pi shifts because sinusoids repeat
    shifts = [0.0, 2.0 * math.pi, -2.0 * math.pi, 4.0 * math.pi, -4.0 * math.pi]
    best = None
    best_score = float("inf")

    for sx in shifts:
        for sy in shifts:
            guess = (xh0 + sx, yv0 + sy)
            sol = _newton_2d(F, guess, tol=tol)
            if sol is None:
                continue
            xh, yv = sol
            cxh, cyh = Ch(xh)

            # Quadrant sanity vs midlines
            if sv > 0 and not (cxh > x_pos):
                continue
            if sv < 0 and not (cxh < x_pos):
                continue
            if sh > 0 and not (cyh > y_pos):
                continue
            if sh < 0 and not (cyh < y_pos):
                continue

            score = (cxh - cx0) ** 2 + (cyh - cy0) ** 2
            if score < best_score:
                best_score = score
                best = (xh, yv)

    if best is None:
        raise RuntimeError("No solution found (try a different branch/initialization or check radius feasibility).")

    xh, yv = best
    (xh_p, yh_p), _, _ = horiz_point_and_unit_normal(xh)
    (xv_p, yv_p), _, _ = vert_point_and_unit_normal(yv)

    # Both computed centers should match; use horizontal one
    cx, cy = Ch(xh)

    p_horz = (xh_p, yh_p)  # tangency on horizontal sinusoid
    p_vert = (xv_p, yv_p)  # tangency on vertical sinusoid

    # Order rule you specified:
    # br: vert then horiz
    # tr: horiz then vert
    # bl: horiz then vert
    # tl: vert then horiz
    if quadrant in {"br", "tl"}:
        ordered = [p_vert, p_horz]
    else:
        ordered = [p_horz, p_vert]

    theta1 = math.atan2(ordered[0][1] - cy, ordered[0][0] - cx)
    theta2 = math.atan2(ordered[1][1] - cy, ordered[1][0] - cx)
    span = _short_arc_span(theta1, theta2)

    return {
        "center": (cx, cy),
        "points": ordered,
        "short_arc_span": span,
        "angles": (theta1, theta2),
        "raw_points": {"vertical": p_vert, "horizontal": p_horz},
        "quadrant": quadrant,
    }


def _horiz_curve(x: float, Ah: float, y_pos: float, x_off_h: float) -> float:
    return y_pos + Ah * math.sin(x - x_off_h)


def _vert_curve(y: float, Av: float, x_pos: float, y_off_v: float) -> float:
    return x_pos + Av * math.sin(y - y_off_v)


def _horiz_offset_center(x: float, Ah: float, y_pos: float, x_off_h: float, r: float, sh: float):
    y = _horiz_curve(x, Ah, y_pos, x_off_h)
    yp = Ah * math.cos(x - x_off_h)
    inv = 1.0 / math.sqrt(1.0 + yp * yp)
    nx, ny = (-yp * inv, 1.0 * inv)
    return (x + sh * r * nx, y + sh * r * ny)


def _vert_offset_center(y: float, Av: float, x_pos: float, y_off_v: float, r: float, sv: float):
    x = _vert_curve(y, Av, x_pos, y_off_v)
    xp = Av * math.cos(y - y_off_v)
    inv = 1.0 / math.sqrt(1.0 + xp * xp)
    mx, my = (1.0 * inv, -xp * inv)
    return (x + sv * r * mx, y + sv * r * my)


def plot_case(
    Av: float, x_pos: float, y_off_v: float,
    Ah: float, y_pos: float, x_off_h: float,
    r: float,
    quadrant: str,
    out_path: str,
    *,
    n: int = 800,
) -> None:
    period = 2.0 * math.pi
    span = 2.5 * period
    x_min = x_pos - span
    x_max = x_pos + span
    y_min = y_pos - span
    y_max = y_pos + span

    xs = [x_min + (x_max - x_min) * i / (n - 1) for i in range(n)]
    ys = [y_min + (y_max - y_min) * i / (n - 1) for i in range(n)]

    y_h = [_horiz_curve(x, Ah, y_pos, x_off_h) for x in xs]
    x_v = [_vert_curve(y, Av, x_pos, y_off_v) for y in ys]

    quadrant = quadrant.lower().strip()
    sh = +1.0 if quadrant in {"tr", "tl"} else -1.0
    sv = +1.0 if quadrant in {"tr", "br"} else -1.0

    ch_x, ch_y = zip(*[_horiz_offset_center(x, Ah, y_pos, x_off_h, r, sh) for x in xs])
    cv_x, cv_y = zip(*[_vert_offset_center(y, Av, x_pos, y_off_v, r, sv) for y in ys])

    sol = tangent_circle_two_sinusoids_offset_intersection(
        Av, x_pos, y_off_v,
        Ah, y_pos, x_off_h,
        r,
        quadrant,
    )
    cx, cy = sol["center"]
    p1, p2 = sol["points"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, y_h, label="horizontal sinusoid", color="tab:blue", lw=1.5)
    ax.plot(x_v, ys, label="vertical sinusoid", color="tab:orange", lw=1.5)

    ax.plot(ch_x, ch_y, label="offset curve (horizontal)", color="tab:blue", ls="--", lw=1.0)
    ax.plot(cv_x, cv_y, label="offset curve (vertical)", color="tab:orange", ls="--", lw=1.0)

    # Circle
    t = [2.0 * math.pi * i / 360 for i in range(361)]
    ax.plot([cx + r * math.cos(tt) for tt in t], [cy + r * math.sin(tt) for tt in t],
            color="tab:green", lw=1.5, label="solution circle")

    ax.scatter([cx], [cy], color="tab:green", s=30, zorder=5, label="circle center")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="tab:red", s=30, zorder=6, label="tangency points")

    ax.set_aspect("equal", "box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sinusoid tangency circle: solution and offset curves")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def test_flat_lines_solution() -> None:
    # With zero amplitudes, the sinusoids are lines: y=y_pos and x=x_pos.
    # The circle center should be exactly (x_pos +/- r, y_pos +/- r).
    r = 2.0
    Av = 0.0
    Ah = 0.0
    x_pos = 1.0
    y_pos = -3.0
    x_off_h = 0.0
    y_off_v = 0.0

    cases = {
        "tr": (x_pos + r, y_pos + r),
        "br": (x_pos + r, y_pos - r),
        "tl": (x_pos - r, y_pos + r),
        "bl": (x_pos - r, y_pos - r),
    }

    for quadrant, expected in cases.items():
        sol = tangent_circle_two_sinusoids_offset_intersection(
            Av, x_pos, y_off_v,
            Ah, y_pos, x_off_h,
            r,
            quadrant,
        )
        cx, cy = sol["center"]
        ex, ey = expected
        assert abs(cx - ex) < 1e-9, (quadrant, cx, ex)
        assert abs(cy - ey) < 1e-9, (quadrant, cy, ey)


def test_offset_phase_center() -> None:
    # User-specified setup: both midlines at 0, with phase offsets chosen so the
    # upper-right circle center should land at (r + amplitude, r + amplitude).
    A = 1.25
    r = 0.75
    period = 2.0 * math.pi

    x_pos = 0.0
    y_pos = 0.0
    x_off_h = -period / 4.0 + r + A
    # For x = A at y = r + A, need y_off_v = (r + A) - pi/2.
    y_off_v = -period / 4.0 + r + A

    sol = tangent_circle_two_sinusoids_offset_intersection(
        Av=A, x_pos=x_pos, y_off_v=y_off_v,
        Ah=A, y_pos=y_pos, x_off_h=x_off_h,
        r=r,
        quadrant="tr",
    )
    cx, cy = sol["center"]

    expected = r + A
    assert abs(cx - expected) < 1e-6, (cx, expected)
    assert abs(cy - expected) < 1e-6, (cy, expected)


def run_tests() -> None:
    test_flat_lines_solution()
    test_offset_phase_center()


if __name__ == "__main__":
    run_tests()
    plot_case(
        Av=1.25, x_pos=0.0, y_off_v=-math.pi / 2.0 + 1.25 + 0.75,
        Ah=1.25, y_pos=0.0, x_off_h=-math.pi / 2.0 + 1.25 + 0.75,
        r=0.75,
        quadrant="tr",
        out_path="sin_sin_circle_test.png",
    )
    print("Wrote plot to sin_sin_circle_test.png")
