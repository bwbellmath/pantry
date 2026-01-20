#!/usr/bin/env python3
"""
Extract exact geometry from shelf computations and export to SVG/PDF.
Uses the EXACT same points that are drawn in visualizations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from geometry import (
    solve_tangent_circle_two_sinusoids,
    generate_circle_arc,
    generate_sinusoid_points,
    wall_to_pantry_coords,
    generate_interior_mask
)
from config import ShelfConfig


def find_brackets(f, x0, x1, samples=12000):
    """Find all bracketing intervals where f changes sign."""
    if x1 < x0:
        x0, x1 = x1, x0
    xs = [x0 + (x1 - x0) * i / samples for i in range(samples + 1)]
    fs = [f(x) for x in xs]

    brackets = []
    for i in range(samples):
        f0, f1 = fs[i], fs[i + 1]
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if f0 == 0.0:
            eps = (x1 - x0) / samples
            brackets.append((max(x0, xs[i] - eps), min(x1, xs[i] + eps)))
        elif f0 * f1 < 0.0:
            brackets.append((xs[i], xs[i + 1]))

    # Merge overlapping brackets
    brackets.sort()
    merged = []
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


def bisect_root(f, lo, hi, max_iter=90, tol=1e-13, debug=False):
    """Find root of f in [lo, hi] using bisection."""
    flo = f(lo)
    fhi = f(hi)
    if debug:
        print(f"    BISECT: lo={lo:.6f}, flo={flo:.6f}, hi={hi:.6f}, fhi={fhi:.6f}")
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        if debug:
            print(f"    BISECT: Non-finite endpoint values")
        return None
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        if debug:
            print(f"    BISECT: Same sign at endpoints (flo*fhi={flo*fhi:.6f})")
        return None

    a, b = lo, hi
    fa, fb = flo, fhi
    for i in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if not np.isfinite(fm):
            if debug:
                print(f"    BISECT: Non-finite at iter {i}, m={m:.6f}")
            return None
        if abs(fm) < tol or (b - a) < tol:
            if debug:
                print(f"    BISECT: Converged at iter {i}, m={m:.6f}, fm={fm:.6e}")
            return m
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    if debug:
        print(f"    BISECT: Max iterations reached")
    return 0.5 * (a + b)


def arc_is_inside_shelf(center_x, center_y, radius, theta_start, theta_sweep,
                        sinusoid_func, y_min, y_max, samples=40, eps=1e-9):
    """Check if arc stays inside shelf bounds."""
    for i in range(1, samples):  # Skip endpoints
        t = i / samples
        theta = theta_start + theta_sweep * t
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)

        # Check y bounds
        if y < y_min - eps or y > y_max + eps:
            return False

        # Check x bounds (must be between wall and sinusoid)
        x_boundary = sinusoid_func(y)
        if not np.isfinite(x_boundary):
            return False

        # For left shelf (wall at x=0): must have 0 <= x <= sinusoid
        # For right shelf (wall at x=48): must have sinusoid <= x <= 48
        if center_x < 24:  # left shelf
            if x < -eps or x > x_boundary + eps:
                return False
        else:  # right shelf
            if x < x_boundary - eps or x > 48 + eps:
                return False

    return True


def solve_tangent_circle_horizontal_sinusoid(horizontal_y, depth, amplitude, period, offset, radius, side='E'):
    """
    Solve for circle tangent to horizontal line and sinusoid using bracketing method.

    Adapted from alt_4.py. Uses the constraint equation:
        F(y) = (x(y) - x_c)² (1 + (dx/dy)²) - r² (dx/dy)²

    where x_c is fixed based on the horizontal tangency requirement.

    Args:
        horizontal_y: Y-coordinate of horizontal line (back edge at y=29")
        depth: Base depth of sinusoid
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Sinusoid phase offset
        radius: Desired circle radius
        side: 'E' for left shelf, 'W' for right shelf

    Returns:
        (center_x, center_y, tangent_y): Circle center and tangency point y-coordinate
    """

    # Center y-coordinate is fixed (tangent to horizontal line)
    center_y = horizontal_y - radius

    def sinusoid_x(y):
        """X-coordinate on sinusoid at given y."""
        if side == 'E':
            return depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        else:  # 'W'
            return 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

    def sinusoid_dx_dy(y):
        """Derivative dx/dy of sinusoid at given y."""
        if side == 'E':
            return amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)
        else:  # 'W'
            return -amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)

    # Use the constraint from alt_4.py adapted to our coordinate system:
    # For tangency: (y_t - y_c)² (1 + (dx/dy)²) = r² (dx/dy)²
    def F(y):
        m = sinusoid_dx_dy(y)

        # (y - center_y)² (1 + m²) - r² m² = 0
        dy = y - center_y
        return (dy * dy) * (1.0 + m * m) - (radius * radius) * (m * m)

    # Find all bracketing intervals
    y_min = max(0, center_y - 2 * radius)
    y_max = horizontal_y
    brackets = find_brackets(F, y_min, y_max, samples=15000)

    if not brackets:
        raise RuntimeError(
            f"No tangency solution found for {side} shelf. "
            f"Try different radius or parameters."
        )

    # Collect all valid candidates
    candidates = []
    eps_slope = 1e-12

    for lo, hi in brackets:
        y_t = bisect_root(F, lo, hi)
        if y_t is None:
            continue

        x_t = sinusoid_x(y_t)
        m_t = sinusoid_dx_dy(y_t)

        if abs(m_t) < eps_slope:
            continue

        # Calculate center_x from tangency condition
        # Circle derivative at (x_t, y_t): dx/dy = -(y_t - y_c)/(x_t - x_c)
        # Must equal sinusoid derivative: m_t
        # So: -(y_t - center_y)/(x_t - x_c) = m_t
        # Rearranging: x_t - x_c = -(y_t - center_y)/m_t
        # Therefore: x_c = x_t + (y_t - center_y)/m_t
        x_c = x_t + (y_t - center_y) / m_t

        if not np.isfinite(x_c):
            continue

        # Validate: distance should equal radius
        dist = np.hypot(x_t - x_c, y_t - center_y)
        if abs(dist - radius) > 1e-6:
            continue

        # Validate: center must be inside shelf bounds
        # For left shelf (side='E'): center should be between wall (x=0) and sinusoid
        # For right shelf (side='W'): center should be between sinusoid and wall (x=48)
        x_boundary_at_center = sinusoid_x(center_y)
        if side == 'E':
            # Left shelf: 0 < center_x < sinusoid
            if x_c <= 0 or x_c >= x_boundary_at_center:
                continue
        else:
            # Right shelf: sinusoid < center_x < 48
            if x_c <= x_boundary_at_center or x_c >= 48:
                continue

        # De-duplicate
        if all(abs(y_t - yt0) > 1e-7 for yt0, _ in candidates):
            candidates.append((y_t, x_c))

    if not candidates:
        raise RuntimeError(
            f"Found tangency roots but none are valid for {side} shelf. "
            f"Try smaller radius."
        )

    # Choose candidate closest to back edge (typical corner rounding)
    y_t, center_x = max(candidates, key=lambda t: t[0])

    return center_x, center_y, y_t


def solve_door_smoothing_radius(door_y, tangent_x, depth, amplitude, period, offset, side='E'):
    """
    Solve for door smoothing radius where horizontal tangent x-coordinate is fixed.

    The circle must be tangent to:
    1. Horizontal line at y = door_y, with tangent point at x = tangent_x
    2. The sinusoid edge

    We solve for the radius that achieves both tangencies.

    Args:
        door_y: Y-coordinate of extended door line (typically -0.75")
        tangent_x: X-coordinate where circle touches door line
        depth: Base depth of sinusoid
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Sinusoid phase offset
        side: 'E' for left shelf, 'W' for right shelf

    Returns:
        (center_x, center_y, radius, tangent_y): Circle center, radius, and sinusoid tangency y
    """

    def sinusoid_x(y):
        """X-coordinate on sinusoid at given y."""
        if side == 'E':
            return depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        else:  # 'W'
            return 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

    def sinusoid_dx_dy(y):
        """Derivative dx/dy of sinusoid at given y."""
        if side == 'E':
            return amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)
        else:  # 'W'
            return -amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)

    # For a given y_t on the sinusoid, we can calculate the required radius
    # From the constraint: (h - r)^2 (1 + m^2) = r^2 m^2
    # where h = y_t - door_y
    # Solving for r: r = h[(1 + m^2) ± |m|sqrt(1 + m^2)]
    #
    # We also need: x_t = tangent_x - (h - r)/m
    # And: x_t = sinusoid_x(y_t)
    #
    # So we find y_t where these constraints are satisfied

    def F(y_t):
        """
        Function that equals zero when all constraints are satisfied.
        """
        try:
            if y_t <= door_y:
                return float('inf')

            h = y_t - door_y  # height above door line
            m_t = sinusoid_dx_dy(y_t)

            if abs(m_t) < 1e-12:
                return float('inf')

            # Calculate required radius from tangency constraint
            # (h - r)^2(1 + m^2) = r^2 m^2
            # Expanding: r^2 - 2hr(1 + m^2) + h^2(1 + m^2) = 0
            # Quadratic formula: r = h*sqrt(1+m^2) * [sqrt(1+m^2) ± |m|]
            term = 1 + m_t * m_t
            sqrt_term = np.sqrt(term)

            r1 = h * sqrt_term * (sqrt_term + abs(m_t))
            r2 = h * sqrt_term * (sqrt_term - abs(m_t))

            # Choose smaller positive radius for smoother transition
            if r1 > 0 and r2 > 0:
                radius = min(r1, r2)
            elif r1 > 0:
                radius = r1
            elif r2 > 0:
                radius = r2
            else:
                return float('inf')

            # Sanity check: radius should be reasonable (not too large)
            if radius > 50:  # 50 inches is unreasonably large for a shelf corner
                return float('inf')

            # Now check if this radius gives correct x_t
            # From tangency: x_t = tangent_x - (h - radius)/m_t
            delta = h - radius
            if abs(delta) > 100:  # Sanity check
                return float('inf')

            x_t_from_geometry = tangent_x - delta / m_t

            # x_t should also be on the sinusoid
            x_t_on_sinusoid = sinusoid_x(y_t)

            # Sanity check
            if not np.isfinite(x_t_from_geometry) or not np.isfinite(x_t_on_sinusoid):
                return float('inf')

            # Return the error
            error = x_t_from_geometry - x_t_on_sinusoid
            if not np.isfinite(error):
                return float('inf')

            return error
        except:
            return float('inf')

    # Search for y_t in the range [door_y, some reasonable upper bound]
    y_min = door_y + 0.01  # Just above door line
    y_max = min(29.0, door_y + 10.0)  # Don't go too far into the shelf

    brackets = find_brackets(F, y_min, y_max, samples=5000)

    if not brackets:
        print(f"  DEBUG: No brackets found for door smoothing. Tangent_x={tangent_x}, door_y={door_y}, side={side}")
        print(f"  DEBUG: Falling back to no door smoothing for this shelf")
        # Return None to indicate no smoothing solution
        return None, None, None, None

    # Find the solution closest to the door (smallest y_t)
    best_y_t = None
    best_radius = None

    print(f"  DEBUG: Found {len(brackets)} brackets for door smoothing")
    for i, (lo, hi) in enumerate(brackets):
        # Filter out brackets with large endpoint values (likely discontinuities)
        flo = F(lo)
        fhi = F(hi)
        if abs(flo) > 10 or abs(fhi) > 10:
            print(f"  DEBUG: Bracket {i}: skipping large endpoints flo={flo:.2f}, fhi={fhi:.2f}")
            continue

        y_t = bisect_root(F, lo, hi, debug=(i == 0 and len(brackets) <= 2))  # Debug first bracket if there aren't many
        if y_t is None:
            print(f"  DEBUG: Bracket {i}: bisect failed for [{lo:.6f}, {hi:.6f}]")
            # Try again with debug to see what's happening
            if i == 0:  # Only debug first failed bracket
                print(f"  DEBUG: Retrying with debug...")
                _ = bisect_root(F, lo, hi, debug=True)
            continue

        h = y_t - door_y
        m_t = sinusoid_dx_dy(y_t)
        term = 1 + m_t * m_t
        sqrt_term = np.sqrt(term)

        r1 = h * (sqrt_term * sqrt_term + abs(m_t) * sqrt_term)
        r2 = h * (sqrt_term * sqrt_term - abs(m_t) * sqrt_term)

        print(f"  DEBUG: Bracket {i}: y_t={y_t:.4f}, h={h:.4f}, m_t={m_t:.4f}, r1={r1:.4f}, r2={r2:.4f}")

        if r1 > 0 and r2 > 0:
            radius = min(r1, r2)
        elif r1 > 0:
            radius = r1
        elif r2 > 0:
            radius = r2
        else:
            print(f"  DEBUG: Bracket {i}: No positive radius")
            continue

        print(f"  DEBUG: Bracket {i}: radius={radius:.4f}")
        if best_y_t is None or y_t < best_y_t:
            best_y_t = y_t
            best_radius = radius

    if best_y_t is None:
        print(f"  DEBUG: No valid solution found. Tangent_x={tangent_x}, door_y={door_y}, side={side}")
        print(f"  DEBUG: Falling back to no door smoothing for this shelf")
        # Return None to indicate no smoothing solution
        return None, None, None, None

    # Calculate final geometry
    center_x = tangent_x
    center_y = door_y + best_radius

    return center_x, center_y, best_radius, best_y_t


def solve_door_smoothing_fixed_radius(door_y, anchor_x, depth, amplitude, period, offset, radius, side='E'):
    """
    Solve for door smoothing arc using min_q bracketing approach.

    Rotates circle center around anchor point and brackets on theta to find tangency.
    Uses q(x;theta) = (x-cx)^2 + (y_sine(x)-cy)^2 - r^2 minimization.

    Args:
        door_y: Y-coordinate of anchor point (door line, typically -0.75")
        anchor_x: X-coordinate of anchor point
        depth: Base depth of sinusoid
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Sinusoid phase offset
        radius: Fixed circle radius (default 5")
        side: 'E' for left shelf, 'W' for right shelf

    Returns:
        (center_x, center_y, radius, tangent_x, tangent_y) or (None, None, None, None, None)
    """

    # Convert from x(y) to y(x) parameterization
    # Our sinusoid is x(y), need to sample along x-axis instead
    def sinusoid_x(y):
        """X-coordinate on sinusoid at given y (original parameterization)."""
        if side == 'E':
            return depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        else:  # 'W'
            return 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

    def center_from_theta(theta):
        """Center position from rotation angle."""
        if side == 'E':
            # Left shelf: rotate from anchor rightward/upward
            return (anchor_x + radius * np.cos(theta), door_y + radius * np.sin(theta))
        else:
            # Right shelf: rotate from anchor leftward/upward (theta=0 points inward)
            return (anchor_x - radius * np.cos(theta), door_y + radius * np.sin(theta))

    def q_of_y(y, cx, cy):
        """q = (x_sine(y) - cx)^2 + (y - cy)^2 - r^2"""
        x = sinusoid_x(y)
        if not np.isfinite(x):
            return float('inf')
        dx = x - cx
        dy = y - cy
        return dx * dx + dy * dy - radius * radius

    def min_q_over_y(cx, cy, samples=1200):
        """Find minimum of q over y in [door_y, 29]."""
        y_min = max(door_y, cy - radius)
        y_max = min(29.0, cy + radius)
        if y_max <= y_min:
            return door_y, float('inf')

        # Coarse sample
        best_q = float('inf')
        best_y = y_min
        for i in range(samples + 1):
            y = y_min + (y_max - y_min) * i / samples
            q = q_of_y(y, cx, cy)
            if np.isfinite(q) and q < best_q:
                best_q = q
                best_y = y

        # Refine with golden section
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        width = (y_max - y_min) / samples
        lo = max(y_min, best_y - 5 * width)
        hi = min(y_max, best_y + 5 * width)

        for _ in range(60):
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            fc = q_of_y(c, cx, cy)
            fd = q_of_y(d, cx, cy)
            if not np.isfinite(fc):
                fc = float('inf')
            if not np.isfinite(fd):
                fd = float('inf')
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = q_of_y(c, cx, cy)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = q_of_y(d, cx, cy)

        y = 0.5 * (lo + hi)
        return y, q_of_y(y, cx, cy)

    def g(theta):
        """min_q as function of theta."""
        cx, cy = center_from_theta(theta)
        _, mq = min_q_over_y(cx, cy)
        return mq

    # Find bracket on theta where g changes sign
    theta_lo = -np.pi / 2
    theta_hi = np.pi / 2
    theta_samples = 200

    thetas = np.linspace(theta_lo, theta_hi, theta_samples + 1)
    vals = [g(t) for t in thetas]

    # Check for already near zero
    best_i = min(range(len(vals)), key=lambda i: abs(vals[i]) if np.isfinite(vals[i]) else float('inf'))
    if abs(vals[best_i]) < 1e-7:
        i = best_i
        lo = thetas[max(0, i - 1)]
        hi = thetas[min(theta_samples, i + 1)]
    else:
        # Find sign change
        found = False
        for i in range(theta_samples):
            v0, v1 = vals[i], vals[i + 1]
            if np.isfinite(v0) and np.isfinite(v1) and v0 * v1 < 0:
                lo, hi = thetas[i], thetas[i + 1]
                found = True
                break
        if not found:
            return None, None, None, None, None

    # Bisect to find theta* where g(theta*) = 0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if abs(gmid) < 1e-9 or (hi - lo) < 1e-9:
            break
        glo = g(lo)
        if glo * gmid <= 0:
            hi = mid
        else:
            lo = mid

    theta_star = 0.5 * (lo + hi)
    cx, cy = center_from_theta(theta_star)
    y_touch, mq = min_q_over_y(cx, cy, samples=2000)

    if abs(mq) > 1e-6:
        return None, None, None, None, None

    x_touch = sinusoid_x(y_touch)
    return cx, cy, radius, x_touch, y_touch


def generate_intermediate_shelf(depth, length, side, amplitude, period, offset, corner_radius=3.0,
                               door_extension=0.0, door_smoothing_tangent_x=None, door_notch_radius=0.0,
                               door_notch_intersection_x=None):
    """
    Generate a simple intermediate shelf polygon with corner radiusing at back interior corner
    and optional door-fitting features.

    Args:
        depth: Base depth in inches (7" for left, 4" for right)
        length: Length from door inward (29")
        side: 'E' for left/east wall, 'W' for right/west wall
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Random phase offset for this shelf
        corner_radius: Radius for back interior corner (default 3")
        door_extension: Distance to extend doorward (inches, default 0)
        door_smoothing_tangent_x: X-coordinate where smoothing arc touches door line (None = no smoothing)
        door_notch_radius: Radius for doorframe notch (default 0 = no notch)
        door_notch_intersection_x: X-coordinate where notch intersects y=0 line (None = no notch)

    Returns:
        Polygon array with sinusoidal interior edge, rounded corners, and door features
    """
    # Generate sinusoid points along the interior edge
    # If door features are enabled, extend sinusoid down to door line
    y_start = -door_extension if door_extension > 0 else 0
    y_points = np.linspace(y_start, length, 100)

    # Generate corner arc (quarter circle)
    arc_angles = np.linspace(0, np.pi/2, 20)

    polygon = []

    if side == 'E':
        # Left shelf: wall at x=0, sinusoid on right side
        # Back interior corner is SE (where south edge meets sinusoid)

        # Solve for back corner tangent circle
        center_x, center_y, tangent_y = solve_tangent_circle_horizontal_sinusoid(
            length, depth, amplitude, period, offset, corner_radius, side='E')

        # Calculate actual tangency points
        x_sinusoid_tangent = depth + amplitude * np.sin(2 * np.pi * tangent_y / period + offset)

        # Horizontal tangency point is directly above center
        x_horizontal_tangent = center_x
        y_horizontal_tangent = length

        # Calculate angles of tangency points from center
        angle_horizontal = np.pi/2  # Directly above center
        angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

        # Door features (if enabled)
        has_door_features = door_extension > 0 and door_smoothing_tangent_x is not None

        if has_door_features:
            # === LEFT SHELF WITH DOOR FEATURES ===
            door_y = -door_extension  # Extended door line

            # 1. Start at original door/wall corner
            polygon.append([0, 0])

            # 2. Line from wall to notch intersection along y=0 (if notch exists)
            if door_notch_radius > 0 and door_notch_intersection_x is not None:
                # Go from wall to notch point b along y=0
                polygon.append([door_notch_intersection_x, 0])

                # 3. Notch arc (concave quarter-circle) - mirror of right shelf
                # Center at (door_notch_intersection_x, door_y), radius = door_notch_radius
                # Arc from 90° (pointing up/north) to 0° (pointing right/east)
                # Point b: center + (0, r) at 90° = (door_notch_intersection_x, 0)
                # Point a: center + (r, 0) at 0° = (door_notch_intersection_x + r, door_y)
                notch_angles = np.linspace(np.pi / 2, 0, 15)
                notch_center_x = door_notch_intersection_x
                notch_center_y = door_y
                for angle in notch_angles:
                    x = notch_center_x + door_notch_radius * np.cos(angle)
                    y = notch_center_y + door_notch_radius * np.sin(angle)
                    polygon.append([x, y])

                # 4. Now at point a, continue along door line to smoothing tangent
                polygon.append([door_smoothing_tangent_x, door_y])
            else:
                # No notch, just door-line to smoothing tangent
                polygon.append([door_smoothing_tangent_x, door_y])

            # 7. Door smoothing arc (convex) - NEW FIXED-RADIUS APPROACH
            # Anchor point is 5/8" to the right of notch end point
            anchor_x_left = door_notch_intersection_x + door_notch_radius + 0.625
            min_radius = 5.0  # Default minimum radius
            smooth_center_x, smooth_center_y, smooth_radius, smooth_tangent_x, smooth_tangent_y = \
                solve_door_smoothing_fixed_radius(door_y, anchor_x_left, depth, amplitude, period, offset, min_radius, side='E')

            if smooth_center_x is not None:
                # Smoothing solution found
                # Tangency point already provided by solver
                x_smooth_sinusoid = smooth_tangent_x

                # Arc from anchor point c to sinusoid tangency
                # Left shelf: COUNTERCLOCKWISE, taking the short arc
                angle_anchor = np.arctan2(door_y - smooth_center_y, anchor_x_left - smooth_center_x)
                angle_smooth_sinusoid = np.arctan2(smooth_tangent_y - smooth_center_y,
                                                  x_smooth_sinusoid - smooth_center_x)

                # Always take the short arc (< 180°)
                angle_diff = angle_smooth_sinusoid - angle_anchor
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                smooth_arc_angles = np.linspace(angle_anchor, angle_anchor + angle_diff, 20)
                for angle in smooth_arc_angles:
                    x = smooth_center_x + smooth_radius * np.cos(angle)
                    y = smooth_center_y + smooth_radius * np.sin(angle)
                    polygon.append([x, y])

                # 8. Sinusoid edge from smoothing tangent to back corner tangent
                for y in y_points:
                    if smooth_tangent_y <= y <= tangent_y:
                        x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                        polygon.append([x, y])
            else:
                # No smoothing solution - go directly from door line to sinusoid
                print("  Warning: No door smoothing solution, using simple geometry")
                # 8. Sinusoid edge from door line to back corner tangent
                for y in y_points:
                    if door_y <= y <= tangent_y:
                        x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                        polygon.append([x, y])

            # 9. Back corner arc (from sinusoid to south edge)
            # Define sinusoid function for arc containment test
            def sinusoid_func(y):
                return depth + amplitude * np.sin(2 * np.pi * y / period + offset)

            # Calculate arc sweep (from sinusoid tangency TO horizontal tangency)
            angle_diff = angle_horizontal - angle_sinusoid
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            arc_short = angle_diff
            arc_long = arc_short - 2 * np.pi if arc_short > 0 else arc_short + 2 * np.pi

            # Test which arc stays inside shelf
            short_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_short,
                sinusoid_func, 0, length
            )
            long_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_long,
                sinusoid_func, 0, length
            )

            if short_ok and not long_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            elif long_ok and not short_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_long, 20)
            elif short_ok and long_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            else:
                raise RuntimeError("Neither arc stays inside shelf - radius may be too large")

            for angle in arc_angles_actual:
                x = center_x + corner_radius * np.cos(angle)
                y = center_y + corner_radius * np.sin(angle)
                polygon.append([x, y])

            # 10. South edge from back corner to wall
            polygon.append([0, length])

            # 11. Wall from south back to start (will close automatically)

        else:
            # === LEFT SHELF WITHOUT DOOR FEATURES (original) ===
            # 1. Start at NW corner (wall meets door)
            polygon.append([0, 0])

            # 2. Wall edge from [0, 0] to [0, length]
            polygon.append([0, length])

            # 3. South edge from wall to horizontal tangency point
            polygon.append([x_horizontal_tangent, y_horizontal_tangent])

            # 4. SE corner arc - from horizontal tangency to sinusoid tangency
            # Define sinusoid function for left shelf
            def sinusoid_func(y):
                return depth + amplitude * np.sin(2 * np.pi * y / period + offset)

            # Calculate both possible arc sweeps
            angle_diff = angle_sinusoid - angle_horizontal
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            arc_short = angle_diff
            arc_long = arc_short - 2 * np.pi if arc_short > 0 else arc_short + 2 * np.pi

            # Test which arc stays inside shelf
            short_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_horizontal, arc_short,
                sinusoid_func, 0, length
            )
            long_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_horizontal, arc_long,
                sinusoid_func, 0, length
            )

            if short_ok and not long_ok:
                arc_angles_actual = np.linspace(angle_horizontal, angle_horizontal + arc_short, 20)
            elif long_ok and not short_ok:
                arc_angles_actual = np.linspace(angle_horizontal, angle_horizontal + arc_long, 20)
            elif short_ok and long_ok:
                # Prefer shorter arc
                arc_angles_actual = np.linspace(angle_horizontal, angle_horizontal + arc_short, 20)
            else:
                raise RuntimeError("Neither arc stays inside shelf - radius may be too large")

            for angle in arc_angles_actual:
                x = center_x + corner_radius * np.cos(angle)
                y = center_y + corner_radius * np.sin(angle)
                polygon.append([x, y])

            # 5. Sinusoid edge going north (from tangent point to y=0)
            for y in y_points[::-1]:
                if y <= tangent_y:
                    x = depth + amplitude * np.sin(2 * np.pi * y / period + offset)
                    polygon.append([x, y])

            # 6. North/door edge from sinusoid back to wall
            x_north = depth + amplitude * np.sin(2 * np.pi * 0 / period + offset)
            polygon.append([x_north, 0])

    else:  # side == 'W'
        # Right shelf: wall at x=48, sinusoid on left side
        # Back interior corner is SW (where south edge meets sinusoid)

        # Solve for back corner tangent circle
        center_x, center_y, tangent_y = solve_tangent_circle_horizontal_sinusoid(
            length, depth, amplitude, period, offset, corner_radius, side='W')

        # Calculate actual tangency points
        x_sinusoid_tangent = 48 - depth - amplitude * np.sin(2 * np.pi * tangent_y / period + offset)

        # Horizontal tangency point is directly above center
        x_horizontal_tangent = center_x
        y_horizontal_tangent = length

        # Calculate angles of tangency points from center
        angle_horizontal = np.pi/2  # Directly above center
        angle_sinusoid = np.arctan2(tangent_y - center_y, x_sinusoid_tangent - center_x)

        # Door features (if enabled)
        has_door_features = door_extension > 0 and door_smoothing_tangent_x is not None

        if has_door_features:
            # === RIGHT SHELF WITH DOOR FEATURES ===
            door_y = -door_extension  # Extended door line

            # 1. Start at original door/wall corner
            polygon.append([48, 0])

            # 2. Line from wall to notch intersection along y=0 (if notch exists)
            if door_notch_radius > 0 and door_notch_intersection_x is not None:
                # Go from wall to notch point b along y=0
                polygon.append([door_notch_intersection_x, 0])

                # 4. Notch arc (concave quarter-circle)
                # Center at (door_notch_intersection_x, door_y), radius = door_notch_radius
                # Arc from 90° (pointing up/north) to 180° (pointing left/west)
                # Point a: center + (-r, 0) at 180°
                # Point b: center + (0, r) at 90°
                notch_angles = np.linspace(np.pi / 2, np.pi, 15)
                notch_center_x = door_notch_intersection_x
                notch_center_y = door_y
                for angle in notch_angles:
                    x = notch_center_x + door_notch_radius * np.cos(angle)
                    y = notch_center_y + door_notch_radius * np.sin(angle)
                    polygon.append([x, y])

                # 5. Now at point a, continue along door line to smoothing tangent
                polygon.append([door_smoothing_tangent_x, door_y])
            else:
                # No notch, just door-line to smoothing tangent
                polygon.append([door_smoothing_tangent_x, door_y])

            # 7. Door smoothing arc (convex) - NEW FIXED-RADIUS APPROACH
            # Anchor point is 5/8" to the left of notch end point
            anchor_x_right = door_notch_intersection_x - door_notch_radius - 0.625
            min_radius = 5.0  # Default minimum radius
            smooth_center_x, smooth_center_y, smooth_radius, smooth_tangent_x, smooth_tangent_y = \
                solve_door_smoothing_fixed_radius(door_y, anchor_x_right, depth, amplitude, period, offset, min_radius, side='W')

            if smooth_center_x is not None:
                # Smoothing solution found
                # Tangency point already provided by solver
                x_smooth_sinusoid = smooth_tangent_x

                # Arc from anchor point c to sinusoid tangency
                # Right shelf: CLOCKWISE, taking the short arc
                angle_anchor = np.arctan2(door_y - smooth_center_y, anchor_x_right - smooth_center_x)
                angle_smooth_sinusoid = np.arctan2(smooth_tangent_y - smooth_center_y,
                                                  x_smooth_sinusoid - smooth_center_x)

                # Always take the short arc (< 180°)
                angle_diff = angle_smooth_sinusoid - angle_anchor
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                smooth_arc_angles = np.linspace(angle_anchor, angle_anchor + angle_diff, 20)
                for angle in smooth_arc_angles:
                    x = smooth_center_x + smooth_radius * np.cos(angle)
                    y = smooth_center_y + smooth_radius * np.sin(angle)
                    polygon.append([x, y])

                # 8. Sinusoid edge from smoothing tangent to back corner tangent
                for y in y_points:
                    if smooth_tangent_y <= y <= tangent_y:
                        x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                        polygon.append([x, y])
            else:
                # No smoothing solution - go directly from door line to sinusoid
                print("  Warning: No door smoothing solution, using simple geometry")
                # 8. Sinusoid edge from door line to back corner tangent
                for y in y_points:
                    if door_y <= y <= tangent_y:
                        x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                        polygon.append([x, y])

            # 9. Back corner arc (from sinusoid to south edge)
            # Define sinusoid function for arc containment test
            def sinusoid_func(y):
                return 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

            # Calculate arc sweep
            angle_diff = angle_horizontal - angle_sinusoid
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            arc_short = angle_diff
            arc_long = arc_short - 2 * np.pi if arc_short > 0 else arc_short + 2 * np.pi

            # Test which arc stays inside shelf
            short_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_short,
                sinusoid_func, 0, length
            )
            long_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_long,
                sinusoid_func, 0, length
            )

            if short_ok and not long_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            elif long_ok and not short_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_long, 20)
            elif short_ok and long_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            else:
                raise RuntimeError("Neither arc stays inside shelf - radius may be too large")

            for angle in arc_angles_actual:
                x = center_x + corner_radius * np.cos(angle)
                y = center_y + corner_radius * np.sin(angle)
                polygon.append([x, y])

            # 10. South edge from back corner to wall
            polygon.append([48, length])

            # 11. Wall from south back to start (will close automatically)

        else:
            # === RIGHT SHELF WITHOUT DOOR FEATURES (original) ===
            # 1. Start at NE corner (wall meets door)
            polygon.append([48, 0])

            # 2. North/door edge from wall to sinusoid
            x_north = 48 - depth - amplitude * np.sin(2 * np.pi * 0 / period + offset)
            polygon.append([x_north, 0])

            # 3. Sinusoid edge going south (from y=0 to tangent point)
            for y in y_points:
                if y <= tangent_y:
                    x = 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)
                    polygon.append([x, y])

            # 4. SW corner arc - from sinusoid tangency to horizontal tangency
            # Define sinusoid function for right shelf
            def sinusoid_func(y):
                return 48 - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

            # Calculate both possible arc sweeps
            angle_diff = angle_horizontal - angle_sinusoid
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            arc_short = angle_diff
            arc_long = arc_short - 2 * np.pi if arc_short > 0 else arc_short + 2 * np.pi

            # Test which arc stays inside shelf
            short_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_short,
                sinusoid_func, 0, length
            )
            long_ok = arc_is_inside_shelf(
                center_x, center_y, corner_radius,
                angle_sinusoid, arc_long,
                sinusoid_func, 0, length
            )

            if short_ok and not long_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            elif long_ok and not short_ok:
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_long, 20)
            elif short_ok and long_ok:
                # Prefer shorter arc
                arc_angles_actual = np.linspace(angle_sinusoid, angle_sinusoid + arc_short, 20)
            else:
                raise RuntimeError("Neither arc stays inside shelf - radius may be too large")

            for angle in arc_angles_actual:
                x = center_x + corner_radius * np.cos(angle)
                y = center_y + corner_radius * np.sin(angle)
                polygon.append([x, y])

            # 5. South edge from horizontal tangency to wall
            polygon.append([48, length])

            # 6. Wall edge from [48, length] back to [48, 0]
            # (will close automatically)

    return np.array(polygon)


def extract_exact_shelf_geometries(config, level):
    """
    Extract the EXACT geometry for each shelf piece using the same calculations
    as the visualization code.

    Returns dict with 'E', 'S', 'W' keys containing polygon point arrays.
    """
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    if not all([left_shelf, back_shelf, right_shelf]):
        return None

    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']
    radius = config.design_params['interior_corner_radius']

    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    left_params = (left_depth, amplitude, period, left_offset)
    right_params = (right_depth, amplitude, period, right_offset)
    back_params = (back_depth, amplitude, period, back_offset)

    # Generate sinusoid curves - EXACT same as in generate_shelves.py
    left_curve_wall = generate_sinusoid_points(0, pantry_depth, left_depth,
                                               amplitude, period, left_offset, num_points=200)
    back_curve_wall = generate_sinusoid_points(0, pantry_width, back_depth,
                                               amplitude, period, back_offset, num_points=200)
    right_curve_wall = generate_sinusoid_points(0, pantry_depth, right_depth,
                                                amplitude, period, right_offset, num_points=200)

    left_curve = np.array([wall_to_pantry_coords(pos, depth, 'E', pantry_width, pantry_depth)
                          for pos, depth in left_curve_wall])
    back_curve = np.array([wall_to_pantry_coords(pos, depth, 'S', pantry_width, pantry_depth)
                          for pos, depth in back_curve_wall])
    right_curve = np.array([wall_to_pantry_coords(pos, depth, 'W', pantry_width, pantry_depth)
                           for pos, depth in right_curve_wall])

    # Solve for corner arcs - EXACT same as in generate_shelves.py
    try:
        lb_center, lb_point1, lb_point2, lb_pos1, lb_pos2 = solve_tangent_circle_two_sinusoids(
            pos1_init=pantry_depth - 10,
            base_depth1=left_depth,
            amplitude1=amplitude,
            period1=period,
            offset1=left_offset,
            pos2_init=10,
            base_depth2=back_depth,
            amplitude2=amplitude,
            period2=period,
            offset2=back_offset,
            radius=radius,
            wall1_type='L',
            wall2_type='B',
            pantry_width=pantry_width,
            pantry_depth=pantry_depth,
            corner_type='left-back',
            left_params=left_params,
            right_params=right_params,
            back_params=back_params
        )
        lb_arc_original = generate_circle_arc(lb_center, lb_point1, lb_point2, num_points=30, interior_arc=True)
        # Reverse arc so it goes from door (lb_point1) to back (lb_point2) for side shelves
        # generate_circle_arc may swap points for counter-clockwise ordering
        if np.linalg.norm(lb_arc_original[0] - lb_point1) > 0.1:
            lb_arc = lb_arc_original[::-1]
        else:
            lb_arc = lb_arc_original
        lb_cut_y = (lb_point1[1] + lb_point2[1]) / 2
    except Exception as e:
        print(f"Error solving left-back corner: {e}")
        return None

    try:
        rb_center, rb_point1, rb_point2, rb_pos1, rb_pos2 = solve_tangent_circle_two_sinusoids(
            pos1_init=pantry_depth - 10,
            base_depth1=right_depth,
            amplitude1=amplitude,
            period1=period,
            offset1=right_offset,
            pos2_init=pantry_width - 10,
            base_depth2=back_depth,
            amplitude2=amplitude,
            period2=period,
            offset2=back_offset,
            radius=radius,
            wall1_type='R',
            wall2_type='B',
            pantry_width=pantry_width,
            pantry_depth=pantry_depth,
            corner_type='right-back',
            left_params=left_params,
            right_params=right_params,
            back_params=back_params
        )
        rb_arc_original = generate_circle_arc(rb_center, rb_point1, rb_point2, num_points=30, interior_arc=True)
        # Reverse arc so it goes from door (rb_point1) to back (rb_point2) for side shelves
        if np.linalg.norm(rb_arc_original[0] - rb_point1) > 0.1:
            rb_arc = rb_arc_original[::-1]
        else:
            rb_arc = rb_arc_original
        rb_cut_y = (rb_point1[1] + rb_point2[1]) / 2
    except Exception as e:
        print(f"Error solving right-back corner: {e}")
        return None

    # Now build the THREE shelf pieces by cutting at the horizontal lines
    # Key: Stop sinusoids at tangency points (lb_point1, lb_point2, rb_point1, rb_point2)

    # Helper function to find closest point in curve to target
    def find_closest_index(curve, target_point):
        distances = np.linalg.norm(curve - target_point, axis=1)
        return np.argmin(distances)

    # LEFT (East) SHELF POLYGON
    # Keep arc articulation, find intersection with horizontal cut line
    left_polygon = []
    left_polygon.append([0, 0])  # 1. NW corner (door side)

    # 2. Add sinusoid points BEFORE tangency point
    lb1_idx = find_closest_index(left_curve, lb_point1)
    for i in range(lb1_idx):  # Stop BEFORE lb1_idx (don't include tangency point)
        left_polygon.append(left_curve[i])

    # 3. Add arc points STRICTLY BELOW cut line, then interpolate exact intersection
    # Skip first arc point (tangency)
    for i, point in enumerate(lb_arc):
        if i > 0 and point[1] < lb_cut_y - 0.01:  # Strictly below cut line
            left_polygon.append(point)

    # 4. Find EXACT intersection of arc with horizontal line at y=lb_cut_y
    # Find consecutive arc points that straddle the cut line
    intersection_found = False
    for i in range(1, len(lb_arc)):
        if lb_arc[i-1][1] < lb_cut_y <= lb_arc[i][1]:
            # Interpolate between lb_arc[i-1] and lb_arc[i]
            p1, p2 = lb_arc[i-1], lb_arc[i]
            t = (lb_cut_y - p1[1]) / (p2[1] - p1[1] + 1e-10)
            intersection_x = p1[0] + t * (p2[0] - p1[0])
            left_polygon.append([intersection_x, lb_cut_y])
            intersection_found = True
            break

    if not intersection_found:
        # Fallback: use last arc point at cut line
        for i in range(len(lb_arc) - 1, 0, -1):
            if lb_arc[i][1] <= lb_cut_y:
                left_polygon.append([lb_arc[i][0], lb_cut_y])
                break

    # 5. Add wall point at cut line
    left_polygon.append([0, lb_cut_y])

    left_polygon = np.array(left_polygon)

    # RIGHT (West) SHELF POLYGON
    # Keep arc articulation, find intersection with horizontal cut line
    right_polygon = []
    right_polygon.append([pantry_width, 0])  # 1. NE corner (door side)

    # 2. Add sinusoid points BEFORE tangency point
    rb1_idx = find_closest_index(right_curve, rb_point1)
    for i in range(rb1_idx):  # Stop BEFORE rb1_idx (don't include tangency point)
        right_polygon.append(right_curve[i])

    # 3. Add arc points STRICTLY BELOW cut line, then interpolate exact intersection
    # Skip first arc point (tangency)
    for i, point in enumerate(rb_arc):
        if i > 0 and point[1] < rb_cut_y - 0.01:  # Strictly below cut line
            right_polygon.append(point)

    # 4. Find EXACT intersection of arc with horizontal line at y=rb_cut_y
    # Find consecutive arc points that straddle the cut line
    intersection_found = False
    for i in range(1, len(rb_arc)):
        if rb_arc[i-1][1] < rb_cut_y <= rb_arc[i][1]:
            # Interpolate between rb_arc[i-1] and rb_arc[i]
            p1, p2 = rb_arc[i-1], rb_arc[i]
            t = (rb_cut_y - p1[1]) / (p2[1] - p1[1] + 1e-10)
            intersection_x = p1[0] + t * (p2[0] - p1[0])
            right_polygon.append([intersection_x, rb_cut_y])
            intersection_found = True
            break

    if not intersection_found:
        # Fallback: use last arc point at cut line
        for i in range(len(rb_arc) - 1, 0, -1):
            if rb_arc[i][1] <= rb_cut_y:
                right_polygon.append([rb_arc[i][0], rb_cut_y])
                break

    # 5. Add wall point at cut line
    right_polygon.append([pantry_width, rb_cut_y])

    right_polygon = np.array(right_polygon)

    # BACK (South) SHELF POLYGON
    # Exact vertex order (going clockwise from left cut):
    # 1. (0, lb_cut_y) - left wall at cut
    # 2. (0, pantry_depth) - SW corner
    # 3. (pantry_width, pantry_depth) - SE corner
    # 4. (pantry_width, rb_cut_y) - right wall at cut
    # 5. Arc points from rb_point2 back toward cut (reversed, above cut) - STOP BEFORE rb_point2
    # 6. Sinusoid points from rb_point2 to lb_point2 (reversed) - arcs provide endpoints
    # 7. Arc points from lb_point2 toward cut (above cut) - STOP BEFORE lb_point2
    # 8. Close
    back_polygon = []

    # 1. Left wall at cut
    back_polygon.append([0, lb_cut_y])

    # 2. SW corner
    back_polygon.append([0, pantry_depth])

    # 3. SE corner (across back wall)
    back_polygon.append([pantry_width, pantry_depth])

    # 4. Right wall at cut
    back_polygon.append([pantry_width, rb_cut_y])

    # 5. Right arc points from cut going up to rb_point2
    # Check which direction the arc goes - sometimes it's point1→point2, sometimes point2→point1
    # For back shelf, we need the arc to go from rb_point1 (at cut) to rb_point2 (at sinusoid)
    if np.linalg.norm(rb_arc_original[0] - rb_point1) < 0.1:
        # Arc already goes from point1 to point2, use as-is
        rb_arc_for_back = rb_arc_original
    else:
        # Arc goes from point2 to point1, reverse it
        rb_arc_for_back = rb_arc_original[::-1]

    for point in rb_arc_for_back:
        if point[1] >= rb_cut_y - 0.1:  # Above cut line
            back_polygon.append(point)

    # 6. Back sinusoid from rb_point2 to lb_point2 (REVERSED)
    # Find indices
    lb2_idx = find_closest_index(back_curve, lb_point2)
    rb2_idx = find_closest_index(back_curve, rb_point2)

    # Add points from rb2 to lb2 (going from right to left)
    for i in range(rb2_idx, lb2_idx - 1, -1):  # Reversed range
        back_polygon.append(back_curve[i])

    # 7. Left arc points from lb_point2 back down to cut
    # Use original arc orientation (before the fix for side shelves), forward direction
    for point in lb_arc_original:
        if point[1] >= lb_cut_y - 0.1:  # Above cut line
            back_polygon.append(point)

    back_polygon = np.array(back_polygon)

    return {
        'E': left_polygon,
        'S': back_polygon,
        'W': right_polygon,
        'lb_cut_y': lb_cut_y,
        'rb_cut_y': rb_cut_y,
        'lb_arc': lb_arc,
        'rb_arc': rb_arc,
        'left_curve': left_curve,
        'back_curve': back_curve,
        'right_curve': right_curve
    }


def export_polygon_to_svg(polygon, filepath, shelf_height, wall_name, width_in, height_in):
    """Export a polygon to SVG using exact coordinates.

    Args:
        polygon: Polygon vertices
        filepath: Output file path
        shelf_height: Bottom-referenced height in inches
        wall_name: 'L', 'R', or 'B' for left/right/back
        width_in: Polygon width
        height_in: Polygon height
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Get bounds
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Add margin
    margin = 5
    viewbox_width = width + 2 * margin
    viewbox_height = height + 2 * margin
    viewbox_x = min_x - margin
    viewbox_y = min_y - margin

    # Create SVG with correct viewBox
    svg_lines = []
    svg_lines.append(f'<?xml version="1.0" encoding="utf-8" ?>')
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" ')
    svg_lines.append(f'     width="{viewbox_width}in" height="{viewbox_height}in" ')
    svg_lines.append(f'     viewBox="{viewbox_x} {viewbox_y} {viewbox_width} {viewbox_height}">')

    # Add title
    svg_lines.append(f'  <title>Shelf {wall_name}{shelf_height}" - {width:.2f}" x {height:.2f}"</title>')

    # Build path data
    path_data = f'M {polygon[0, 0]},{polygon[0, 1]}'
    for point in polygon[1:]:
        path_data += f' L {point[0]},{point[1]}'
    path_data += ' Z'  # Close path

    # Add polygon as path
    svg_lines.append(f'  <path d="{path_data}" ')
    svg_lines.append(f'        fill="lightblue" stroke="black" stroke-width="0.1" />')

    # Add dimension text
    svg_lines.append(f'  <text x="{min_x}" y="{min_y - 2}" font-size="2" font-family="Arial">')
    svg_lines.append(f'    Shelf {wall_name}{shelf_height}"')
    svg_lines.append(f'  </text>')
    svg_lines.append(f'  <text x="{min_x}" y="{min_y - 0.5}" font-size="1.5" font-family="Arial">')
    svg_lines.append(f'    {width:.2f}" × {height:.2f}"')
    svg_lines.append(f'  </text>')

    svg_lines.append('</svg>')

    # Write file
    with open(filepath, 'w') as f:
        f.write('\n'.join(svg_lines))


def export_polygon_to_dxf(polygon, filepath, shelf_height, wall_name, width_in, height_in):
    """Export a polygon to DXF format for Fusion 360.

    Args:
        polygon: Polygon vertices
        filepath: Output file path
        shelf_height: Bottom-referenced height in inches
        wall_name: 'L', 'R', or 'B' for left/right/back
        width_in: Polygon width
        height_in: Polygon height
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DXF document
    doc = ezdxf.new('R2010', setup=True)
    doc.units = units.IN  # Set units to inches

    msp = doc.modelspace()

    # Create polyline from polygon vertices
    # Convert polygon points to 3D points (z=0)
    points_3d = [(float(x), float(y), 0.0) for x, y in polygon]

    # Add closed polyline
    polyline = msp.add_lwpolyline(points_3d, close=True)
    polyline.dxf.layer = "OUTLINE"

    # Calculate centroid for label placement
    centroid_x = float(np.mean(polygon[:, 0]))
    centroid_y = float(np.mean(polygon[:, 1]))

    # Get bounds
    min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
    min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
    width = max_x - min_x
    height = max_y - min_y

    # Calculate a safe font size - small enough to not touch edges
    # Use 5% of the minimum dimension, capped at 0.5 inches
    safe_size = min(0.5, min(width, height) * 0.05)

    # Create label text
    label_text = f"{wall_name}{shelf_height}\""

    # Add text at centroid
    text = msp.add_text(
        label_text,
        dxfattribs={
            'layer': 'LABELS',
            'height': safe_size,
            'style': 'Standard',
        }
    )
    text.set_placement((centroid_x, centroid_y, 0.0), align=TextEntityAlignment.MIDDLE_CENTER)

    # Save the DXF file
    doc.saveas(filepath)


def export_all_shelves_to_combined_dxf(placements, filepath, bin_width=96, bin_height=48):
    """Export all shelves to a single DXF file with non-overlapping bounding boxes.

    Args:
        placements: List of (bin_num, polygon, x, y, rotated, shelf_height, wall_name) tuples
        filepath: Output DXF file path
        bin_width: Width of each plywood sheet
        bin_height: Height of each plywood sheet
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DXF document
    doc = ezdxf.new('R2010', setup=True)
    doc.units = units.IN  # Set units to inches

    msp = doc.modelspace()

    # Group placements by bin number
    bins = {}
    for bin_num, poly, x, y, rotated, shelf_height, wall_name in placements:
        if bin_num not in bins:
            bins[bin_num] = []
        bins[bin_num].append((poly, x, y, rotated, shelf_height, wall_name))

    # Place each bin side by side with spacing
    bin_spacing = 4  # 4" spacing between sheets
    current_x_offset = 0

    for bin_num in sorted(bins.keys()):
        bin_placements = bins[bin_num]

        # Add each polygon in this bin
        for poly, x, y, rotated, shelf_height, wall_name in bin_placements:
            # Position polygon with bin offset
            poly_placed = poly + np.array([x + current_x_offset, y])

            # Convert to 3D points
            points_3d = [(float(px), float(py), 0.0) for px, py in poly_placed]

            # Add closed polyline
            polyline = msp.add_lwpolyline(points_3d, close=True)
            polyline.dxf.layer = "OUTLINE"

        # Move to next bin position
        current_x_offset += bin_width + bin_spacing

    # Save the DXF file
    doc.saveas(filepath)

    return len(bins)


def simple_2d_pack(polygons, bin_width=96, bin_height=48):
    """
    Simple shelf-by-shelf packing.

    Args:
        polygons: List of (polygon, shelf_height, wall_name) tuples

    Returns:
        List of (bin_num, polygon, x, y, rotated, shelf_height, wall_name) tuples
    """
    placements = []
    current_bin = 0
    current_y = 0
    row_height = 0
    current_x = 0

    for polygon, shelf_height, wall_name in polygons:
        # Get bounds
        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        # Normalize polygon to origin
        norm_poly = polygon - np.array([min_x, min_y])

        # Check if fits in current row
        if current_x + width > bin_width:
            # Move to next row
            current_x = 0
            current_y += row_height + 2  # 2" spacing
            row_height = 0

            # Check if need new bin
            if current_y + height > bin_height:
                current_bin += 1
                current_y = 0
                row_height = 0

        # Place piece
        placements.append((current_bin, norm_poly, current_x, current_y, False, shelf_height, wall_name))

        # Update trackers
        current_x += width + 2  # 2" spacing
        row_height = max(row_height, height)

    return placements


def create_overview_page(ax, config, level, geom_data):
    """Create overview page matching generate_shelves.py exactly."""
    pantry_width = config.pantry['width']
    pantry_depth = config.pantry['depth']
    period = config.design_params['sinusoid_period']
    amplitude = config.design_params['sinusoid_amplitude']

    # Draw pantry outline
    pantry_outline = np.array([
        [0, 0], [pantry_width, 0], [pantry_width, pantry_depth],
        [0, pantry_depth], [0, 0]
    ])
    ax.plot(pantry_outline[:, 0], pantry_outline[:, 1], 'k-', linewidth=3)

    # Get shelves
    left_shelf = config.get_shelf(level, 'E')
    back_shelf = config.get_shelf(level, 'S')
    right_shelf = config.get_shelf(level, 'W')

    left_depth = config.design_params['shelf_base_depth_east']
    back_depth = config.design_params['shelf_base_depth_south']
    right_depth = config.design_params['shelf_base_depth_west']

    left_offset = left_shelf['sinusoid_offset']
    back_offset = back_shelf['sinusoid_offset']
    right_offset = right_shelf['sinusoid_offset']

    # Generate interior mask
    X, Y, interior_mask = generate_interior_mask(
        left_depth, amplitude, period, left_offset,
        right_depth, amplitude, period, right_offset,
        back_depth, amplitude, period, back_offset,
        pantry_width, pantry_depth,
        resolution=150
    )

    ax.contourf(X, Y, ~interior_mask, levels=[0.5, 1.5], colors=['#FFE4B5'], alpha=0.4)
    ax.contourf(X, Y, interior_mask, levels=[0.5, 1.5], colors=['#E0F7FA'], alpha=0.3)

    # Plot curves using EXACT same data
    ax.plot(geom_data['left_curve'][:, 0], geom_data['left_curve'][:, 1],
           'b-', linewidth=3, label='Left shelf', alpha=0.8)
    ax.plot(geom_data['back_curve'][:, 0], geom_data['back_curve'][:, 1],
           'r-', linewidth=3, label='Back shelf', alpha=0.8)
    ax.plot(geom_data['right_curve'][:, 0], geom_data['right_curve'][:, 1],
           'g-', linewidth=3, label='Right shelf', alpha=0.8)

    # Plot arcs
    ax.plot(geom_data['lb_arc'][:, 0], geom_data['lb_arc'][:, 1],
           'c-', linewidth=4, label='Corner arcs', alpha=0.9)
    ax.plot(geom_data['rb_arc'][:, 0], geom_data['rb_arc'][:, 1],
           'm-', linewidth=4, alpha=0.9)

    # Draw cut lines - ONLY from arc to wall
    lb_cut_y = geom_data['lb_cut_y']
    rb_cut_y = geom_data['rb_cut_y']

    # Find where arcs intersect cut lines
    lb_arc_x = geom_data['lb_arc'][-1][0]  # Last point of arc at cut
    rb_arc_x = geom_data['rb_arc'][-1][0]  # Last point of arc at cut

    ax.plot([0, lb_arc_x], [lb_cut_y, lb_cut_y], 'r--', linewidth=2, alpha=0.8, label='Cut lines')
    ax.plot([rb_arc_x, pantry_width], [rb_cut_y, rb_cut_y], 'r--', linewidth=2, alpha=0.8)

    height = left_shelf['height']
    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=12)
    ax.set_ylabel('Y (inches)', fontsize=12)
    ax.set_title(f'Level {level} - Height: {height:.1f}" - Complete Assembly',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)


def create_height_level_page(ax, height, shelves_at_height, pantry_width=48, pantry_depth=49):
    """Create visualization page showing all shelves at a specific height.

    Args:
        ax: Matplotlib axes
        height: Bottom-referenced height in inches
        shelves_at_height: Dict with keys 'left', 'right', 'back' (each optional) containing polygons
        pantry_width: Pantry width in inches
        pantry_depth: Pantry depth in inches
    """
    # Draw full pantry outline
    ax.plot([0, pantry_width, pantry_width, 0, 0],
           [0, 0, pantry_depth, pantry_depth, 0],
           'k-', linewidth=2, label='Pantry walls')

    # Draw left shelf if present
    if 'left' in shelves_at_height:
        left_poly = shelves_at_height['left']
        ax.plot(left_poly[:, 0], left_poly[:, 1], 'b-', linewidth=2, label='Left shelf')
        ax.fill(left_poly[:, 0], left_poly[:, 1], color='blue', alpha=0.3)

    # Draw right shelf if present
    if 'right' in shelves_at_height:
        right_poly = shelves_at_height['right']
        ax.plot(right_poly[:, 0], right_poly[:, 1], 'g-', linewidth=2, label='Right shelf')
        ax.fill(right_poly[:, 0], right_poly[:, 1], color='green', alpha=0.3)

    # Draw back shelf if present
    if 'back' in shelves_at_height:
        back_poly = shelves_at_height['back']
        ax.plot(back_poly[:, 0], back_poly[:, 1], 'r-', linewidth=2, label='Back shelf')
        ax.fill(back_poly[:, 0], back_poly[:, 1], color='red', alpha=0.3)

    # Add door edge indicator
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Door edge')

    # Add 29" depth line if we have intermediate shelves
    if 'left' in shelves_at_height or 'right' in shelves_at_height:
        if 'back' not in shelves_at_height:
            ax.axhline(y=29, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Intermediate depth (29")')

    ax.set_xlim(-2, pantry_width + 2)
    ax.set_ylim(-2, pantry_depth + 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (inches)', fontsize=11)
    ax.set_ylabel('Y (inches)', fontsize=11)
    ax.set_title(f'Shelf Level at {height}" Height (full pantry view)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)


def create_plywood_page(ax, bin_num, placements, bin_width=96, bin_height=48):
    """Create plywood layout page with EXACT geometry at life-size scale."""
    # Draw sheet outline (thinner for printing)
    sheet_rect = mpatches.Rectangle(
        (0, 0), bin_width, bin_height,
        linewidth=0.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(sheet_rect)

    colors = {'L': 'blue', 'R': 'green', 'B': 'red'}

    for b, poly, x, y, rotated, shelf_height, wall_name in placements:
        if b != bin_num:
            continue

        # Position polygon
        poly_placed = poly + np.array([x, y])

        # Draw outline (no fill to avoid obscuring cut lines)
        color = colors.get(wall_name, 'gray')
        ax.plot(poly_placed[:, 0], poly_placed[:, 1], color=color, linewidth=0.8, alpha=0.8)

        # Label - position offset from center toward interior to avoid edges
        center_x = np.mean(poly_placed[:, 0])
        center_y = np.mean(poly_placed[:, 1])

        # Find bounding box to offset label inward
        min_x, max_x = np.min(poly_placed[:, 0]), np.max(poly_placed[:, 0])
        min_y, max_y = np.min(poly_placed[:, 1]), np.max(poly_placed[:, 1])
        width = max_x - min_x
        height = max_y - min_y

        # Offset label toward upper-left interior (away from typical cut edges)
        label_x = center_x - width * 0.15
        label_y = center_y + height * 0.15

        # Create label with wall name and height
        label_text = f"{wall_name}{shelf_height}\""

        ax.text(label_x, label_y, label_text,
               fontsize=8, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, alpha=0.9, linewidth=0.5))

    ax.set_xlim(0, bin_width)
    ax.set_ylim(0, bin_height)
    ax.set_aspect('equal')
    ax.grid(False)  # No grid for cleaner printing
    ax.axis('off')  # Remove axes for clean template

    # Add sheet number label in corner
    ax.text(2, bin_height - 2, f'Sheet {bin_num + 1}',
           fontsize=12, fontweight='bold', ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='black', linewidth=1))


def main():
    print("="*60)
    print("Extracting and Exporting Exact Geometry - Height-Based Organization")
    print("="*60)

    config_path = Path('configs/pantry_0002.json')
    print(f"Loading config: {config_path}\n")
    config = ShelfConfig.from_file(config_path)

    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define specific heights for each shelf type (bottom-referenced)
    main_shelf_heights = [19, 39, 59, 79]  # Full L-shaped shelves with left-right-back
    left_intermediate_heights = [9, 29, 49, 69]  # Left side only, 7" depth
    right_intermediate_heights = [5, 13, 26, 33, 46, 53, 66, 73, 86]  # Right side only, 4" depth

    # Get the levels from config (assuming they match the main shelf count)
    levels = sorted(set(s['level'] for s in config.shelves))

    # Extract geometry parameters
    amplitude = config.design_params['sinusoid_amplitude']
    period = config.design_params['sinusoid_period']
    left_depth = config.design_params['shelf_base_depth_east']
    right_depth = config.design_params['shelf_base_depth_west']
    shelf_length = 29.0  # 29" from door inward for intermediate shelves
    corner_radius = config.design_params['interior_corner_radius']

    # Extract door fitting parameters
    door_extension = config.design_params.get('door_extension', 0.0)
    door_smoothing_tangent_x_east = config.design_params.get('door_smoothing_tangent_x_east', None)
    door_smoothing_tangent_x_west_dist = config.design_params.get('door_smoothing_tangent_x_west', None)
    door_notch_radius = config.design_params.get('door_notch_radius', 0.0)
    door_notch_intersection_x_east = config.design_params.get('door_notch_intersection_x_east', None)
    door_notch_intersection_x_west_dist = config.design_params.get('door_notch_intersection_x_west', None)

    # Convert right shelf coordinates from wall-relative to absolute
    # Right wall is at x=48, so "3.75" from right wall = 48 - 3.75 = 44.25 absolute
    door_smoothing_tangent_x_west = 48.0 - door_smoothing_tangent_x_west_dist if door_smoothing_tangent_x_west_dist is not None else None
    door_notch_intersection_x_west = 48.0 - door_notch_intersection_x_west_dist if door_notch_intersection_x_west_dist is not None else None

    # Storage for all polygons and visualizations
    all_polygons = []  # List of (polygon, height, wall_name) tuples
    shelves_by_height = {}  # Dict of {height: {'left': poly, 'right': poly, 'back': poly}}

    # =================================================================
    # MAIN SHELVES (Full L-shaped with left-right-back portions)
    # =================================================================
    print(f"Generating main shelves at heights: {main_shelf_heights}")
    for i, height in enumerate(main_shelf_heights):
        if i >= len(levels):
            print(f"  Warning: Not enough levels in config for height {height}\"")
            continue

        level = levels[i]
        print(f"  Extracting geometry for height {height}\" (config level {level})...")
        geom_data = extract_exact_shelf_geometries(config, level)

        if not geom_data:
            print(f"    Failed to extract geometry")
            continue

        # Initialize height in shelves_by_height
        if height not in shelves_by_height:
            shelves_by_height[height] = {}

        # Export SVG for each piece with height-based naming
        wall_to_name = {'E': 'L', 'S': 'B', 'W': 'R'}
        for wall in ['E', 'S', 'W']:
            polygon = geom_data[wall]
            wall_name = wall_to_name[wall]

            min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
            min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
            width = max_x - min_x
            poly_height = max_y - min_y

            # Export SVG with height-based naming: shelf_L19_exact.svg
            svg_path = output_dir / f'shelf_{wall_name}{height}_exact.svg'
            export_polygon_to_svg(polygon, svg_path, height, wall_name, width, poly_height)
            print(f"    Exported SVG: {svg_path}")

            # Export DXF with height-based naming: shelf_L19_exact.dxf
            dxf_path = output_dir / f'shelf_{wall_name}{height}_exact.dxf'
            export_polygon_to_dxf(polygon, dxf_path, height, wall_name, width, poly_height)
            print(f"    Exported DXF: {dxf_path}")

            # Add to polygons list for packing
            all_polygons.append((polygon, height, wall_name))

            # Store for visualization
            if wall == 'E':
                shelves_by_height[height]['left'] = polygon
            elif wall == 'S':
                shelves_by_height[height]['back'] = polygon
            elif wall == 'W':
                shelves_by_height[height]['right'] = polygon

    # =================================================================
    # LEFT INTERMEDIATE SHELVES (7" depth, 29" length)
    # =================================================================
    print(f"\nGenerating left intermediate shelves at heights: {left_intermediate_heights}")
    np.random.seed(42)  # For reproducible random offsets - left shelves
    for height in left_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(
            left_depth, shelf_length, 'E',
            amplitude, period, offset, corner_radius,
            door_extension=door_extension,
            door_smoothing_tangent_x=door_smoothing_tangent_x_east,
            door_notch_radius=door_notch_radius,
            door_notch_intersection_x=door_notch_intersection_x_east
        )

        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        # Export SVG: shelf_L9_exact.svg
        svg_path = output_dir / f'shelf_L{height}_exact.svg'
        export_polygon_to_svg(polygon, svg_path, height, 'L', width, poly_height)
        print(f"  Exported SVG: {svg_path}")

        # Export DXF: shelf_L9_exact.dxf
        dxf_path = output_dir / f'shelf_L{height}_exact.dxf'
        export_polygon_to_dxf(polygon, dxf_path, height, 'L', width, poly_height)
        print(f"  Exported DXF: {dxf_path}")

        # Add to polygons and visualization
        all_polygons.append((polygon, height, 'L'))
        if height not in shelves_by_height:
            shelves_by_height[height] = {}
        shelves_by_height[height]['left'] = polygon

    # =================================================================
    # RIGHT INTERMEDIATE SHELVES (4" depth, 29" length)
    # =================================================================
    print(f"\nGenerating right intermediate shelves at heights: {right_intermediate_heights}")
    np.random.seed(43)  # For reproducible random offsets - right shelves (different seed from left)
    for height in right_intermediate_heights:
        offset = np.random.uniform(0, 2 * np.pi)
        polygon = generate_intermediate_shelf(
            right_depth, shelf_length, 'W',
            amplitude, period, offset, corner_radius,
            door_extension=door_extension,
            door_smoothing_tangent_x=door_smoothing_tangent_x_west,
            door_notch_radius=door_notch_radius,
            door_notch_intersection_x=door_notch_intersection_x_west
        )

        min_x, max_x = np.min(polygon[:, 0]), np.max(polygon[:, 0])
        min_y, max_y = np.min(polygon[:, 1]), np.max(polygon[:, 1])
        width = max_x - min_x
        poly_height = max_y - min_y

        # Export SVG: shelf_R5_exact.svg
        svg_path = output_dir / f'shelf_R{height}_exact.svg'
        export_polygon_to_svg(polygon, svg_path, height, 'R', width, poly_height)
        print(f"  Exported SVG: {svg_path}")

        # Export DXF: shelf_R5_exact.dxf
        dxf_path = output_dir / f'shelf_R{height}_exact.dxf'
        export_polygon_to_dxf(polygon, dxf_path, height, 'R', width, poly_height)
        print(f"  Exported DXF: {dxf_path}")

        # Add to polygons and visualization
        all_polygons.append((polygon, height, 'R'))
        if height not in shelves_by_height:
            shelves_by_height[height] = {}
        shelves_by_height[height]['right'] = polygon

    # =================================================================
    # PLYWOOD PACKING
    # =================================================================
    print("\nPacking on plywood...")
    placements = simple_2d_pack(all_polygons)
    num_bins = max(p[0] for p in placements) + 1
    print(f"  Using {num_bins} sheet(s)")
    print(f"  Total pieces: {len(all_polygons)}")

    # =================================================================
    # COMBINED DXF EXPORT
    # =================================================================
    combined_dxf_path = output_dir / 'all_shelves_combined.dxf'
    print(f"\nGenerating combined DXF: {combined_dxf_path}")
    num_sheets = export_all_shelves_to_combined_dxf(placements, combined_dxf_path)
    print(f"  Exported {len(all_polygons)} shelves across {num_sheets} sheet(s)")

    # =================================================================
    # PDF GENERATION
    # =================================================================
    pdf_path = output_dir / 'exact_cutting_templates.pdf'
    print(f"\nGenerating PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        title_text = (
            f'Pantry Shelf Cutting Templates - EXACT GEOMETRY\n\n'
            f'Config: {config.version}\n'
            f'Pantry: {config.pantry["width"]}" × {config.pantry["depth"]}" × {config.pantry["height"]}"\n'
            f'Main Shelves: {len(main_shelf_heights)} at heights {main_shelf_heights}\n'
            f'Left Intermediate: {len(left_intermediate_heights)} at heights {left_intermediate_heights}\n'
            f'Right Intermediate: {len(right_intermediate_heights)} at heights {right_intermediate_heights}\n'
            f'Total Pieces: {len(all_polygons)}\n'
            f'Sheets: {num_bins}\n\n'
            f'Geometry extracted from exact solver\n'
            f'SVG files use identical coordinates'
        )

        ax.text(0.5, 0.5, title_text,
               transform=ax.transAxes, fontsize=12, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Visualization pages - one per height level
        print("  Generating height-level visualization pages...")
        for height in sorted(shelves_by_height.keys()):
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            create_height_level_page(ax, height, shelves_by_height[height])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            print(f"    Added visualization for height {height}\"")

        # Plywood pages - LIFE-SIZE at high resolution
        print("  Generating plywood layout pages...")
        for bin_num in range(num_bins):
            # Life-size: 96" x 48" at 150 DPI for high-quality printing
            fig = plt.figure(figsize=(96, 48), dpi=150)
            ax = fig.add_subplot(111)
            create_plywood_page(ax, bin_num, placements)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Added plywood sheet {bin_num + 1} (life-size, 150 DPI)")

    print(f"\nDone!")
    print(f"Total shelf heights: {len(shelves_by_height)}")
    print(f"Total pieces: {len(all_polygons)}")
    print("="*60)


if __name__ == '__main__':
    main()
