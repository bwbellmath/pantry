#!/usr/bin/env python3
"""
Generate shelf geometry from construction-based JSON patterns.

This script:
1. Parses shelf_level_patterns.json
2. Resolves references and evaluates calculations
3. Computes base geometry for each level
4. Executes construction sequences to generate polygons
5. Exports to SVG, DXF, and PDF
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from typing import Any, Dict, List, Tuple, Optional, Union

# Import existing solvers
from geometry import solve_tangent_circle_two_sinusoids, solve_tangent_circle_two_sinusoids_newton


# ============================================================================
# PART 1: PARSER & REFERENCE RESOLVER
# ============================================================================

class ReferenceResolver:
    """Resolves references and calculations in configuration."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}

    def resolve(self, value: Any, context: Optional[Dict] = None) -> Any:
        """
        Recursively resolve a value that might contain references or calculations.

        Args:
            value: Value to resolve (can be dict with "ref" or "calc", or plain value)
            context: Additional context for resolution (e.g., computed base geometry)

        Returns:
            Resolved value
        """
        if isinstance(value, dict):
            if "ref" in value:
                return self._resolve_reference(value["ref"], context)
            elif "calc" in value:
                return self._evaluate_calculation(value["calc"], context)
            else:
                # Recursively resolve all values in dict
                return {k: self.resolve(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(v, context) for v in value]
        else:
            return value

    def _resolve_reference(self, ref_path: str, context: Optional[Dict]) -> Any:
        """
        Resolve a reference path like "pantry_width" or "arcs.back_right_arc.center_x"

        Args:
            ref_path: Dot-separated path to value
            context: Additional context (e.g., computed base geometry)

        Returns:
            Referenced value
        """
        # Check cache first
        cache_key = f"ref:{ref_path}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try context first (for runtime-computed values)
        if context and ref_path in context:
            return context[ref_path]

        # Try common shortcuts
        shortcuts = {
            'pantry_width': 'base_dimensions.pantry_width',
            'pantry_depth': 'base_dimensions.pantry_depth',
            'pantry_height': 'base_dimensions.pantry_height',
            'door_clearance_east': 'base_dimensions.door_clearance_east',
            'door_clearance_west': 'base_dimensions.door_clearance_west',
            'door_extension': 'design_params.door_extension',
            'door_notch_radius': 'design_params.door_notch_radius',
        }

        if ref_path in shortcuts:
            ref_path = shortcuts[ref_path]

        # Handle step references (e.g., "step_5.tangent_point_y")
        if ref_path.startswith('step_') and '.tangent_point_y' in ref_path:
            # Map to last door smoothing result
            if context and 'last_door_smoothing_tangent_y' in context:
                return context['last_door_smoothing_tangent_y']
        if ref_path.startswith('step_') and '.tangent_point_x' in ref_path:
            if context and 'last_door_smoothing_tangent_x' in context:
                return context['last_door_smoothing_tangent_x']

        # Try to resolve from context using flattened path
        # e.g., "arc_midpoints.back_left_arc_midpoint.x" -> "arc_midpoint_back_left_arc_midpoint_x"
        if context:
            # Try direct match first
            if ref_path in context:
                return context[ref_path]

            # Try flattened version
            flattened = ref_path.replace('.', '_')
            if flattened in context:
                return context[flattened]

            # Try with prefixes stripped
            for prefix in ['arc_midpoints.', 'arcs.', 'intersections.']:
                if ref_path.startswith(prefix):
                    rest = ref_path[len(prefix):]
                    # Try with prefix as part of key
                    key = prefix.replace('.', '') + '_' + rest.replace('.', '_')
                    if key in context:
                        return context[key]
                    # Try without prefix
                    key2 = rest.replace('.', '_')
                    if key2 in context:
                        return context[key2]

        # Navigate through config
        parts = ref_path.split('.')
        current = self.config

        for part in parts:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                elif context and part in context:
                    current = context[part]
                else:
                    # Try to find in context with full path
                    if context and ref_path in context:
                        result = context[ref_path]
                        self.cache[cache_key] = result
                        return result
                    raise KeyError(f"Cannot resolve reference: {ref_path} (failed at {part})")
            else:
                raise KeyError(f"Cannot navigate reference: {ref_path} (not a dict at {part})")

        self.cache[cache_key] = current
        return current

    def _evaluate_calculation(self, expr: str, context: Optional[Dict]) -> float:
        """
        Evaluate a simple arithmetic expression.

        Args:
            expr: Expression like "pantry_width - door_clearance_west"
            context: Additional context for variable resolution

        Returns:
            Computed value
        """
        # Replace variable names with their values
        # This is a simple implementation - could be made more robust

        # Build a safe evaluation environment
        eval_env = {
            'pantry_width': self._resolve_reference('base_dimensions.pantry_width', context),
            'pantry_depth': self._resolve_reference('base_dimensions.pantry_depth', context),
            'pantry_height': self._resolve_reference('base_dimensions.pantry_height', context),
            'door_clearance_east': self._resolve_reference('base_dimensions.door_clearance_east', context),
            'door_clearance_west': self._resolve_reference('base_dimensions.door_clearance_west', context),
            'door_extension': self._resolve_reference('design_params.door_extension', context),
            'door_notch_radius': self._resolve_reference('design_params.door_notch_radius', context),
        }

        # Add context values
        if context:
            eval_env.update(context)

        try:
            result = eval(expr, {"__builtins__": {}}, eval_env)
            return float(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate calculation '{expr}': {e}")


# ============================================================================
# DOOR SMOOTHING SOLVER (alt_9.py approach)
# ============================================================================

def solve_door_smoothing_alt9(
    anchor_x: float,
    anchor_y: float,
    radius: float,
    depth: float,
    amplitude: float,
    period: float,
    offset: float,
    side: str = 'E',
    shelf_length: float = 49.0,
    pantry_width: float = 48.0
) -> Optional[Dict[str, Any]]:
    """
    Solve for door smoothing arc using the alt_9.py rotation approach.

    Given an anchor point on the door line, rotate a circle center around it
    and find the angle where the circle becomes tangent to the sinusoidal edge.

    Args:
        anchor_x: X-coordinate of anchor point on door line
        anchor_y: Y-coordinate of door line (typically -0.75")
        radius: Circle radius (typically 5")
        depth: Base depth of sinusoid
        amplitude: Sinusoid amplitude
        period: Sinusoid period
        offset: Sinusoid phase offset
        side: 'E' for left shelf, 'W' for right shelf
        shelf_length: Length along y-axis (default 49")
        pantry_width: Width of pantry (default 48")

    Returns:
        Dictionary with center_x, center_y, radius, tangent_x, tangent_y, theta
        or None if no solution found
    """

    # Sinusoid function
    def sine_x(y: float) -> float:
        """X-coordinate on sinusoid at given y."""
        if side == 'E':
            return depth + amplitude * np.sin(2 * np.pi * y / period + offset)
        else:  # 'W'
            return pantry_width - depth - amplitude * np.sin(2 * np.pi * y / period + offset)

    def sine_dx_dy(y: float) -> float:
        """Derivative dx/dy of sinusoid."""
        if side == 'E':
            return amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)
        else:  # 'W'
            return -amplitude * (2 * np.pi / period) * np.cos(2 * np.pi * y / period + offset)

    # Center parameterization: rotate around anchor
    def center_from_theta(theta: float) -> Tuple[float, float]:
        """
        Rotate center around anchor point.
        theta=0: center to the right of anchor
        theta=90°: center above anchor
        theta=180°: center to the left of anchor
        """
        return (anchor_x + radius * np.cos(theta), anchor_y + radius * np.sin(theta))

    # q(y; theta) = distance² from circle to curve at parameter y
    def q_of_y(y: float, cx: float, cy: float) -> float:
        """q = (x_sine(y) - cx)^2 + (y - cy)^2 - r^2"""
        x = sine_x(y)
        if not np.isfinite(x):
            return float('inf')
        dx = x - cx
        dy = y - cy
        return dx * dx + dy * dy - radius * radius

    # Golden section search for 1D minimization
    def golden_section_min(f, a: float, b: float, iters: int = 60) -> Tuple[float, float]:
        """Minimize f over [a, b] using golden section search."""
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        for _ in range(iters):
            if not np.isfinite(fc):
                fc = float('inf')
            if not np.isfinite(fd):
                fd = float('inf')
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = f(d)
        y = 0.5 * (a + b)
        return y, f(y)

    # Find minimum of q over y for a given center
    def min_q_over_y(cx: float, cy: float, samples: int = 1200) -> Tuple[float, float]:
        """
        Find minimum of q(y) over the valid y range.
        Returns (y_min, min_q)
        """
        # Search range: circle can only touch curve within circle's y-range
        y_min = max(0.0, cy - radius)
        y_max = min(shelf_length, cy + radius)

        if y_max <= y_min:
            return (y_min, float('inf'))

        # Coarse sample to find basin
        f = lambda y: q_of_y(y, cx, cy)
        best_y = y_min
        best_q = f(y_min)

        for i in range(1, samples + 1):
            y = y_min + (y_max - y_min) * i / samples
            q = f(y)
            if np.isfinite(q) and q < best_q:
                best_q = q
                best_y = y

        # Refine locally around best_y
        width = (y_max - y_min) / samples
        lo = max(y_min, best_y - 5.0 * width)
        hi = min(y_max, best_y + 5.0 * width)
        y_refined, q_refined = golden_section_min(f, lo, hi, iters=70)

        return (y_refined, q_refined)

    # Bisection root finding
    def bisect_root(g, lo: float, hi: float, max_iter: int = 70, tol: float = 1e-9) -> float:
        """Find root of g in [lo, hi] using bisection."""
        glo = g(lo)
        ghi = g(hi)

        if not (np.isfinite(glo) and np.isfinite(ghi)):
            raise RuntimeError("Non-finite g at bracket endpoints")

        if abs(glo) < tol:
            return lo
        if abs(ghi) < tol:
            return hi
        if glo * ghi > 0:
            raise RuntimeError(f"Root not bracketed: g(lo)={glo:.3e}, g(hi)={ghi:.3e}")

        a, b = lo, hi
        fa, fb = glo, ghi

        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = g(m)

            if not np.isfinite(fm):
                m = np.nextafter(m, a)
                fm = g(m)
                if not np.isfinite(fm):
                    break

            if abs(fm) < tol or (b - a) < tol:
                return m

            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return 0.5 * (a + b)

    # Find bracket for theta where min_q changes sign
    def find_theta_bracket(theta_lo: float, theta_hi: float, theta_samples: int = 200) -> Tuple[float, float]:
        """Find interval where min_q(theta) changes sign."""
        def g(theta: float) -> float:
            cx, cy = center_from_theta(theta)
            _, mq = min_q_over_y(cx, cy)
            return mq

        thetas = [theta_lo + (theta_hi - theta_lo) * i / theta_samples for i in range(theta_samples + 1)]
        vals = [g(t) for t in thetas]

        # Check if we already hit ~0
        best_i = min(range(len(vals)), key=lambda i: abs(vals[i]) if np.isfinite(vals[i]) else float('inf'))
        if abs(vals[best_i]) < 1e-7:
            i = best_i
            lo = thetas[max(0, i - 1)]
            hi = thetas[min(theta_samples, i + 1)]
            return (lo, hi)

        # Find sign change
        for i in range(theta_samples):
            v0, v1 = vals[i], vals[i + 1]
            if not (np.isfinite(v0) and np.isfinite(v1)):
                continue
            if v0 * v1 < 0:
                return (thetas[i], thetas[i + 1])

        raise RuntimeError(
            f"No sign change found in min_q(theta) over [{theta_lo:.3f}, {theta_hi:.3f}]. "
            f"Try expanding theta range or adjusting parameters."
        )

    # Main solve: find theta where min_q(theta) = 0
    try:
        # Theta range: 0 to 180 degrees to keep center above anchor line
        # theta=0: center to the right of anchor
        # theta=90: center above anchor
        # theta=180: center to the left of anchor
        theta_lo = 0
        theta_hi = np.pi

        # Find bracket
        lo, hi = find_theta_bracket(theta_lo, theta_hi, theta_samples=220)

        # Define function to minimize
        def g(theta: float) -> float:
            cx, cy = center_from_theta(theta)
            _, mq = min_q_over_y(cx, cy)
            return mq

        # Solve for theta*
        theta_star = bisect_root(g, lo, hi, max_iter=75, tol=1e-9)

        # Compute final geometry
        cx, cy = center_from_theta(theta_star)
        y_touch, mq = min_q_over_y(cx, cy, samples=2000)
        x_touch = sine_x(y_touch)

        # Verify tangency (min_q should be ~0)
        if abs(mq) > 1e-6:
            print(f"  Warning: min_q residual = {mq:.3e} (expected ~0)")
            return None

        # Verify slope matching (optional check)
        # For vertical sinusoid x(y), we need dx/dy slopes
        denom = (x_touch - cx)
        m_circle = float('inf') if abs(denom) < 1e-12 else -(y_touch - cy) / denom
        m_sine = sine_dx_dy(y_touch)
        slope_err = abs(m_sine - m_circle) if np.isfinite(m_circle) else abs(y_touch - cy)

        if slope_err > 0.01:
            print(f"  Warning: slope error = {slope_err:.3e} (expected ~0)")

        return {
            'center_x': cx,
            'center_y': cy,
            'radius': radius,
            'tangent_x': x_touch,
            'tangent_y': y_touch,
            'theta': theta_star,
            'slope_err': slope_err
        }

    except Exception as e:
        print(f"  Door smoothing solver failed: {e}")
        return None


# ============================================================================
# PART 2: BASE GEOMETRY SOLVER
# ============================================================================

class BaseGeometrySolver:
    """Computes base geometry (sinusoids, intersections, arcs) for a level."""

    def __init__(self, resolver: ReferenceResolver):
        self.resolver = resolver

    def solve_level(self, level_config: Dict[str, Any], base_level_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compute all base geometry for a level.

        Args:
            level_config: Level configuration from JSON
            base_level_config: Base level config for templates (level 0)

        Returns:
            Dictionary of computed geometry elements
        """
        context = {}
        if base_level_config:
            context['base_level_config'] = base_level_config

        base_geo = level_config.get('base_geometry', {})

        # Step 1: Define sinusoid functions
        sinusoids = self._solve_sinusoids(base_geo.get('sinusoids', {}), context)
        context.update(sinusoids)

        # Step 2: Solve intersections (inherit from base level if not specified AND level type matches)
        intersection_configs = base_geo.get('intersections', {})
        if (not intersection_configs and base_level_config and
            level_config.get('type') == base_level_config.get('type')):
            intersection_configs = base_level_config.get('base_geometry', {}).get('intersections', {})
        intersections = self._solve_intersections(intersection_configs, context)
        context.update(intersections)

        # Step 3: Solve arcs (inherit from base level if not specified AND level type matches)
        arc_configs = base_geo.get('arcs', {})
        if (not arc_configs and base_level_config and
            level_config.get('type') == base_level_config.get('type')):
            arc_configs = base_level_config.get('base_geometry', {}).get('arcs', {})
        arcs = self._solve_arcs(arc_configs, context)
        context.update(arcs)

        # Step 4: Compute arc midpoints (inherit from base level if not specified AND level type matches)
        midpoint_configs = base_geo.get('arc_midpoints', {})
        if (not midpoint_configs and base_level_config and
            level_config.get('type') == base_level_config.get('type')):
            midpoint_configs = base_level_config.get('base_geometry', {}).get('arc_midpoints', {})
        arc_midpoints = self._solve_arc_midpoints(midpoint_configs, context)
        context.update(arc_midpoints)

        return context

    def _solve_sinusoids(self, sinusoid_configs: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Create sinusoid function objects."""
        sinusoids = {}

        for name, config in sinusoid_configs.items():
            # If config only has phase_offset, merge with template from base level
            if 'phase_offset' in config and 'type' not in config:
                # This is a template reference - get base from level 0
                if 'base_level_config' in context:
                    base_sinusoids = context['base_level_config'].get('base_geometry', {}).get('sinusoids', {})
                    base_config = base_sinusoids.get(name, {})
                    # Merge: base config + current phase_offset
                    config = {**base_config, 'phase_offset': config['phase_offset']}
                else:
                    print(f"    Warning: Template reference for {name} but no base config found")
                    continue

            # Resolve all parameters
            config = self.resolver.resolve(config, context)

            # Create sinusoid function
            if config.get('type') == 'vertical':
                # x(y) - different formulas for left (E) and right (W) walls
                # Left (E): x = x_base + offset + amplitude * sin(...)
                # Right (W): x = x_base - offset - amplitude * sin(...)
                def make_vertical_sinusoid(cfg):
                    wall = cfg.get('wall', 'E')
                    if wall == 'E':  # Left wall
                        def x_of_y(y):
                            return cfg['x_base'] + cfg['offset_from_wall'] + cfg['amplitude'] * np.sin(
                                2 * np.pi * y / cfg['period'] + cfg['phase_offset']
                            )
                        def dx_dy(y):
                            return cfg['amplitude'] * (2 * np.pi / cfg['period']) * np.cos(
                                2 * np.pi * y / cfg['period'] + cfg['phase_offset']
                            )
                    else:  # Right wall (W)
                        def x_of_y(y):
                            return cfg['x_base'] - cfg['offset_from_wall'] - cfg['amplitude'] * np.sin(
                                2 * np.pi * y / cfg['period'] + cfg['phase_offset']
                            )
                        def dx_dy(y):
                            return -cfg['amplitude'] * (2 * np.pi / cfg['period']) * np.cos(
                                2 * np.pi * y / cfg['period'] + cfg['phase_offset']
                            )
                    return x_of_y, dx_dy

                x_func, dx_dy_func = make_vertical_sinusoid(config)
                sinusoids[f'sinusoid_{name}_x'] = x_func
                sinusoids[f'sinusoid_{name}_dx_dy'] = dx_dy_func
                sinusoids[f'sinusoid_{name}_config'] = config

            elif config['type'] == 'horizontal':
                # y(x) = y_base - offset_from_wall - amplitude * sin(2π*x/period + phase)
                def make_horizontal_sinusoid(cfg):
                    def y_of_x(x):
                        return cfg['y_base'] - cfg['offset_from_wall'] - cfg['amplitude'] * np.sin(
                            2 * np.pi * x / cfg['period'] + cfg['phase_offset']
                        )
                    def dy_dx(x):
                        return -cfg['amplitude'] * (2 * np.pi / cfg['period']) * np.cos(
                            2 * np.pi * x / cfg['period'] + cfg['phase_offset']
                        )
                    return y_of_x, dy_dx

                y_func, dy_dx_func = make_horizontal_sinusoid(config)
                sinusoids[f'sinusoid_{name}_y'] = y_func
                sinusoids[f'sinusoid_{name}_dy_dx'] = dy_dx_func
                sinusoids[f'sinusoid_{name}_config'] = config

        return sinusoids

    def _solve_intersections(self, intersection_configs: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Solve for intersection points between sinusoids."""
        intersections = {}

        for name, config in intersection_configs.items():
            config = self.resolver.resolve(config, context)

            if config['solve_method'] == 'two_sinusoid_intersection':
                sin1_name = config['sinusoid_1']
                sin2_name = config['sinusoid_2']

                # Get sinusoid configs
                sin1_config = context[f'sinusoid_{sin1_name}_config']
                sin2_config = context[f'sinusoid_{sin2_name}_config']

                # Solve intersection
                point = self._solve_two_sinusoid_intersection(sin1_config, sin2_config)
                intersections[f'intersection_{name}'] = point
                intersections[f'intersection_{name}_x'] = point[0]
                intersections[f'intersection_{name}_y'] = point[1]

        return intersections

    def _solve_two_sinusoid_intersection(self, sin1: Dict, sin2: Dict) -> Tuple[float, float]:
        """
        Find intersection of two sinusoids.

        For a vertical and horizontal sinusoid:
        - Vertical: x = x_base - offset - A*sin(2π*y/P + φ)
        - Horizontal: y = y_base - offset - A*sin(2π*x/P + φ)

        Solve for (x, y) where both equations are satisfied.
        """
        # Use numerical solver
        from scipy.optimize import fsolve

        if sin1['type'] == 'vertical' and sin2['type'] == 'horizontal':
            # sin1 gives x(y), sin2 gives y(x)
            def equations(vars):
                x, y = vars
                x_from_sin1 = sin1['x_base'] - sin1['offset_from_wall'] - sin1['amplitude'] * np.sin(
                    2 * np.pi * y / sin1['period'] + sin1['phase_offset']
                )
                y_from_sin2 = sin2['y_base'] - sin2['offset_from_wall'] - sin2['amplitude'] * np.sin(
                    2 * np.pi * x / sin2['period'] + sin2['phase_offset']
                )
                return [x - x_from_sin1, y - y_from_sin2]

            # Initial guess: midpoint of ranges
            x0 = (sin1['x_base'] - sin1['offset_from_wall'])
            y0 = (sin2['y_base'] - sin2['offset_from_wall'])

            solution = fsolve(equations, [x0, y0])
            return tuple(solution)

        else:
            raise NotImplementedError(f"Intersection not implemented for {sin1['type']} and {sin2['type']}")

    def _solve_arcs(self, arc_configs: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Solve for tangent arcs."""
        arcs = {}

        for name, config in arc_configs.items():
            # Don't resolve all config yet - wait until we need specific values
            solve_method = config.get('solve_method')

            if solve_method == 'slope_matching_two_sinusoids':
                # Tangent circle to two sinusoids
                result = self._solve_tangent_circle_two_sinusoids(config, context)
                sin1_name = self.resolver.resolve(config['tangent_to'][0], context)
                sin2_name = self.resolver.resolve(config['tangent_to'][1], context)

                arcs[f'arc_{name}'] = result
                arcs[f'arc_{name}_center_x'] = result['center_x']
                arcs[f'arc_{name}_center_y'] = result['center_y']

                # Store with multiple key formats for compatibility
                arcs[f'arc_{name}_tangent_point_on_{sin1_name}_x'] = result['tangent_1_x']
                arcs[f'arc_{name}_tangent_point_on_{sin1_name}_y'] = result['tangent_1_y']
                arcs[f'arc_{name}_tangent_point_on_{sin2_name}_x'] = result['tangent_2_x']
                arcs[f'arc_{name}_tangent_point_on_{sin2_name}_y'] = result['tangent_2_y']

                # Also store with "arcs_" prefix for JSON references like "arcs.back_right_arc.tangent_point_on_right.y"
                arcs[f'arcs_{name}_tangent_point_on_{sin1_name}_x'] = result['tangent_1_x']
                arcs[f'arcs_{name}_tangent_point_on_{sin1_name}_y'] = result['tangent_1_y']
                arcs[f'arcs_{name}_tangent_point_on_{sin2_name}_x'] = result['tangent_2_x']
                arcs[f'arcs_{name}_tangent_point_on_{sin2_name}_y'] = result['tangent_2_y']

            elif solve_method == 'tangent_horizontal_and_sinusoid':
                # Tangent circle to horizontal line and sinusoid
                result = self._solve_tangent_horizontal_sinusoid(config, context)
                arcs[f'arc_{name}'] = result
                arcs[f'arc_{name}_center_x'] = result['center_x']
                arcs[f'arc_{name}_center_y'] = result['center_y']
                # Store multiple reference formats for flexibility
                sin_name = config["tangent_to_sinusoid"]
                arcs[f'arc_{name}_tangent_point_on_horizontal_x'] = result['tangent_horiz_x']
                arcs[f'arc_{name}_tangent_point_on_horizontal_y'] = result['tangent_horiz_y']
                arcs[f'arc_{name}_tangent_point_on_{sin_name}_x'] = result['tangent_sin_x']
                arcs[f'arc_{name}_tangent_point_on_{sin_name}_y'] = result['tangent_sin_y']
                # Also store without underscore prefix for easier access
                arcs[f'arcs_{name}_tangent_point_on_horizontal_x'] = result['tangent_horiz_x']
                arcs[f'arcs_{name}_tangent_point_on_horizontal_y'] = result['tangent_horiz_y']
                arcs[f'arcs_{name}_tangent_point_on_{sin_name}_x'] = result['tangent_sin_x']
                arcs[f'arcs_{name}_tangent_point_on_{sin_name}_y'] = result['tangent_sin_y']

            elif solve_method == 'alt9_door_smoothing':
                # Door smoothing arc using alt_9.py rotation method
                result = self._solve_door_smoothing_alt9(config, context)
                if result is not None:
                    sin_name = config["tangent_to_sinusoid"]
                    arcs[f'arc_{name}'] = result
                    arcs[f'arc_{name}_center_x'] = result['center_x']
                    arcs[f'arc_{name}_center_y'] = result['center_y']
                    arcs[f'arc_{name}_radius'] = result['radius']
                    arcs[f'arc_{name}_tangent_x'] = result['tangent_x']
                    arcs[f'arc_{name}_tangent_y'] = result['tangent_y']
                    # Store with "arcs_" prefix for JSON references
                    arcs[f'arcs_{name}_center_x'] = result['center_x']
                    arcs[f'arcs_{name}_center_y'] = result['center_y']
                    arcs[f'arcs_{name}_tangent_point_on_{sin_name}_x'] = result['tangent_x']
                    arcs[f'arcs_{name}_tangent_point_on_{sin_name}_y'] = result['tangent_y']
                    arcs[f'arcs_{name}_anchor_x'] = result.get('anchor_x', config['anchor_point'][0])
                    arcs[f'arcs_{name}_anchor_y'] = result.get('anchor_y', config['anchor_point'][1])

        return arcs

    def _solve_tangent_circle_two_sinusoids(self, config: Dict, context: Dict) -> Dict[str, Any]:
        """Solve for circle tangent to two sinusoids."""
        sin1_name = self.resolver.resolve(config['tangent_to'][0], context)
        sin2_name = self.resolver.resolve(config['tangent_to'][1], context)
        radius = self.resolver.resolve(config['radius'], context)

        sin1_config = context[f'sinusoid_{sin1_name}_config']
        sin2_config = context[f'sinusoid_{sin2_name}_config']

        # Get intersection point as initial guess
        intersection_ref = config.get('intersection_point', {})
        if isinstance(intersection_ref, dict) and 'ref' in intersection_ref:
            int_name = intersection_ref['ref'].split('.')[-1]
            y_int = context[f'intersection_{int_name}_y']
        else:
            # Fallback: use midpoint
            y_int = 24.0

        # Map wall names to wall types
        wall_map = {'E': 'L', 'W': 'R', 'S': 'B', 'left': 'L', 'right': 'R', 'back': 'B'}
        wall1_type = wall_map.get(sin1_config['wall'], sin1_config['wall'])
        wall2_type = wall_map.get(sin2_config['wall'], sin2_config['wall'])

        # Determine corner type
        if wall1_type == 'L' and wall2_type == 'B':
            corner_type = 'left-back'
        elif wall1_type == 'R' and wall2_type == 'B':
            corner_type = 'right-back'
        else:
            corner_type = 'left-back'

        # Get pantry dimensions
        pantry_width = self.resolver.resolve({'ref': 'base_dimensions.pantry_width'}, context)
        pantry_depth = self.resolver.resolve({'ref': 'base_dimensions.pantry_depth'}, context)

        # Use new Newton solver for back corners (more robust)
        if corner_type in ['left-back', 'right-back']:
            center, tangent_1, tangent_2 = solve_tangent_circle_two_sinusoids_newton(
                base_depth1=sin1_config['offset_from_wall'],
                amplitude1=sin1_config['amplitude'],
                period1=sin1_config['period'],
                offset1=sin1_config['phase_offset'],
                wall1_type=wall1_type,
                base_depth2=sin2_config['offset_from_wall'],
                amplitude2=sin2_config['amplitude'],
                period2=sin2_config['period'],
                offset2=sin2_config['phase_offset'],
                wall2_type=wall2_type,
                radius=radius,
                pantry_width=pantry_width,
                pantry_depth=pantry_depth,
                corner_type=corner_type
            )
        else:
            # Use original solver for other corners
            result = solve_tangent_circle_two_sinusoids(
                pos1_init=y_int,
                base_depth1=sin1_config['offset_from_wall'],
                amplitude1=sin1_config['amplitude'],
                period1=sin1_config['period'],
                offset1=sin1_config['phase_offset'],
                pos2_init=y_int if wall1_type == 'L' else pantry_width / 2,
                base_depth2=sin2_config['offset_from_wall'],
                amplitude2=sin2_config['amplitude'],
                period2=sin2_config['period'],
                offset2=sin2_config['phase_offset'],
                radius=radius,
                wall1_type=wall1_type,
                wall2_type=wall2_type,
                pantry_width=pantry_width,
                pantry_depth=pantry_depth,
                corner_type=corner_type
            )
            center, tangent_1, tangent_2, pos1, pos2 = result

        return {
            'center_x': center[0],
            'center_y': center[1],
            'radius': radius,
            'tangent_1_x': tangent_1[0],
            'tangent_1_y': tangent_1[1],
            'tangent_2_x': tangent_2[0],
            'tangent_2_y': tangent_2[1]
        }

    def _solve_tangent_horizontal_sinusoid(self, config: Dict, context: Dict) -> Dict[str, Any]:
        """Solve for circle tangent to horizontal line and sinusoid."""
        sin_name = self.resolver.resolve(config['tangent_to_sinusoid'], context)
        line_y = self.resolver.resolve(config['line_y'], context)
        radius = self.resolver.resolve(config['radius'], context)

        sin_config = context[f'sinusoid_{sin_name}_config']

        # Import solver from extract_and_export_geometry
        from extract_and_export_geometry import solve_tangent_circle_horizontal_sinusoid

        side = 'E' if sin_config['wall'] == 'E' else 'W'

        print(f"    Solving tangent arc: horizontal line y={line_y} + {sin_name} sinusoid, r={radius}")
        print(f"      Sinusoid: depth={sin_config['offset_from_wall']}, amp={sin_config['amplitude']}, "
              f"period={sin_config['period']}, phase={sin_config['phase_offset']:.3f}")
        print(f"      Using algorithm: solve_tangent_circle_horizontal_sinusoid (bracketing + bisection)")

        try:
            center_x, center_y, tangent_y = solve_tangent_circle_horizontal_sinusoid(
                horizontal_y=line_y,
                depth=sin_config['offset_from_wall'],
                amplitude=sin_config['amplitude'],
                period=sin_config['period'],
                offset=sin_config['phase_offset'],
                radius=radius,
                side=side
            )

            # Calculate tangent point on sinusoid
            if side == 'E':
                tangent_x = sin_config['x_base'] + sin_config['offset_from_wall'] + sin_config['amplitude'] * np.sin(
                    2 * np.pi * tangent_y / sin_config['period'] + sin_config['phase_offset']
                )
            else:
                tangent_x = sin_config['x_base'] - sin_config['offset_from_wall'] - sin_config['amplitude'] * np.sin(
                    2 * np.pi * tangent_y / sin_config['period'] + sin_config['phase_offset']
                )

            print(f"      Solution: center=({center_x:.3f}, {center_y:.3f}), "
                  f"tangent_y={tangent_y:.3f}, tangent_x={tangent_x:.3f}")

            return {
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
                'tangent_horiz_x': center_x,
                'tangent_horiz_y': line_y,
                'tangent_sin_x': tangent_x,
                'tangent_sin_y': tangent_y
            }

        except Exception as e:
            print(f"      ERROR: Arc solver failed: {e}")
            raise

    def _solve_door_smoothing_alt9(self, config: Dict, context: Dict) -> Dict[str, Any]:
        """Solve for door smoothing arc using alt_9.py rotation method."""
        sin_name = self.resolver.resolve(config['tangent_to_sinusoid'], context)
        radius = self.resolver.resolve(config['radius'], context)
        anchor_point = self.resolver.resolve(config['anchor_point'], context)
        anchor_x, anchor_y = anchor_point[0], anchor_point[1]

        sin_config = context[f'sinusoid_{sin_name}_config']

        # Get pantry dimensions
        pantry_width = self.resolver.resolve({'ref': 'base_dimensions.pantry_width'}, context)
        pantry_depth = self.resolver.resolve({'ref': 'base_dimensions.pantry_depth'}, context)

        # Determine side
        side = 'E' if sin_config['wall'] == 'E' else 'W'

        print(f"    Solving door smoothing arc (alt_9.py): anchor=({anchor_x:.3f}, {anchor_y:.3f}), r={radius}")
        print(f"      Sinusoid: depth={sin_config['offset_from_wall']}, amp={sin_config['amplitude']}, "
              f"period={sin_config['period']}, phase={sin_config['phase_offset']:.3f}")
        print(f"      Using algorithm: alt9_door_smoothing (rotation + bracketing + bisection)")

        try:
            result = solve_door_smoothing_alt9(
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                radius=radius,
                depth=sin_config['offset_from_wall'],
                amplitude=sin_config['amplitude'],
                period=sin_config['period'],
                offset=sin_config['phase_offset'],
                side=side,
                shelf_length=pantry_depth,
                pantry_width=pantry_width
            )

            if result is None:
                print(f"      ERROR: alt9 door smoothing solver failed to find solution")
                return None

            print(f"      Solution: center=({result['center_x']:.3f}, {result['center_y']:.3f}), "
                  f"tangent=({result['tangent_x']:.3f}, {result['tangent_y']:.3f}), "
                  f"slope_err={result['slope_err']:.3e}")

            return {
                'center_x': result['center_x'],
                'center_y': result['center_y'],
                'radius': result['radius'],
                'tangent_x': result['tangent_x'],
                'tangent_y': result['tangent_y'],
                'theta': result['theta'],
                'anchor_x': anchor_x,
                'anchor_y': anchor_y
            }

        except Exception as e:
            print(f"      ERROR: alt9 door smoothing solver failed: {e}")
            return None

    def _solve_arc_midpoints(self, midpoint_configs: Dict[str, Any], context: Dict) -> Dict[str, Any]:
        """Compute arc midpoint coordinates."""
        midpoints = {}

        for name, config in midpoint_configs.items():
            position = config.get('position')

            if position == 'midpoint':
                arc_ref = config['arc']['ref']
                arc_name = arc_ref.split('.')[-1]
                arc = context[f'arc_{arc_name}']

                # Get arc angles
                center_x = arc['center_x']
                center_y = arc['center_y']
                radius = arc['radius']

                # Get tangent points
                t1_x = arc['tangent_1_x']
                t1_y = arc['tangent_1_y']
                t2_x = arc['tangent_2_x']
                t2_y = arc['tangent_2_y']

                # Calculate angles
                angle_1 = np.arctan2(t1_y - center_y, t1_x - center_x)
                angle_2 = np.arctan2(t2_y - center_y, t2_x - center_x)

                # Midpoint angle
                angle_mid = (angle_1 + angle_2) / 2

                # Midpoint coordinates
                mid_x = center_x + radius * np.cos(angle_mid)
                mid_y = center_y + radius * np.sin(angle_mid)

                # Store with "midpoints" (plural) to match JSON references
                midpoints[f'arc_midpoints_{name}'] = {'x': mid_x, 'y': mid_y}
                midpoints[f'arc_midpoints_{name}_x'] = mid_x
                midpoints[f'arc_midpoints_{name}_y'] = mid_y

        return midpoints


# ============================================================================
# PART 3: CONSTRUCTION RENDERER
# ============================================================================

class ConstructionRenderer:
    """Executes construction sequences to generate polygon coordinates."""

    def __init__(self, resolver: ReferenceResolver):
        self.resolver = resolver

    def render_shelf(self, construction: List[Dict], context: Dict) -> np.ndarray:
        """
        Execute a construction sequence to generate polygon vertices.

        Args:
            construction: List of construction steps
            context: Computed base geometry

        Returns:
            Array of [x, y] coordinates forming closed polygon
        """
        vertices = []
        current_pos = None

        for step in construction:
            step_type = step['type']

            if step_type == 'point':
                coords = self.resolver.resolve(step['coords'], context)
                current_pos = coords
                vertices.append(coords)

            elif step_type == 'straight_line':
                to_coords = self.resolver.resolve(step['to'], context)
                vertices.append(to_coords)
                current_pos = to_coords

            elif step_type == 'arc':
                # TODO: Handle solved arcs
                if 'solve_method' in step:
                    arc_vertices = self._render_solved_arc(step, context, current_pos)
                else:
                    arc_vertices = self._render_explicit_arc(step, context)
                vertices.extend(arc_vertices)
                current_pos = arc_vertices[-1] if arc_vertices else current_pos

            elif step_type == 'sinusoid_segment':
                sin_vertices = self._render_sinusoid_segment(step, context)
                vertices.extend(sin_vertices)
                current_pos = sin_vertices[-1] if sin_vertices else current_pos

            elif step_type == 'arc_segment':
                arc_vertices = self._render_arc_segment(step, context)
                vertices.extend(arc_vertices)
                current_pos = arc_vertices[-1] if arc_vertices else current_pos

        return np.array(vertices)

    def _render_explicit_arc(self, step: Dict, context: Dict) -> List[List[float]]:
        """Render an explicit arc with center, radius, angles."""
        center = self.resolver.resolve(step['center'], context)
        radius = self.resolver.resolve(step['radius'], context)
        start_angle = self.resolver.resolve(step['start_angle'], context)
        end_angle = self.resolver.resolve(step['end_angle'], context)

        # Convert degrees to radians
        start_rad = np.deg2rad(start_angle)
        end_rad = np.deg2rad(end_angle)

        # Generate arc points
        num_points = 15
        angles = np.linspace(start_rad, end_rad, num_points)

        vertices = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            vertices.append([x, y])

        return vertices

    def _render_solved_arc(self, step: Dict, context: Dict, current_pos: List[float]) -> List[List[float]]:
        """Render a solved arc (door smoothing, etc.)."""
        solve_method = step.get('solve_method')

        if solve_method == 'tangent_horizontal_line_and_sinusoid':
            # Door smoothing arc using alt_9.py approach
            anchor_point = self.resolver.resolve(step['anchor_point'], context)
            sin_name = step['tangent_to_sinusoid']
            min_radius = self.resolver.resolve(step['min_radius'], context)
            direction = step.get('direction', 'ccw')

            # Get sinusoid config
            sin_config = context[f'sinusoid_{sin_name}_config']

            # Get pantry dimensions
            pantry_width = self.resolver.resolve({'ref': 'base_dimensions.pantry_width'}, context)
            shelf_length = self.resolver.resolve({'ref': 'base_dimensions.pantry_depth'}, context)

            # Map wall to side
            side = 'E' if sin_config['wall'] == 'E' else 'W'

            # Solve for arc
            result = solve_door_smoothing_alt9(
                anchor_x=anchor_point[0],
                anchor_y=anchor_point[1],
                radius=min_radius,
                depth=sin_config['offset_from_wall'],
                amplitude=sin_config['amplitude'],
                period=sin_config['period'],
                offset=sin_config['phase_offset'],
                side=side,
                shelf_length=shelf_length,
                pantry_width=pantry_width
            )

            if result is None:
                print(f"  Warning: Door smoothing solver failed")
                return []

            # Store result in context for later reference
            context['last_door_smoothing_tangent_y'] = result['tangent_y']
            context['last_door_smoothing_tangent_x'] = result['tangent_x']

            # Generate arc points from anchor to tangency point
            center_x = result['center_x']
            center_y = result['center_y']
            radius = result['radius']

            # Calculate angles
            angle_anchor = np.arctan2(anchor_point[1] - center_y, anchor_point[0] - center_x)
            angle_tangent = np.arctan2(result['tangent_y'] - center_y, result['tangent_x'] - center_x)

            # Determine arc direction
            angle_diff = angle_tangent - angle_anchor
            # Normalize to [-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Generate arc points
            num_points = 20
            angles = np.linspace(angle_anchor, angle_anchor + angle_diff, num_points)

            vertices = []
            for angle in angles:
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                vertices.append([x, y])

            return vertices

        else:
            print(f"  Warning: Skipping solved arc with method {solve_method}")
            return []

    def _render_sinusoid_segment(self, step: Dict, context: Dict) -> List[List[float]]:
        """Render a segment of a sinusoid curve."""
        sin_name = step['sinusoid']
        from_param = self.resolver.resolve(step['from'], context)
        to_param = self.resolver.resolve(step['to'], context)

        sin_config = context[f'sinusoid_{sin_name}_config']

        # Generate points along sinusoid
        num_points = 50
        params = np.linspace(from_param, to_param, num_points)

        vertices = []
        if sin_config['type'] == 'vertical':
            # x(y) - different formulas for left (E) and right (W) walls
            wall = sin_config.get('wall', 'E')
            if wall == 'E':  # Left wall
                for y in params:
                    x = sin_config['x_base'] + sin_config['offset_from_wall'] + sin_config['amplitude'] * np.sin(
                        2 * np.pi * y / sin_config['period'] + sin_config['phase_offset']
                    )
                    vertices.append([x, y])
            else:  # Right wall (W)
                for y in params:
                    x = sin_config['x_base'] - sin_config['offset_from_wall'] - sin_config['amplitude'] * np.sin(
                        2 * np.pi * y / sin_config['period'] + sin_config['phase_offset']
                    )
                    vertices.append([x, y])
        else:  # horizontal
            # y(x)
            for x in params:
                y = sin_config['y_base'] - sin_config['offset_from_wall'] - sin_config['amplitude'] * np.sin(
                    2 * np.pi * x / sin_config['period'] + sin_config['phase_offset']
                )
                vertices.append([x, y])

        return vertices

    def _render_arc_segment(self, step: Dict, context: Dict) -> List[List[float]]:
        """Render a segment of a pre-computed arc."""
        arc_ref = step['arc']['ref'] if isinstance(step['arc'], dict) else step['arc']
        arc_name = arc_ref.split('.')[-1]
        arc = context[f'arc_{arc_name}']

        center_x = arc['center_x']
        center_y = arc['center_y']
        radius = arc['radius']

        # Determine start and end points
        if 'from_tangent' in step:
            # Start from tangent point - try multiple key formats
            sin_name = step['from_tangent']

            # Special case for "horizontal" tangent
            if sin_name == 'horizontal':
                start_x = arc.get('tangent_horiz_x')
                start_y = arc.get('tangent_horiz_y')
            # Special case for "anchor" point (door moulding arcs)
            elif sin_name == 'anchor':
                start_x = arc.get('anchor_x')
                start_y = arc.get('anchor_y')
            else:
                start_x = (arc.get(f'tangent_{sin_name}_x') or
                          arc.get(f'tangent_sin_x') or  # Fallback for horizontal/sinusoid arcs
                          arc.get('tangent_x') or  # For alt9 door smoothing
                          arc.get('tangent_1_x'))
                start_y = (arc.get(f'tangent_{sin_name}_y') or
                          arc.get(f'tangent_sin_y') or
                          arc.get('tangent_y') or  # For alt9 door smoothing
                          arc.get('tangent_1_y'))
        elif 'from' in step:
            from_ref = self.resolver.resolve(step['from'], context)
            start_x = from_ref['x'] if isinstance(from_ref, dict) else from_ref[0]
            start_y = from_ref['y'] if isinstance(from_ref, dict) else from_ref[1]
        else:
            start_x = arc.get('tangent_horiz_x') or arc.get('tangent_1_x')
            start_y = arc.get('tangent_horiz_y') or arc.get('tangent_1_y')

        if 'to_tangent' in step:
            # End at tangent point - try multiple key formats
            sin_name = step['to_tangent']

            # Special case for "horizontal" tangent
            if sin_name == 'horizontal':
                end_x = arc.get('tangent_horiz_x')
                end_y = arc.get('tangent_horiz_y')
            # Special case for "anchor" point (door moulding arcs)
            elif sin_name == 'anchor':
                end_x = arc.get('anchor_x')
                end_y = arc.get('anchor_y')
            else:
                # For sinusoid tangent, use tangent_sin_x/y or the named tangent
                end_x = (arc.get(f'tangent_{sin_name}_x') or
                        arc.get(f'tangent_sin_x') or
                        arc.get('tangent_2_x'))
                end_y = (arc.get(f'tangent_{sin_name}_y') or
                        arc.get(f'tangent_sin_y') or
                        arc.get('tangent_2_y'))
        elif 'to' in step:
            to_ref = self.resolver.resolve(step['to'], context)
            if isinstance(to_ref, dict):
                end_x = to_ref['x']
                end_y = to_ref['y']
            elif isinstance(to_ref, str):
                # Reference to arc midpoint
                midpoint = context[to_ref]
                end_x = midpoint['x'] if isinstance(midpoint, dict) else midpoint[0]
                end_y = midpoint['y'] if isinstance(midpoint, dict) else midpoint[1]
            else:
                end_x = to_ref[0]
                end_y = to_ref[1]
        else:
            end_x = arc.get('tangent_sin_x') or arc.get('tangent_2_x')
            end_y = arc.get('tangent_sin_y') or arc.get('tangent_2_y')

        # Debug output for arc rendering
        if start_x is None or start_y is None or end_x is None or end_y is None:
            print(f"      WARNING: Arc segment has None coordinates!")
            print(f"        start: ({start_x}, {start_y}), end: ({end_x}, {end_y})")
            print(f"        Arc keys: {list(arc.keys())}")
            return []

        # Calculate angles
        start_angle = np.arctan2(start_y - center_y, start_x - center_x)
        end_angle = np.arctan2(end_y - center_y, end_x - center_x)

        # Calculate angular difference
        angle_diff = end_angle - start_angle

        # Normalize to [-π, π] to get the shorter arc
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Note: We keep angle_diff in [-π, π] to always take the shorter arc path
        # This is correct for interior corner arcs and produces counter-clockwise
        # polygon traversal when the construction sequence is defined correctly

        # Debug: Check if angles are the same
        if abs(angle_diff) < 1e-6:
            print(f"      WARNING: Arc has identical start/end angles!")
            print(f"        Center: ({center_x:.3f}, {center_y:.3f}), Radius: {radius:.3f}")
            print(f"        Start: ({start_x:.3f}, {start_y:.3f}), angle: {np.degrees(start_angle):.1f}°")
            print(f"        End: ({end_x:.3f}, {end_y:.3f}), angle: {np.degrees(end_angle):.1f}°")

        # Generate arc points going counter-clockwise
        num_points = 20
        angles = np.linspace(start_angle, start_angle + angle_diff, num_points)

        vertices = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            vertices.append([x, y])

        return vertices


# ============================================================================
# PART 4: EXPORTER (SVG, DXF, PDF)
# ============================================================================

class GeometryExporter:
    """Exports shelf geometry to various formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.level_data = []  # Store data for layout PDF

    def export_svg(self, name: str, polygon: np.ndarray, width: float, height: float):
        """Export polygon to SVG file."""
        filepath = self.output_dir / f"{name}.svg"

        # Create SVG path
        path_data = f"M {polygon[0][0]:.6f},{polygon[0][1]:.6f}"
        for i in range(1, len(polygon)):
            path_data += f" L {polygon[i][0]:.6f},{polygon[i][1]:.6f}"
        path_data += " Z"

        # Write SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}" height="{height:.2f}" viewBox="0 0 {width:.2f} {height:.2f}">
  <path d="{path_data}" fill="none" stroke="black" stroke-width="0.1"/>
</svg>'''

        with open(filepath, 'w') as f:
            f.write(svg_content)

        print(f"  Exported SVG: {filepath}")

    def export_dxf(self, name: str, polygon: np.ndarray):
        """Export polygon to DXF file."""
        filepath = self.output_dir / f"{name}.dxf"

        # Create DXF document
        doc = ezdxf.new('R2010')
        doc.units = units.IN  # Inches
        msp = doc.modelspace()

        # Add polyline
        points = [(p[0], p[1], 0) for p in polygon]
        msp.add_lwpolyline(points, close=True)

        # Save
        doc.saveas(filepath)
        print(f"  Exported DXF: {filepath}")

    def add_level_data(self, height: float, shelf_name: str, wall: str, polygon: np.ndarray):
        """Store shelf data for layout PDF generation."""
        self.level_data.append({
            'height': height,
            'name': shelf_name,
            'wall': wall,
            'polygon': polygon.copy()
        })

    def export_layout_pdf(self, pantry_width: float, pantry_depth: float, pantry_height: float):
        """
        Export a layout PDF showing shelves in their correct positions within the pantry.
        One page per level with color-coded shelves.
        """
        filepath = self.output_dir / "pantry_layout.pdf"

        # Group shelves by level
        levels = {}
        for data in self.level_data:
            h = data['height']
            if h not in levels:
                levels[h] = []
            levels[h].append(data)

        # Sort levels by height
        sorted_heights = sorted(levels.keys())

        # Color mapping for walls
        colors = {
            'E': '#4A90E2',  # Blue for left/east
            'W': '#E24A4A',  # Red for right/west
            'S': '#50C878',  # Green for back/south
            'left': '#4A90E2',
            'right': '#E24A4A',
            'back': '#50C878'
        }

        # Create PDF
        with PdfPages(filepath) as pdf:
            for height in sorted_heights:
                fig, ax = plt.subplots(figsize=(11, 8.5))

                # Draw pantry outline
                pantry_outline = mpatches.Rectangle(
                    (0, 0), pantry_width, pantry_depth,
                    fill=False, edgecolor='black', linewidth=2
                )
                ax.add_patch(pantry_outline)

                # Draw door opening (north wall)
                door_text_y = -2
                ax.text(pantry_width/2, door_text_y, 'DOOR',
                       ha='center', va='top', fontsize=12, fontweight='bold')
                ax.plot([0, pantry_width], [0, 0], 'k--', linewidth=1, alpha=0.5)

                # Add wall labels
                ax.text(-2, pantry_depth/2, 'EAST\n(Left)',
                       ha='right', va='center', fontsize=10, rotation=90)
                ax.text(pantry_width + 2, pantry_depth/2, 'WEST\n(Right)',
                       ha='left', va='center', fontsize=10, rotation=90)
                ax.text(pantry_width/2, pantry_depth + 2, 'SOUTH\n(Back)',
                       ha='center', va='bottom', fontsize=10)

                # Draw each shelf at this level
                for shelf_data in levels[height]:
                    polygon = shelf_data['polygon']
                    wall = shelf_data['wall']
                    name = shelf_data['name']

                    # Determine color based on wall
                    color = colors.get(wall, '#CCCCCC')

                    # Create polygon patch
                    poly_patch = mpatches.Polygon(
                        polygon, closed=True,
                        facecolor=color, edgecolor='black',
                        linewidth=1.5, alpha=0.7
                    )
                    ax.add_patch(poly_patch)

                    # Add shelf label at centroid
                    centroid_x = np.mean(polygon[:, 0])
                    centroid_y = np.mean(polygon[:, 1])
                    ax.text(centroid_x, centroid_y, name,
                           ha='center', va='center',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.8))

                # Set title
                ax.set_title(f'Level: {height}" from floor\n({len(levels[height])} shelf pieces)',
                           fontsize=14, fontweight='bold', pad=20)

                # Set axis properties
                ax.set_xlim(-5, pantry_width + 5)
                ax.set_ylim(-5, pantry_depth + 5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel('Width (inches)', fontsize=10)
                ax.set_ylabel('Depth (inches)', fontsize=10)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=colors['E'], edgecolor='black', label='Left (East)'),
                    Patch(facecolor=colors['W'], edgecolor='black', label='Right (West)'),
                    Patch(facecolor=colors['S'], edgecolor='black', label='Back (South)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

                # Add dimension annotations
                ax.annotate('', xy=(pantry_width, -3), xytext=(0, -3),
                           arrowprops=dict(arrowstyle='<->', lw=1.5))
                ax.text(pantry_width/2, -3.5, f'{pantry_width}"',
                       ha='center', va='top', fontsize=9)

                ax.annotate('', xy=(pantry_width + 3, pantry_depth), xytext=(pantry_width + 3, 0),
                           arrowprops=dict(arrowstyle='<->', lw=1.5))
                ax.text(pantry_width + 3.5, pantry_depth/2, f'{pantry_depth}"',
                       ha='left', va='center', fontsize=9, rotation=90)

                plt.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        print(f"\n  Exported layout PDF: {filepath}")
        print(f"    {len(sorted_heights)} pages (one per level)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline: parse, solve, render, export."""

    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'shelf_level_patterns.json'
    print(f"Loading configuration from {config_path}...")

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loaded {len(config['levels'])} levels")

    # Create components
    resolver = ReferenceResolver(config)
    base_solver = BaseGeometrySolver(resolver)
    renderer = ConstructionRenderer(resolver)
    exporter = GeometryExporter(Path(__file__).parent.parent / 'output')

    # Process each level
    base_level_configs = {}  # Store base configs by type
    for level_idx, level in enumerate(config['levels']):
        height = level['height']
        level_type = level['type']
        print(f"\n{'='*60}")
        print(f"Level {level_idx + 1}/{len(config['levels'])}: {height}\" ({level_type})")
        print(f"{'='*60}")

        # Store first level of each type as base for templates
        if level_type not in base_level_configs:
            # Check if this level has full base_geometry (not just phase offsets)
            base_geo = level.get('base_geometry', {})
            has_full_config = (
                'arcs' in base_geo or
                'intersections' in base_geo or
                any(isinstance(v, dict) and 'type' in v for v in base_geo.get('sinusoids', {}).values())
            )
            if has_full_config:
                base_level_configs[level_type] = level

        # Get base level config for this type
        base_level_config = base_level_configs.get(level_type)

        # Solve base geometry
        print("  Solving base geometry...")
        context = base_solver.solve_level(level, base_level_config)
        print(f"    Computed {len(context)} geometry elements")

        # Render each shelf
        shelves = level.get('shelves', {})
        for shelf_key, shelf_config in shelves.items():
            shelf_name = shelf_config.get('name', f'shelf_{shelf_key}_{height}')

            # Determine wall: use 'wall' field or infer from shelf_key
            if 'wall' in shelf_config:
                shelf_wall = shelf_config['wall']
            else:
                # Map shelf_key to wall (right->W, left->E, back->S)
                wall_map = {'right': 'W', 'left': 'E', 'back': 'S'}
                shelf_wall = wall_map.get(shelf_key, shelf_key)

            print(f"\n  Rendering shelf: {shelf_name}")

            # Get construction sequence
            if 'construction_template' in shelf_config:
                # Use template from another level
                print(f"    Using construction template")
                template_ref = shelf_config['construction_template']['ref']
                # Parse reference like "levels[0].shelves.right"
                parts = template_ref.split('.')
                template_shelves = config
                for part in parts:
                    if '[' in part:
                        # Handle array index like "levels[0]"
                        key = part.split('[')[0]
                        idx = int(part.split('[')[1].split(']')[0])
                        template_shelves = template_shelves[key][idx]
                    else:
                        template_shelves = template_shelves[part]

                construction = template_shelves.get('construction', [])
            else:
                construction = shelf_config.get('construction', [])
            if not construction:
                print(f"    No construction sequence found, skipping")
                continue

            try:
                # Render polygon
                polygon = renderer.render_shelf(construction, context)
                print(f"    Generated polygon with {len(polygon)} vertices")

                # Export
                # Determine dimensions for SVG viewbox
                if shelf_wall in ['left', 'E']:
                    width = 15.0  # Approximate
                    height_box = 50.0
                elif shelf_wall in ['right', 'W']:
                    width = 15.0
                    height_box = 50.0
                else:  # back
                    width = 50.0
                    height_box = 30.0

                exporter.export_svg(shelf_name, polygon, width, height_box)
                exporter.export_dxf(shelf_name, polygon)

                # Store for layout PDF
                exporter.add_level_data(height, shelf_name, shelf_wall, polygon)

            except Exception as e:
                print(f"    ERROR rendering shelf: {e}")
                import traceback
                traceback.print_exc()

    # Generate layout PDF
    print(f"\n{'='*60}")
    print("Generating pantry layout PDF...")
    print(f"{'='*60}")

    pantry_width = resolver.resolve({'ref': 'base_dimensions.pantry_width'}, {})
    pantry_depth = resolver.resolve({'ref': 'base_dimensions.pantry_depth'}, {})
    pantry_height = resolver.resolve({'ref': 'base_dimensions.pantry_height'}, {})

    exporter.export_layout_pdf(pantry_width, pantry_depth, pantry_height)

    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"{'='*60}")

    # Summary
    svg_count = len(list(exporter.output_dir.glob('shelf_*.svg')))
    dxf_count = len(list(exporter.output_dir.glob('shelf_*.dxf')))

    print(f"\nGenerated files:")
    print(f"  {svg_count} SVG cutting templates")
    print(f"  {dxf_count} DXF CAD files")
    print(f"  1 pantry layout PDF ({len(sorted(set(d['height'] for d in exporter.level_data)))} levels)")
    print(f"\nOutput directory: {exporter.output_dir}")
    print(f"\nKey files:")
    print(f"  - pantry_layout.pdf: Assembly guide showing shelves in position")
    print(f"  - shelf_*.svg: Individual cutting templates for laser/CNC")
    print(f"  - shelf_*.dxf: CAD files for manufacturing")


if __name__ == '__main__':
    main()
