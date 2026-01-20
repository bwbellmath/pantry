#!/usr/bin/env python3
"""
Test script for 2D Gaussian Process random curves between anchor points.

This creates and visualizes GP curves conditioned on:
- Two anchor points (hard observations with zero variance)
- Optional derivative constraints at endpoints
- Smoothness via covariance kernel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path


class GaussianProcess1D:
    """
    Simple 1D Gaussian Process for curves x(y).

    Given y coordinates, predicts x coordinates using GP kriging.
    """

    def __init__(self, length_scale=5.0, variance=1.0, nugget=1e-6):
        """
        Initialize GP with covariance parameters.

        Args:
            length_scale: Correlation length for squared exponential kernel
            variance: Overall variance/amplitude of the process
            nugget: Small value added to diagonal for numerical stability
        """
        self.length_scale = length_scale
        self.variance = variance
        self.nugget = nugget

    def kernel(self, y1, y2):
        """
        Squared exponential (RBF/Gaussian) covariance kernel.

        k(y1, y2) = σ² * exp(-|y1 - y2|² / (2 * ℓ²))

        Args:
            y1: Array of y values, shape (n1,) or (n1, 1)
            y2: Array of y values, shape (n2,) or (n2, 1)

        Returns:
            Covariance matrix K, shape (n1, n2)
        """
        y1 = np.atleast_2d(y1).reshape(-1, 1)
        y2 = np.atleast_2d(y2).reshape(-1, 1)

        # Squared distances
        dist_sq = cdist(y1, y2, metric='sqeuclidean')

        # Squared exponential kernel
        K = self.variance * np.exp(-dist_sq / (2 * self.length_scale**2))

        return K

    def kernel_derivative(self, y1, y2):
        """
        Derivative of kernel with respect to y2.

        ∂k(y1, y2)/∂y2 = k(y1, y2) * (y1 - y2) / ℓ²

        This is used for incorporating derivative observations.
        """
        y1 = np.atleast_2d(y1).reshape(-1, 1)
        y2 = np.atleast_2d(y2).reshape(-1, 1)

        K = self.kernel(y1, y2)
        y_diff = y1 - y2.T  # Broadcasting to get (n1, n2)

        dK_dy = K * y_diff / (self.length_scale**2)

        return dK_dy

    def condition_on_observations(self, y_obs, x_obs, dx_dy_obs=None):
        """
        Condition GP on observed points and optional derivatives.

        Args:
            y_obs: Y coordinates of observations, shape (n_obs,)
            x_obs: X coordinates of observations, shape (n_obs,)
            dx_dy_obs: Optional derivatives dx/dy at observation points, shape (n_deriv,)
                       If provided, derivatives are for the first n_deriv observation points
        """
        self.y_obs = np.atleast_1d(y_obs)
        self.x_obs = np.atleast_1d(x_obs)

        n_obs = len(self.y_obs)

        # Build covariance matrix for observations
        K_obs = self.kernel(self.y_obs, self.y_obs)

        if dx_dy_obs is not None:
            self.dx_dy_obs = np.atleast_1d(dx_dy_obs)
            n_deriv = len(self.dx_dy_obs)

            # For derivative observations at the first n_deriv points
            y_deriv = self.y_obs[:n_deriv]

            # Cross-covariance between function values and derivatives
            # Cov(f(y), df/dy'(y')) = ∂k/∂y'(y, y')
            K_obs_deriv = self.kernel_derivative(self.y_obs, y_deriv)

            # Covariance between derivatives
            # Cov(df/dy(y), df/dy'(y')) = -∂²k/∂y∂y'(y, y')
            K_deriv = self.kernel(y_deriv, y_deriv)
            K_deriv_deriv = -K_deriv / (self.length_scale**2)

            # Build augmented covariance matrix
            # [K_obs,         K_obs_deriv    ]
            # [K_obs_deriv^T, K_deriv_deriv  ]
            K_full = np.block([
                [K_obs, K_obs_deriv],
                [K_obs_deriv.T, K_deriv_deriv]
            ])

            # Add nugget for numerical stability
            K_full += self.nugget * np.eye(n_obs + n_deriv)

            # Combined observation vector [x_obs, dx/dy_obs]
            y_full = np.concatenate([self.x_obs, self.dx_dy_obs])

            # Precompute for predictions
            self.K_inv = np.linalg.inv(K_full)
            self.alpha = self.K_inv @ y_full

            self.has_derivatives = True
            self.n_deriv = n_deriv
            self.y_deriv = y_deriv

        else:
            # Standard GP without derivatives
            K_obs += self.nugget * np.eye(n_obs)

            self.K_inv = np.linalg.inv(K_obs)
            self.alpha = self.K_inv @ self.x_obs

            self.has_derivatives = False

    def predict(self, y_test, return_std=False):
        """
        Predict x values at test y coordinates.

        Args:
            y_test: Y coordinates for prediction, shape (n_test,)
            return_std: If True, also return standard deviation

        Returns:
            x_mean: Predicted x values, shape (n_test,)
            x_std: (optional) Standard deviation, shape (n_test,)
        """
        y_test = np.atleast_1d(y_test)
        n_test = len(y_test)

        # Cross-covariance between test and observation points
        K_test_obs = self.kernel(y_test, self.y_obs)

        if self.has_derivatives:
            # Also include cross-covariance with derivative observations
            K_test_deriv = self.kernel_derivative(y_test, self.y_deriv)

            # Stack horizontally: [K_test_obs, K_test_deriv]
            K_test_all = np.hstack([K_test_obs, K_test_deriv])

            # Predictive mean
            x_mean = K_test_all @ self.alpha

        else:
            # No derivatives
            x_mean = K_test_obs @ self.alpha
            K_test_all = K_test_obs

        if return_std:
            # Predictive variance
            K_test = self.kernel(y_test, y_test)
            K_pred = K_test - K_test_all @ self.K_inv @ K_test_all.T

            # Extract diagonal (variance at each point)
            x_var = np.diag(K_pred)
            x_std = np.sqrt(np.maximum(x_var, 0))  # Avoid negative due to numerics

            return x_mean, x_std

        return x_mean


def test_gp_curve_basic():
    """Test basic GP curve between two anchor points."""
    print("=" * 60)
    print("Test 1: Basic GP curve (no derivative constraints)")
    print("=" * 60)

    # Two anchor points
    y_obs = np.array([0, 10])
    x_obs = np.array([0, 8])

    # Create GP with smaller length_scale for more variation
    gp = GaussianProcess1D(length_scale=1.5, variance=1.0)
    gp.condition_on_observations(y_obs, x_obs)

    # Predict on dense grid
    y_test = np.linspace(0, 10, 100)
    x_mean, x_std = gp.predict(y_test, return_std=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Curve with uncertainty
    ax1.plot(x_mean, y_test, 'b-', linewidth=2, label='GP mean')
    ax1.fill_betweenx(y_test, x_mean - 2*x_std, x_mean + 2*x_std,
                      alpha=0.3, color='blue', label='±2σ uncertainty')
    ax1.plot(x_obs, y_obs, 'ro', markersize=10, label='Anchor points', zorder=5)

    # Straight line for reference
    x_straight = np.linspace(x_obs[0], x_obs[1], 100)
    y_straight = np.linspace(y_obs[0], y_obs[1], 100)
    ax1.plot(x_straight, y_straight, 'k--', alpha=0.5, label='Straight line')

    ax1.set_xlabel('x (depth)', fontsize=12)
    ax1.set_ylabel('y (length)', fontsize=12)
    ax1.set_title('GP Curve (length_scale=1.5, variance=1.0)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Standard deviation
    ax2.plot(x_std, y_test, 'r-', linewidth=2)
    ax2.set_xlabel('Standard deviation', fontsize=12)
    ax2.set_ylabel('y (length)', fontsize=12)
    ax2.set_title('Uncertainty along curve', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/test_gp_basic.png', dpi=150)
    print(f"  Saved: output/test_gp_basic.png")
    print(f"  Max std dev: {x_std.max():.3f}")
    print()


def test_gp_curve_with_derivatives():
    """Test GP curve with derivative constraints at endpoints."""
    print("=" * 60)
    print("Test 2: GP curve with derivative constraints")
    print("=" * 60)

    # Two anchor points
    y_obs = np.array([0, 10])
    x_obs = np.array([0, 8])

    # Derivative constraints: horizontal at both ends (dx/dy = 0)
    dx_dy_obs = np.array([0.0, 0.0])

    # Create GP with smaller length_scale for more variation
    gp = GaussianProcess1D(length_scale=1.5, variance=1.0)
    gp.condition_on_observations(y_obs, x_obs, dx_dy_obs)

    # Predict
    y_test = np.linspace(0, 10, 100)
    x_mean, x_std = gp.predict(y_test, return_std=True)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x_mean, y_test, 'b-', linewidth=2, label='GP mean (dx/dy=0 at ends)')
    ax1.fill_betweenx(y_test, x_mean - 2*x_std, x_mean + 2*x_std,
                      alpha=0.3, color='blue', label='±2σ uncertainty')
    ax1.plot(x_obs, y_obs, 'ro', markersize=10, label='Anchor points', zorder=5)

    # Add arrows to show derivative constraints
    arrow_len = 0.5
    ax1.arrow(x_obs[0], y_obs[0], arrow_len, 0, head_width=0.3,
              head_length=0.2, fc='green', ec='green', linewidth=2)
    ax1.arrow(x_obs[1], y_obs[1], arrow_len, 0, head_width=0.3,
              head_length=0.2, fc='green', ec='green', linewidth=2)

    ax1.set_xlabel('x (depth)', fontsize=12)
    ax1.set_ylabel('y (length)', fontsize=12)
    ax1.set_title('GP Curve with Derivative Constraints', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2.plot(x_std, y_test, 'r-', linewidth=2)
    ax2.set_xlabel('Standard deviation', fontsize=12)
    ax2.set_ylabel('y (length)', fontsize=12)
    ax2.set_title('Uncertainty along curve', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/test_gp_derivatives.png', dpi=150)
    print(f"  Saved: output/test_gp_derivatives.png")
    print(f"  Max std dev: {x_std.max():.3f}")
    print()


def test_gp_curve_varied_slopes():
    """Test GP curves with different slope constraints."""
    print("=" * 60)
    print("Test 3: GP curves with varied slope constraints")
    print("=" * 60)

    # Two anchor points
    y_obs = np.array([0, 10])
    x_obs = np.array([0, 8])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    slope_configs = [
        (0.0, 0.0, "Horizontal at both ends"),
        (1.0, 1.0, "Slope = 1.0 at both ends"),
        (2.0, -2.0, "Slope = 2.0 at start, -2.0 at end"),
        (0.0, -1.0, "Horizontal at start, slope = -1.0 at end")
    ]

    for idx, (slope_start, slope_end, title) in enumerate(slope_configs):
        dx_dy_obs = np.array([slope_start, slope_end])

        gp = GaussianProcess1D(length_scale=1.5, variance=1.0)
        gp.condition_on_observations(y_obs, x_obs, dx_dy_obs)

        y_test = np.linspace(0, 10, 100)
        x_mean, x_std = gp.predict(y_test, return_std=True)

        ax = axes[idx]
        ax.plot(x_mean, y_test, 'b-', linewidth=2)
        ax.fill_betweenx(y_test, x_mean - 2*x_std, x_mean + 2*x_std,
                         alpha=0.3, color='blue')
        ax.plot(x_obs, y_obs, 'ro', markersize=10, zorder=5)

        # Draw slope indicators
        arrow_scale = 0.3
        # At start
        dx_start = arrow_scale
        dy_start = arrow_scale * slope_start
        ax.arrow(x_obs[0], y_obs[0], dx_start, dy_start, head_width=0.3,
                 head_length=0.2, fc='green', ec='green', linewidth=2)

        # At end
        dx_end = arrow_scale
        dy_end = arrow_scale * slope_end
        ax.arrow(x_obs[1], y_obs[1], dx_end, dy_end, head_width=0.3,
                 head_length=0.2, fc='green', ec='green', linewidth=2)

        ax.set_xlabel('x (depth)', fontsize=10)
        ax.set_ylabel('y (length)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-2, 12)
        ax.set_ylim(-1, 11)

    plt.tight_layout()
    plt.savefig('output/test_gp_varied_slopes.png', dpi=150)
    print(f"  Saved: output/test_gp_varied_slopes.png")
    print()


def test_gp_multiple_samples():
    """Generate multiple random samples from the same GP."""
    print("=" * 60)
    print("Test 4: Multiple random samples from GP")
    print("=" * 60)

    # Two anchor points
    y_obs = np.array([0, 10])
    x_obs = np.array([0, 8])

    # Create GP with smaller length_scale for more variation
    gp = GaussianProcess1D(length_scale=2.0, variance=2.0)
    gp.condition_on_observations(y_obs, x_obs)

    # Predict
    y_test = np.linspace(0, 10, 100)
    x_mean, x_std = gp.predict(y_test, return_std=True)

    # Generate random samples
    n_samples = 10

    # Get full covariance matrix
    K_test_obs = gp.kernel(y_test, gp.y_obs)
    K_test = gp.kernel(y_test, y_test)
    K_pred = K_test - K_test_obs @ gp.K_inv @ K_test_obs.T

    # Add small nugget for numerical stability
    K_pred += 1e-6 * np.eye(len(y_test))

    # Sample from multivariate normal
    x_samples = np.random.multivariate_normal(x_mean, K_pred, size=n_samples)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot samples
    for i, x_sample in enumerate(x_samples):
        ax.plot(x_sample, y_test, 'b-', alpha=0.4, linewidth=1)

    # Plot mean
    ax.plot(x_mean, y_test, 'r-', linewidth=3, label='GP mean', zorder=10)

    # Plot anchor points
    ax.plot(x_obs, y_obs, 'ko', markersize=12, label='Anchor points', zorder=15)

    ax.set_xlabel('x (depth)', fontsize=12)
    ax.set_ylabel('y (length)', fontsize=12)
    ax.set_title(f'{n_samples} Random Samples from GP (length_scale=2.0, variance=2.0)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('output/test_gp_samples.png', dpi=150)
    print(f"  Saved: output/test_gp_samples.png")
    print(f"  Generated {n_samples} random curve samples")
    print()


def main():
    # Create output directory
    Path('output').mkdir(exist_ok=True)

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Gaussian Process Curve Testing" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    # Run tests
    test_gp_curve_basic()
    test_gp_curve_with_derivatives()
    test_gp_curve_varied_slopes()
    test_gp_multiple_samples()

    print("=" * 60)
    print("All tests complete! Check output/ for visualizations.")
    print("=" * 60)


if __name__ == '__main__':
    main()
