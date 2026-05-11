#!/usr/bin/env python3
"""
Estimate per-line ShopBot command throughput speed for an .sbp file.

This does not know the controller's real timing. It estimates speed as:

    estimated_ipm = distance_commanded_inches * commands_per_second * 60

That makes short segmented moves stand out when the controller is effectively
limited by how many commands it can consume per second.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_COMMANDS_PER_SECOND = 100.0


def command_token(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("'"):
        return None
    return stripped.split(",", 1)[0].split(None, 1)[0].strip().upper()


def parse_numbers_after_command(line):
    if "," not in line:
        return []
    values = []
    for raw in line.split(",", 1)[1].split(","):
        raw = raw.strip()
        if not raw:
            values.append(None)
            continue
        try:
            values.append(float(raw))
        except ValueError:
            values.append(None)
    return values


def move_position(cmd, values, last_pos):
    if cmd not in {"M2", "M3", "M5", "J2", "J3", "J5"}:
        return None

    dimensions = int(cmd[1])
    present = [v for v in values if v is not None]
    if len(present) < dimensions:
        return None

    x = present[0]
    y = present[1] if dimensions >= 2 else (last_pos[1] if last_pos else None)
    z = present[2] if dimensions >= 3 else (last_pos[2] if last_pos else None)
    return (x, y, z)


def linear_distance(a, b):
    if a is None or b is None or None in a or None in b:
        return 0.0, 0.0
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy = math.hypot(dx, dy)
    xyz = math.sqrt(dx * dx + dy * dy + dz * dz)
    return xy, xyz


def angle_sweep(start_angle, end_angle, direction):
    ccw = (end_angle - start_angle) % (2 * math.pi)
    if direction is not None and direction < 0:
        return ccw - 2 * math.pi
    return ccw


def arc_distance(values, last_pos):
    """Return (end_pos, xy_arc_len, xyz_len) for ShopBot CG commands.

    The Fusion-generated file uses:
      CG, , end_x, end_y, center_offset_x, center_offset_y, , direction
    where center offsets are relative to the current position.
    """
    if last_pos is None or None in last_pos:
        return None, 0.0, 0.0

    present = [v for v in values if v is not None]
    if len(present) < 4:
        return None, 0.0, 0.0

    end_x, end_y = present[0], present[1]
    i_off, j_off = present[2], present[3]
    direction = present[4] if len(present) >= 5 else None

    start_x, start_y, start_z = last_pos
    center_x = start_x + i_off
    center_y = start_y + j_off

    r_start = math.hypot(start_x - center_x, start_y - center_y)
    r_end = math.hypot(end_x - center_x, end_y - center_y)
    radius = (r_start + r_end) / 2.0
    if radius <= 1e-12:
        end_pos = (end_x, end_y, start_z)
        xy, xyz = linear_distance(last_pos, end_pos)
        return end_pos, xy, xyz

    a0 = math.atan2(start_y - center_y, start_x - center_x)
    a1 = math.atan2(end_y - center_y, end_x - center_x)
    sweep = angle_sweep(a0, a1, direction)
    xy_len = abs(sweep) * radius

    end_pos = (end_x, end_y, start_z)
    return end_pos, xy_len, xy_len


def speed_bucket(ipm):
    if ipm >= 100.0:
        return "planned", "black"
    if ipm >= 10.0:
        return "acceptable_slow", "0.55"
    if ipm >= 1.0:
        return "danger_slow", "orange"
    if ipm > 0.0:
        return "risk_crawl", "red"
    return "non_motion", "0.85"


def analyze(path, commands_per_second):
    rows = []
    last_pos = None
    motion_index = 0
    elapsed_seconds = 0.0

    lines = path.read_text(errors="replace").splitlines()
    for line_no, raw in enumerate(lines, 1):
        cmd = command_token(raw)
        distance_xy = 0.0
        distance_xyz = 0.0
        ipm = 0.0
        is_motion = False

        if cmd in {"M2", "M3", "M5", "J2", "J3", "J5"}:
            pos = move_position(cmd, parse_numbers_after_command(raw), last_pos)
            if pos is not None:
                distance_xy, distance_xyz = linear_distance(last_pos, pos)
                last_pos = pos
                is_motion = True
        elif cmd == "CG":
            pos, distance_xy, distance_xyz = arc_distance(
                parse_numbers_after_command(raw), last_pos
            )
            if pos is not None:
                last_pos = pos
                is_motion = True

        if is_motion:
            motion_index += 1
            elapsed_seconds = motion_index / commands_per_second
            ipm = distance_xyz * commands_per_second * 60.0

        bucket, color = speed_bucket(ipm)
        rows.append(
            {
                "line": line_no,
                "time_seconds": elapsed_seconds,
                "command": cmd or "",
                "distance_xy_in": distance_xy,
                "distance_xyz_in": distance_xyz,
                "estimated_ipm": ipm,
                "bucket": bucket,
                "raw": raw,
                "color": color,
            }
        )

    return rows


def write_csv(rows, path):
    fields = [
        "line",
        "time_seconds",
        "command",
        "distance_xy_in",
        "distance_xyz_in",
        "estimated_ipm",
        "bucket",
        "raw",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def plot_rows(rows, path):
    motion = [r for r in rows if r["estimated_ipm"] > 0.0]
    if not motion:
        raise RuntimeError("No motion rows found to plot.")

    fig, ax = plt.subplots(figsize=(9, 14))

    for bucket, label, color in [
        ("planned", ">= 100 ipm", "black"),
        ("acceptable_slow", "10-100 ipm", "0.55"),
        ("danger_slow", "1-10 ipm", "orange"),
        ("risk_crawl", "< 1 ipm", "red"),
    ]:
        xs = [r["estimated_ipm"] for r in motion if r["bucket"] == bucket]
        ys = [r["time_seconds"] for r in motion if r["bucket"] == bucket]
        if xs:
            ax.scatter(xs, ys, s=5, c=color, label=label, linewidths=0)

    ax.axvline(150, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(40, color="0.55", linewidth=0.8, alpha=0.5)
    ax.axvline(10, color="orange", linewidth=1.1, alpha=0.8)
    ax.axvline(1, color="red", linewidth=1.1, alpha=0.8)

    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("Estimated speed (ipm)")
    ax.set_ylabel("Estimated time through file (seconds)")
    ax.set_title("Estimated ShopBot command-throughput speed")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-line speed from ShopBot .sbp command distances."
    )
    parser.add_argument("sbp", type=Path, help="Input .sbp file")
    parser.add_argument(
        "--commands-per-second",
        type=float,
        default=DEFAULT_COMMANDS_PER_SECOND,
        help=f"Assumed controller command throughput. Default: {DEFAULT_COMMANDS_PER_SECOND}",
    )
    parser.add_argument("--csv", type=Path, help="Output CSV path")
    parser.add_argument("--plot", type=Path, help="Output PNG plot path")
    args = parser.parse_args()

    rows = analyze(args.sbp, args.commands_per_second)

    csv_path = args.csv or args.sbp.with_suffix(".speed_estimate.csv")
    plot_path = args.plot or args.sbp.with_suffix(".speed_estimate.png")

    write_csv(rows, csv_path)
    plot_rows(rows, plot_path)

    motion = [r for r in rows if r["estimated_ipm"] > 0.0]
    print(f"Input: {args.sbp}")
    print(f"Commands/second assumption: {args.commands_per_second:g}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {plot_path}")
    print(f"Motion rows: {len(motion)}")
    for cutoff in (150, 100, 40, 10, 1):
        count = sum(1 for r in motion if r["estimated_ipm"] < cutoff)
        print(f"  below {cutoff:>3} ipm: {count}")


if __name__ == "__main__":
    main()
