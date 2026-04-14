#!/usr/bin/env python3
"""
Generate ~1000 noise map PLY files from ville.ply with random source positions.

For each source position, runs NoiseMap with each drone model (M2, I2, F-4, S-9).
Output files: Ville_colored_x_y_z_100_DRONE.ply

The script:
1. Parses ville.ply to build a 2D heightmap of buildings
2. Generates random source positions above buildings
3. Runs the NoiseMap program for each (position, drone) combination
"""

import subprocess
import sys
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
NOISEMAP_BIN = Path(__file__).parent / "build" / "NoiseMap"
PLY_INPUT    = Path(__file__).parent / "Data" / "ville.ply"
OUTPUT_DIR   = Path(__file__).parent / "Data" / "generated"
SCALE        = 100
DRONES       = ["M2", "I2", "F-4", "S-9"]
N_POSITIONS  = 250  # × 4 drones = 1000 files
MIN_CLEARANCE = 0.05  # mesh units above building tops (5m at scale 100)
# Drone altitude range in mesh units (at scale 100: 0.1 = 10m, 1.5 = 150m)
Z_MIN = 0.1
Z_MAX = 1.5
# XY margin from mesh edges
XY_MARGIN = 1.0
SEED = 42
N_WORKERS = os.cpu_count() or 4

# ─── Parse PLY to get vertex positions ──────────────────────────────────────

def parse_ply_vertices(ply_path):
    """Read vertex positions from an ASCII PLY file."""
    vertices = []
    n_vertices = 0
    with open(ply_path, 'r') as f:
        # Parse header
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            if line == 'end_header':
                break

        # Read vertices
        for _ in range(n_vertices):
            parts = f.readline().strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            vertices.append((x, y, z))

    return np.array(vertices)


# ─── Build 2D heightmap ────────────────────────────────────────────────────

def build_heightmap(vertices, resolution=0.5):
    """
    Build a 2D grid of max Z values (building height) from mesh vertices.
    Returns: heightmap (2D array), x_edges, y_edges
    """
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)

    heightmap = np.zeros((len(x_edges) - 1, len(y_edges) - 1))

    # Bin vertices into grid cells and track max Z
    x_idx = np.digitize(vertices[:, 0], x_edges) - 1
    y_idx = np.digitize(vertices[:, 1], y_edges) - 1

    # Clamp to valid range
    x_idx = np.clip(x_idx, 0, len(x_edges) - 2)
    y_idx = np.clip(y_idx, 0, len(y_edges) - 2)

    for i in range(len(vertices)):
        xi, yi = x_idx[i], y_idx[i]
        heightmap[xi, yi] = max(heightmap[xi, yi], vertices[i, 2])

    return heightmap, x_edges, y_edges


def get_building_height_at(x, y, heightmap, x_edges, y_edges):
    """Get the max building height at a given XY position."""
    xi = np.searchsorted(x_edges, x) - 1
    yi = np.searchsorted(y_edges, y) - 1
    xi = np.clip(xi, 0, heightmap.shape[0] - 1)
    yi = np.clip(yi, 0, heightmap.shape[1] - 1)
    return heightmap[xi, yi]


# ─── Generate valid source positions ───────────────────────────────────────

def generate_positions(n, heightmap, x_edges, y_edges, rng):
    """Generate n random source positions above buildings."""
    x_min = x_edges[0] + XY_MARGIN
    x_max = x_edges[-1] - XY_MARGIN
    y_min = y_edges[0] + XY_MARGIN
    y_max = y_edges[-1] - XY_MARGIN

    positions = []
    attempts = 0
    max_attempts = n * 20

    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)

        # Get building height at this XY
        bh = get_building_height_at(x, y, heightmap, x_edges, y_edges)

        # Source Z must be above buildings with clearance
        z_floor = max(Z_MIN, bh + MIN_CLEARANCE)

        if z_floor > Z_MAX:
            # This position is over a very tall building, skip
            continue

        z = rng.uniform(z_floor, Z_MAX)
        positions.append((round(x, 4), round(y, 4), round(z, 4)))

    return positions


# ─── Single job (called by each worker thread) ────────────────────────────

def run_one(job):
    """Run NoiseMap for one (position, drone) combination.
    Each worker uses its own temp symlink so outputs don't collide.
    Returns (out_name, True/False, error_msg).
    """
    idx, x, y, z, drone, total = job
    out_name = f"Ville_colored_{x}_{y}_{z}_{SCALE}_{drone}.ply"
    out_path = OUTPUT_DIR / out_name

    if out_path.exists():
        return (out_name, idx, total, "skip", "")

    # Create a unique temp symlink so NoiseMap writes to a unique output file
    tmp_dir = tempfile.mkdtemp(prefix="noisemap_")
    tmp_input = Path(tmp_dir) / "input.ply"
    tmp_input.symlink_to(PLY_INPUT.resolve())
    tmp_output = Path(tmp_dir) / "input_noisemap.ply"

    cmd = [
        str(NOISEMAP_BIN),
        str(tmp_input),
        str(x), str(y), str(z),
        "--drone", drone,
        "--scale", str(SCALE),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return (out_name, idx, total, "fail", result.stderr.strip()[-200:])

        if tmp_output.exists():
            shutil.move(str(tmp_output), str(out_path))
            return (out_name, idx, total, "ok", "")
        else:
            return (out_name, idx, total, "fail", "no output file produced")

    except subprocess.TimeoutExpired:
        return (out_name, idx, total, "timeout", "")
    except Exception as e:
        return (out_name, idx, total, "error", str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Parsing {PLY_INPUT}...")
    vertices = parse_ply_vertices(PLY_INPUT)
    print(f"  {len(vertices)} vertices loaded")
    print(f"  X: [{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}]")
    print(f"  Y: [{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}]")
    print(f"  Z: [{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")

    print("Building heightmap...")
    heightmap, x_edges, y_edges = build_heightmap(vertices)

    rng = np.random.default_rng(SEED)

    print(f"Generating {N_POSITIONS} source positions...")
    positions = generate_positions(N_POSITIONS, heightmap, x_edges, y_edges, rng)
    print(f"  {len(positions)} valid positions generated")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of all jobs
    jobs = []
    idx = 0
    for i, (x, y, z) in enumerate(positions):
        for drone in DRONES:
            idx += 1
            jobs.append((idx, x, y, z, drone, len(positions) * len(DRONES)))
    total = len(jobs)

    print(f"\nWill generate {total} files ({len(positions)} positions × {len(DRONES)} drones)")
    print(f"Using {N_WORKERS} parallel workers")
    print(f"Output directory: {OUTPUT_DIR}\n")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}

        for future in as_completed(futures):
            out_name, idx, total, status, err = future.result()

            if status == "skip":
                success += 1
            elif status == "ok":
                success += 1
                if success % 50 == 0 or (success + failed) == total:
                    print(f"  [{success + failed}/{total}] OK: {out_name}")
            elif status == "timeout":
                print(f"  [{success + failed}/{total}] TIMEOUT: {out_name}")
                failed += 1
            else:
                print(f"  [{success + failed}/{total}] FAIL: {out_name}")
                if err:
                    print(f"    {err}")
                failed += 1

    print(f"\nDone! {success} succeeded, {failed} failed out of {total}")
    print(f"Files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
