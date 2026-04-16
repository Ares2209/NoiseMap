#!/usr/bin/env python3
"""
Génère un dataset en exécutant ./NoiseMap sur un ensemble de points (x,y,z)
échantillonnés dans ../Data/ville.ply, au-dessus des bâtiments, répartis entre
4 drones (M2, I2, F-4, S-9).

Exécution depuis n'importe quel CWD : le script calcule les chemins par rapport
à son emplacement.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
BUILD_DIR    = SCRIPT_DIR / "build"
DATA_DIR     = SCRIPT_DIR / "Data"
PLY_INPUTS   = [DATA_DIR / f for f in
                ["ville.ply", "ville2.ply", "ville3.ply", "ville4.ply", "ville5.ply"]]
NOISEMAP_BIN = BUILD_DIR / "NoiseMap"

RESULTS_DIR = SCRIPT_DIR / "generated"
TMP_DIR     = SCRIPT_DIR / "generated" / "_tmp"

DRONES         = ["M2", "I2", "F-4", "S-9"]
TOTAL_POINTS   = 2000              # 2000 fichiers au total
POINTS_PER_DRONE = TOTAL_POINTS // len(DRONES)  # 500 par drone
# Répartition uniforme sur les cartes : chaque drone a le même nombre de points
# par carte → 500/5 = 100 pts/drone/carte, 400 pts/carte au total.
POINTS_PER_DRONE_PER_MAP = POINTS_PER_DRONE // len(PLY_INPUTS)

# Altitude du drone au-dessus du toit (en unités maillage, 1 unité = 100 m par
# défaut). On reste bas pour des valeurs SPL intéressantes.
Z_OFFSET_MIN = 0.2
Z_OFFSET_MAX = 1.0

# Marge d'inset depuis les bords de la bbox (fraction de la taille).
EDGE_MARGIN_FRAC = 0.02

# Taille de la grille 2D pour la hauteur max des bâtiments.
GRID_RES = 256

N_WORKERS = min(os.cpu_count() or 4, 16)

RNG_SEED = 42


# ── Lecture du PLY (ASCII) ───────────────────────────────────────────────────
def read_ply_vertices(path: Path) -> np.ndarray:
    """Retourne un tableau (N,3) des coordonnées x,y,z des sommets."""
    with path.open("r") as f:
        n_vertex = 0
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY header incomplet")
            line = line.strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line == "end_header":
                break
        verts = np.empty((n_vertex, 3), dtype=np.float32)
        for i in range(n_vertex):
            parts = f.readline().split()
            verts[i, 0] = float(parts[0])
            verts[i, 1] = float(parts[1])
            verts[i, 2] = float(parts[2])
    return verts


# ── Échantillonnage de points au-dessus des bâtiments ───────────────────────
def build_height_map(verts: np.ndarray, res: int):
    """Retourne (height_grid, x_min, y_min, dx, dy) : max Z par cellule (x,y)."""
    x_min, y_min = verts[:, 0].min(), verts[:, 1].min()
    x_max, y_max = verts[:, 0].max(), verts[:, 1].max()
    dx = (x_max - x_min) / res
    dy = (y_max - y_min) / res

    ix = np.clip(((verts[:, 0] - x_min) / dx).astype(np.int32), 0, res - 1)
    iy = np.clip(((verts[:, 1] - y_min) / dy).astype(np.int32), 0, res - 1)

    height = np.full((res, res), -np.inf, dtype=np.float32)
    # np.maximum.at fait l'accumulation max sans boucle Python.
    np.maximum.at(height, (ix, iy), verts[:, 2])
    # Cellules vides → 0 (sol).
    height[~np.isfinite(height)] = 0.0
    return height, float(x_min), float(y_min), float(dx), float(dy), float(x_max), float(y_max)


def _sample_in_cell(n, x0, x1, y0, y1, height, gx_min, gy_min, gdx, gdy,
                    gres, z_max_map, rng):
    """Échantillonne n points (x,y) uniformément dans la cellule rect donnée
    et calcule z = toit_local + offset."""
    xs = rng.uniform(x0, x1, n)
    ys = rng.uniform(y0, y1, n)
    ix = np.clip(((xs - gx_min) / gdx).astype(np.int32), 0, gres - 1)
    iy = np.clip(((ys - gy_min) / gdy).astype(np.int32), 0, gres - 1)
    hs = height[ix, iy]
    offs = rng.uniform(Z_OFFSET_MIN, Z_OFFSET_MAX, n)
    zs = np.minimum(hs + offs, z_max_map - 1e-3)
    return xs, ys, zs, hs


def sample_points_stratified(n_per_drone: int, n_drones: int,
                             verts: np.ndarray,
                             rng: np.random.Generator):
    """Retourne (pts (N,3), drone_idx (N,)) avec N = n_per_drone*n_drones.

    Stratification : la bbox est découpée en grille S×S ; chaque drone reçoit
    le même nombre de points dans chaque cellule.  Cela garantit que chaque
    drone couvre uniformément toute la carte (pas de biais N/S/E/O).
    """
    height, x_min, y_min, dx, dy, x_max, y_max = build_height_map(verts, GRID_RES)
    z_max_map = float(verts[:, 2].max())
    span_x = x_max - x_min
    span_y = y_max - y_min
    inset_x = EDGE_MARGIN_FRAC * span_x
    inset_y = EDGE_MARGIN_FRAC * span_y
    x_lo, x_hi = x_min + inset_x, x_max - inset_x
    y_lo, y_hi = y_min + inset_y, y_max - inset_y

    # Grille de stratification S×S : ~n_per_drone/S² points par drone par case.
    # On prend S tel que n_per_drone soit divisible par S² → répartition exacte.
    S = 5  # 25 cellules → 250/25 = 10 pts/drone/cellule, 40 pts/cellule total.
    if n_per_drone % (S * S) != 0:
        # fallback souple : on tolère un reste qu'on distribue aléatoirement.
        pass
    per_cell = n_per_drone // (S * S)
    rem = n_per_drone - per_cell * S * S

    cell_w = (x_hi - x_lo) / S
    cell_h = (y_hi - y_lo) / S

    total = n_per_drone * n_drones
    pts = np.empty((total, 3), dtype=np.float64)
    drone_idx = np.empty(total, dtype=np.int32)
    k = 0
    for d in range(n_drones):
        # Points du drone d : per_cell par cellule + rem dispatchés.
        extra_cells = rng.choice(S * S, size=rem, replace=False) if rem else []
        extra_set = set(int(c) for c in extra_cells)
        for ci in range(S):
            for cj in range(S):
                n_here = per_cell + (1 if (ci * S + cj) in extra_set else 0)
                if n_here == 0:
                    continue
                x0 = x_lo + ci * cell_w
                x1 = x0 + cell_w
                y0 = y_lo + cj * cell_h
                y1 = y0 + cell_h
                # rejection : on garde ceux strictement au-dessus du toit local.
                got = 0
                while got < n_here:
                    batch = max(n_here - got, 16)
                    xs, ys, zs, hs = _sample_in_cell(
                        batch, x0, x1, y0, y1,
                        height, x_min, y_min, dx, dy, GRID_RES,
                        z_max_map, rng)
                    ok = zs > hs + 1e-3
                    take = min(n_here - got, int(ok.sum()))
                    sel = np.where(ok)[0][:take]
                    pts[k:k + take, 0] = xs[sel]
                    pts[k:k + take, 1] = ys[sel]
                    pts[k:k + take, 2] = zs[sel]
                    drone_idx[k:k + take] = d
                    k += take
                    got += take
    assert k == total
    return pts, drone_idx


# ── Exécution d'un point ─────────────────────────────────────────────────────
def run_one(args):
    idx, drone, x, y, z, tmp_ply, results_drone_dir, map_name = args
    # Chaque worker a son propre PLY d'entrée → sortie unique.
    out_src = tmp_ply.with_name(tmp_ply.stem + "_noisemap.ply")
    out_dst = Path(results_drone_dir) / (
        f"NoiseMap_{map_name}_{x:.4f}_{y:.4f}_{z:.4f}_{drone}.ply"
    )

    cmd = [
        str(NOISEMAP_BIN),
        str(tmp_ply),
        f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
        "--drone", drone,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(BUILD_DIR),
                          capture_output=True, text=True)
    dt = time.time() - t0

    ok = proc.returncode == 0 and out_src.exists()
    spl_min = spl_max = spl_mean = float("nan")
    if ok:
        shutil.move(str(out_src), str(out_dst))
        # Parse stats depuis stderr (spdlog écrit sur stderr par défaut).
        for line in (proc.stderr + proc.stdout).splitlines():
            if "Min" in line and "dB(A)" in line:
                try: spl_min = float(line.split(":")[-1].split("dB")[0])
                except: pass
            elif "Max" in line and "dB(A)" in line:
                try: spl_max = float(line.split(":")[-1].split("dB")[0])
                except: pass
            elif "Mean" in line and "dB(A)" in line:
                try: spl_mean = float(line.split(":")[-1].split("dB")[0])
                except: pass

    return {
        "idx": idx, "drone": drone, "map": map_name,
        "x": x, "y": y, "z": z,
        "ok": ok, "dt": dt,
        "spl_min": spl_min, "spl_max": spl_max, "spl_mean": spl_mean,
        "stderr_tail": "" if ok else proc.stderr[-400:],
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not NOISEMAP_BIN.exists():
        sys.exit(f"Binaire introuvable : {NOISEMAP_BIN}")
    for ply in PLY_INPUTS:
        if not ply.exists():
            sys.exit(f"PLY introuvable : {ply}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for d in DRONES:
        (RESULTS_DIR / d).mkdir(exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    all_tasks = []  # (idx_global, drone, x, y, z, map_name, ply_path)

    # ── Étape 1-2 : lecture + échantillonnage stratifié sur chaque carte ──
    n_maps = len(PLY_INPUTS)
    for ply_path in PLY_INPUTS:
        map_name = ply_path.stem  # "ville", "ville2", …
        print(f"[1/4] Lecture de {ply_path.name}…", flush=True)
        verts = read_ply_vertices(ply_path)
        print(f"      {len(verts)} sommets, bbox "
              f"x=[{verts[:,0].min():.2f},{verts[:,0].max():.2f}] "
              f"y=[{verts[:,1].min():.2f},{verts[:,1].max():.2f}] "
              f"z=[{verts[:,2].min():.2f},{verts[:,2].max():.2f}]", flush=True)

        print(f"[2/4] Échantillonnage stratifié sur {map_name} : "
              f"{POINTS_PER_DRONE_PER_MAP} pts/drone × {len(DRONES)} drones "
              f"= {POINTS_PER_DRONE_PER_MAP * len(DRONES)} pts…", flush=True)
        pts, drone_idx = sample_points_stratified(
            POINTS_PER_DRONE_PER_MAP, len(DRONES), verts, rng)
        for i in range(len(pts)):
            all_tasks.append((
                DRONES[drone_idx[i]],
                float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2]),
                map_name, ply_path,
            ))

    # Mélange global de l'ordre d'exécution (la stratification reste intacte
    # puisqu'elle est déjà encodée dans les couples (pt, drone, map)).
    rng.shuffle(all_tasks)

    # ── Étape 3 : copies temporaires du PLY, une par (worker, carte) ──────
    # Le binaire écrit <input>_noisemap.ply → chaque worker doit avoir son
    # propre fichier d'entrée, et les cartes différentes doivent être
    # séparées pour ne pas écraser le contenu.
    print(f"[3/4] Préparation de {N_WORKERS}×{n_maps} copies PLY worker…",
          flush=True)
    tmp_plys = {}  # (worker_id, map_name) → Path
    for w in range(N_WORKERS):
        for ply_path in PLY_INPUTS:
            map_name = ply_path.stem
            p = TMP_DIR / f"worker_{w}_{map_name}.ply"
            if not p.exists():
                shutil.copy2(ply_path, p)
            tmp_plys[(w, map_name)] = p

    # Construction de la liste de tâches finale avec assignation des workers.
    tasks = []
    for i, (drone, x, y, z, map_name, _ply_path) in enumerate(all_tasks):
        w = i % N_WORKERS
        tmp_ply = tmp_plys[(w, map_name)]
        tasks.append((i, drone, x, y, z,
                      tmp_ply, RESULTS_DIR / drone, map_name))

    print(f"[4/4] Exécution de {len(tasks)} runs sur {N_WORKERS} workers…",
          flush=True)
    csv_path = RESULTS_DIR / "dataset.csv"
    t0 = time.time()
    n_ok = n_ko = 0
    with csv_path.open("w", newline="") as fcsv, \
         ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "drone", "map", "x", "y", "z",
                         "ok", "dt_s", "spl_min", "spl_max", "spl_mean"])
        # On lance en batch par groupe de N_WORKERS pour éviter la
        # concurrence sur le même fichier d'entrée.
        for start in range(0, len(tasks), N_WORKERS):
            chunk = tasks[start:start + N_WORKERS]
            futures = [pool.submit(run_one, t) for t in chunk]
            for fut in as_completed(futures):
                r = fut.result()
                writer.writerow([r["idx"], r["drone"], r["map"],
                                 f"{r['x']:.6f}", f"{r['y']:.6f}", f"{r['z']:.6f}",
                                 int(r["ok"]), f"{r['dt']:.3f}",
                                 r["spl_min"], r["spl_max"], r["spl_mean"]])
                if r["ok"]:
                    n_ok += 1
                else:
                    n_ko += 1
                    print(f"  [FAIL idx={r['idx']}] {r['stderr_tail']}",
                          file=sys.stderr)
            done = start + len(chunk)
            elapsed = time.time() - t0
            eta = elapsed / done * (len(tasks) - done) if done else 0
            print(f"  {done}/{len(tasks)}  "
                  f"ok={n_ok} ko={n_ko}  "
                  f"elapsed={elapsed:.1f}s  eta={eta:.1f}s", flush=True)

    print(f"Terminé : {n_ok} OK / {n_ko} KO en {time.time()-t0:.1f}s")
    print(f"CSV : {csv_path}")
    print(f"PLYs : {RESULTS_DIR}/<drone>/")


if __name__ == "__main__":
    main()
