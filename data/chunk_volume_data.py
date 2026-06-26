"""
Chunk the DrivAerML volume data into stride-sampled parts.

The raw DrivAerML volume data is expected to be organized as:
    <data_root>/
        run_1/
            run_1_cell_centers.npy       # (N, 3) cell center XYZ coordinates
            run_1_CptMeanTrim.npy        # (N,)   pressure coefficient
            run_1_UMeanTrim.npy          # (N, 3) mean velocity
        run_2/
            ...

Each run is split into `interval` parts using stride sampling:
    part0 = points[0::interval]
    part1 = points[1::interval]
    ...

This ensures each part uniformly covers the full geometry, making any
single part a valid representative sample for physical state caching.

Usage:
    python chunk_volume_data.py --data_root /path/to/raw --out_root /path/to/chunked --interval 400
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def chunk_run(run_dir, out_dir, run_name, label_fields, interval=400):
    points_path = os.path.join(run_dir, f"{run_name}_cell_centers.npy")
    if not os.path.exists(points_path):
        print(f"Warning: points file not found: {points_path}")
        return

    points = np.load(points_path)
    N = points.shape[0]

    labels = {}
    for field in label_fields:
        path = os.path.join(run_dir, f"{run_name}_{field}.npy")
        if not os.path.exists(path):
            print(f"Warning: label file not found: {path}")
            return
        labels[field] = np.load(path)

    save_dir = os.path.join(out_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(interval):
        part_points_path = os.path.join(save_dir, f"{run_name}_cell_centers_part{idx}.npy")
        if os.path.exists(part_points_path):
            continue

        np.save(part_points_path, points[idx::interval])
        for field in label_fields:
            np.save(
                os.path.join(save_dir, f"{run_name}_{field}_part{idx}.npy"),
                labels[field][idx::interval]
            )


def chunk_all_runs(data_root, out_root, run_list, label_fields, interval=400):
    os.makedirs(out_root, exist_ok=True)
    for run_name in tqdm(run_list, desc="Chunking runs"):
        run_dir = os.path.join(data_root, run_name)
        chunk_run(run_dir, out_root, run_name, label_fields, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root directory of raw DrivAerML volume data")
    parser.add_argument("--out_root", required=True,
                        help="Output directory for chunked data")
    parser.add_argument("--interval", type=int, default=400,
                        help="Number of stride-sampled parts per run (default: 400)")
    parser.add_argument("--n_runs", type=int, default=500,
                        help="Total number of runs (default: 500)")
    args = parser.parse_args()

    label_fields = ["CptMeanTrim", "UMeanTrim"]
    run_list = [f"run_{i}" for i in range(1, args.n_runs + 1)]

    chunk_all_runs(
        data_root=args.data_root,
        out_root=args.out_root,
        run_list=run_list,
        label_fields=label_fields,
        interval=args.interval,
    )
