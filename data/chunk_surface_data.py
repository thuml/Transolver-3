"""
Chunk the DrivAerML surface data into stride-sampled parts.

The raw DrivAerML surface data is expected to be organized as:
    <data_root>/
        run_1/
            boundary_1.vtp    # surface mesh with CellData fields
        run_2/
            boundary_2.vtp
        ...

Each VTP file is read with PyVista. Cell centers, cell normals, cell areas,
and all CellData fields are extracted and split into `num_parts` parts using
stride sampling:
    part0 = cells[0::num_parts]
    part1 = cells[1::num_parts]
    ...

Requires: pip install pyvista

Usage:
    python chunk_surface_data.py --data_root /path/to/raw --out_root /path/to/chunked --num_parts 20
"""

import os
import argparse
import numpy as np
import pyvista as pv
from tqdm import tqdm


def chunk_vtp_file(vtp_path, out_dir, num_parts=20):
    vtp_name = os.path.splitext(os.path.basename(vtp_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    mesh = pv.read(vtp_path)
    coords = mesh.cell_centers().points      # (N, 3)
    normals = mesh.cell_normals              # (N, 3)
    cell_sizes = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(cell_sizes["Area"])   # (N,)
    N = coords.shape[0]

    var_names = list(mesh.cell_data.keys())
    assert len(var_names) > 0, f"No CellData found in {vtp_path}"

    prefix = vtp_name
    for idx in tqdm(range(num_parts), desc=os.path.basename(vtp_path), leave=False):
        np.save(os.path.join(out_dir, f"{prefix}_points_part{idx}.npy"),  coords[idx::num_parts])
        np.save(os.path.join(out_dir, f"{prefix}_normals_part{idx}.npy"), normals[idx::num_parts])
        np.save(os.path.join(out_dir, f"{prefix}_area_part{idx}.npy"),    areas[idx::num_parts])
        for name in var_names:
            np.save(
                os.path.join(out_dir, f"{prefix}_{name}_part{idx}.npy"),
                mesh.cell_data[name][idx::num_parts]
            )


def chunk_all_vtp(data_root, out_root, num_parts=20):
    os.makedirs(out_root, exist_ok=True)

    run_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    print(f"Found {len(run_dirs)} run directories")

    for run_name in run_dirs:
        run_path = os.path.join(data_root, run_name)
        out_run_dir = os.path.join(out_root, run_name)

        vtp_files = [f for f in os.listdir(run_path) if f.endswith(".vtp")]
        if len(vtp_files) != 1:
            print(f"Warning: expected 1 VTP in {run_name}, found {len(vtp_files)} — skipping")
            continue

        vtp_path = os.path.join(run_path, vtp_files[0])
        print(f"Processing {run_name}")
        chunk_vtp_file(vtp_path, out_run_dir, num_parts=num_parts)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root directory of raw DrivAerML surface data (one VTP per run)")
    parser.add_argument("--out_root", required=True,
                        help="Output directory for chunked data")
    parser.add_argument("--num_parts", type=int, default=20,
                        help="Number of stride-sampled parts per run (default: 20)")
    args = parser.parse_args()

    chunk_all_vtp(
        data_root=args.data_root,
        out_root=args.out_root,
        num_parts=args.num_parts,
    )
