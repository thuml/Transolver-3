"""
Extract per-run NPY files from DrivAerML volume VTU files.

DrivAerML distributes volume data as VTU files (one per run). Some download
platforms split each VTU into numbered parts; if so, reassemble first:

    cat volume_1.vtu.part* > volume_1.vtu

This script reads each assembled VTU with PyVista, extracts cell-centered
coordinates and the requested field arrays, and saves them as NPY files:

    <out_root>/run_1/run_1_cell_centers.npy   # (N, 3)  float64
    <out_root>/run_1/run_1_CptMeanTrim.npy    # (N,)    float32
    <out_root>/run_1/run_1_UMeanTrim.npy      # (N, 3)  float32

These NPY files are the input expected by chunk_volume_data.py.

Available fields in the DrivAerML VTU (all stored as CellData):
    CpMeanTrim, CptMeanTrim, magUMeanNormTrim, microDragMeanTrim,
    nutMeanTrim, pMeanTrim, pPrime2MeanTrim, UMeanTrim,
    UPrime2MeanTrim, turbulenceProperties:RMeanTrim

Usage:
    python extract_volume_npy.py --data_root /path/to/VTUs --out_root /path/to/npy
"""

import os
import argparse
import numpy as np
import pyvista as pv
from tqdm import tqdm


def extract_run(run_dir, out_dir, run_name, fields):
    vtu_path = os.path.join(run_dir, f"volume_{run_name.split('_')[1]}.vtu")
    if not os.path.exists(vtu_path):
        print(f"Warning: VTU not found: {vtu_path}")
        return

    out_run_dir = os.path.join(out_dir, run_name)
    centers_path = os.path.join(out_run_dir, f"{run_name}_cell_centers.npy")
    if os.path.exists(centers_path):
        return  # already done

    os.makedirs(out_run_dir, exist_ok=True)
    mesh = pv.read(vtu_path)

    np.save(centers_path, mesh.cell_centers().points)

    for field in fields:
        arr = np.array(mesh.cell_data[field])
        np.save(os.path.join(out_run_dir, f"{run_name}_{field}.npy"), arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing run_xxx/volume_xxx.vtu files")
    parser.add_argument("--out_root", required=True,
                        help="Output directory for per-run NPY files")
    parser.add_argument("--n_runs", type=int, default=500,
                        help="Total number of runs (default: 500)")
    parser.add_argument("--fields", nargs="+",
                        default=["CptMeanTrim", "UMeanTrim"],
                        help="CellData fields to extract (default: CptMeanTrim UMeanTrim)")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    run_list = [f"run_{i}" for i in range(1, args.n_runs + 1)]

    for run_name in tqdm(run_list, desc="Extracting"):
        run_dir = os.path.join(args.data_root, run_name)
        extract_run(run_dir, args.out_root, run_name, args.fields)
