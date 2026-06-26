# Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries

This repository provides the official implementation of **Transolver-3**, a neural surrogate for CFD prediction on industrial-scale unstructured meshes with 160M+ cells. It includes training and evaluation code for the [DrivAerML](https://github.com/DrivAerML/DrivAerML) benchmark (surface pressure/wall shear stress and volume flow fields).

**Paper**: [Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries](https://arxiv.org/abs/2602.04940)

## Requirements

```bash
pip install -r requirements.txt
```

For data preparation, PyVista is additionally required:

```bash
pip install pyvista
```

## Data Preparation

### Download DrivAerML

Download the DrivAerML dataset from the [official repository](https://github.com/DrivAerML/DrivAerML). The dataset contains 500 aerodynamic simulations of the DrivAer geometry.

- **Surface data**: one `.vtp` file per run (e.g. `boundary_1.vtp`), containing cell-centered surface fields
- **Volume data**: one `.vtu` file per run (e.g. `volume_1.vtu`), containing cell-centered volume fields. The intermediate per-run `.npy` files expected by `chunk_volume_data.py` are exported directly from these `.vtu` files using PyVista (`mesh.cell_centers().points` for coordinates, `mesh.cell_data[field]` for each field).

### Chunk the Data

Both pipelines use stride-based sampling to split each run into `K` parts:

```
part_k = all_cells[k::K],  k = 0, 1, ..., K-1
```

This ensures every part uniformly covers the full geometry, making any single part a valid representative for physical state caching.

**Surface** (VTP → chunked NPY, `K=20`):

```bash
python data/chunk_surface_data.py \
    --data_root /path/to/DrivAerML_surface \
    --out_root  /path/to/DrivAerML_surface_chunked \
    --num_parts 20
```

Expected input layout:
```
<data_root>/
  run_1/boundary_1.vtp
  run_2/boundary_2.vtp
  ...
```

Output layout:
```
<out_root>/
  run_1/
    boundary_1_points_part0.npy        # (N/20, 3)  XYZ coordinates
    boundary_1_normals_part0.npy       # (N/20, 3)  cell normals
    boundary_1_area_part0.npy          # (N/20,)    cell areas
    boundary_1_pMeanTrim_part0.npy     # (N/20,)    mean pressure
    boundary_1_wallShearStressMeanTrim_part0.npy  # (N/20, 3)
    ...  (part1 through part19)
  run_2/
    ...
```

**Volume** (VTU → NPY → chunked NPY, `K=400`):

Step 1 — extract cell-centered arrays from VTU files (one per run):

```bash
python data/extract_volume_npy.py \
    --data_root /path/to/DrivAerML_volume \
    --out_root  /path/to/DrivAerML_volume_npy
```

If the VTUs were downloaded as split parts (`volume_1.vtu.part*`), reassemble them first:
```bash
cat run_1/volume_1.vtu.part* > run_1/volume_1.vtu
```

Step 2 — stride-sample into chunks:

```bash
python data/chunk_volume_data.py \
    --data_root /path/to/DrivAerML_volume_npy \
    --out_root  /path/to/DrivAerML_volume_chunked \
    --interval  400
```

Expected input layout for Step 2:
```
<data_root>/
  run_1/
    run_1_cell_centers.npy   # (N, 3)
    run_1_CptMeanTrim.npy    # (N,)
    run_1_UMeanTrim.npy      # (N, 3)
  run_2/
    ...
```

Output layout:
```
<out_root>/
  run_1/
    run_1_cell_centers_part0.npy   # (N/400, 3)
    run_1_CptMeanTrim_part0.npy    # (N/400,)
    run_1_UMeanTrim_part0.npy      # (N/400, 3)
    ...  (part1 through part399)
  run_2/
    ...
```

The train/test split is specified by `drivaerml.json`.

## Training

Edit the data and output paths in the training script, then launch with `torchrun`:

```bash
# Surface (4 GPUs, 800 epochs)
bash scripts/train_drivaerml_surface.sh

# Volume (4 GPUs, 600 epochs)
bash scripts/train_drivaerml_volume.sh
```

Key training arguments:

| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Root directory of chunked data | required |
| `--save_dir` | Directory to save checkpoints and logs | required |
| `--json_file` | Path to train/test split JSON | `./drivaerml.json` |
| `--nb_epochs` | Number of training epochs | `200` |
| `--cfd_model` | Model variant (`Transolver_chunk_opt_matrix_mul`) | required |

## Pretrained Checkpoints

We provide pretrained checkpoints for both surface and volume tasks on DrivAerML.

| Task | File |
|------|------|
| Surface (pressure + wall shear stress) | `DrivAerML_surface.pth` |
| Volume (pressure + velocity) | `DrivAerML_volume.pth` |

Download with `gdown` (install via `pip install gdown`):

```bash
# Surface checkpoint
gdown https://drive.google.com/uc?id=19m3-0rI6tcwpkZ7x28YkmZfwA1juWGOo -O DrivAerML_surface.pth

# Volume checkpoint
gdown https://drive.google.com/uc?id=1WIRTe5fWBfT3tWbNt751wRtNsgHb7SSu -O DrivAerML_volume.pth
```

Or browse the [Google Drive folder](https://drive.google.com/drive/folders/1D3CRunLV42OBN1SYpBdkCvvzwdHHbNkp?usp=sharing) directly.

## Evaluation

Edit the paths in the eval script and run on a single GPU:

```bash
# Surface
bash scripts/eval_drivaerml_surface.sh

# Volume
bash scripts/eval_drivaerml_volume.sh
```

Both scripts default to `--eval 2` (decoupled inference, the method proposed in the paper). `--eval 1` (standard single-pass evaluation) is also available as a commented alternative.

### Decoupled Inference Framework

Transolver-3 introduces a two-stage inference framework for memory-efficient prediction on full industrial-scale meshes:

**Stage 1 — Physical state caching**: The model processes the full mesh in memory-compatible chunks, layer by layer, to build a *physical state cache* `s_cache = {s(l)}_{l=1..L}`. Each layer's physical state captures the global physics structure of the simulation without requiring the entire mesh to fit in GPU memory simultaneously.

**Stage 2 — Full mesh decoding**: Using the precomputed physical state cache, the model runs inference at arbitrary query coordinates in a single forward pass per chunk.

Both stages use the same trained checkpoint. The `PhysicalStateCachingModel` and `FullMeshDecodingModel` classes share identical parameters with the standard `Model`, so any trained checkpoint can be loaded into either without modification.

Key eval arguments:

| Argument | Description |
|---|---|
| `--eval 2` | Decoupled inference (physical state caching + full mesh decoding) |
| `--eval 1` | Standard evaluation (single-pass, loads one chunk per run) |
| `--model_ckpt` | Path to trained checkpoint (`.pth` state dict) |
| `--max_ref_chunks` | Max chunks per run used in Stage 1 (default: all) |
| `--gpu` | GPU index to use (single GPU only for eval) |

## Contact

For questions or issues, please contact [zhou-h23@mails.tsinghua.edu.cn](mailto:zhou-h23@mails.tsinghua.edu.cn).
