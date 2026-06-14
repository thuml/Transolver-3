# Transolver

Transolver is a physics-aware transformer for CFD simulation prediction on large-scale unstructured meshes. This repository provides training code for surface pressure and wall shear stress prediction on the [DrivAerML](https://github.com/DrivAerML/DrivAerML) dataset.

## Requirements

```bash
pip install -r requirements.txt
```

## Data Preparation

The training script expects the DrivAerML surface data to be pre-chunked into per-run directories. Each run directory should contain:

```
<data_dir>/
  run_1/
    boundary_1_points_part0.npy       # (N, 3) XYZ coordinates
    boundary_1_normals_part0.npy      # (N, 3) surface normals
    boundary_1_pMeanTrim_part0.npy    # (N,)   mean pressure
    boundary_1_wallShearStressMeanTrim_part0.npy  # (N, 3) wall shear stress
    boundary_1_points_part1.npy
    ...
  run_2/
    ...
```

The train/test split is specified by `drivaerml.json`.

## Training

Edit `scripts/train_drivaerml_surface.sh` to set your data and output directories, then run:

```bash
bash scripts/train_drivaerml_surface.sh
```

Training saves model `state_dict` files such as `model_800.pth` under `--save_dir`.

Key arguments for `main_drivaerml_surface.py`:

| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Root directory of chunked surface data | required |
| `--save_dir` | Directory to save checkpoints and logs | `./output` |
| `--json_file` | Path to train/test split JSON | `./drivaerml.json` |
| `--val_dir` | Validation data directory (defaults to `--data_dir`) | `None` |
| `--nb_epochs` | Number of training epochs | `200` |
| `--cfd_model` | Model variant to use | `Transolver++` |
| `--ema` | EMA decay for slice token statistics | `0.8` |
| `--eval` | Set to `1` to run evaluation only | `0` |
| `--model_ckpt` | Checkpoint path used in eval mode | `./model.pth` |

## Model

The recommended model for surface prediction is `Transolver_chunk_opt_matrix_mul` (24 layers, 256 hidden dim, 64 slices). It processes chunked point clouds with per-surface-normal features and predicts pressure + wall shear stress.

## Evaluation

```bash
torchrun --nproc_per_node 4 main_drivaerml_surface.py \
    --data_dir /path/to/data \
    --save_dir ./output \
    --cfd_model Transolver_chunk_opt_matrix_mul \
    --eval 1 \
    --model_ckpt /path/to/model.pth
```
