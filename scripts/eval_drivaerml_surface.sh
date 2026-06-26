export CUDA_VISIBLE_DEVICES="0"

DATA_DIR="<YOUR_SURFACE_DATA_DIR>"   # e.g. /path/to/DrivAerML_surface_cell_chunked_20parts_with_normal_with_measure
JSON_FILE="./drivaerml.json"
CKPT="<YOUR_CHECKPOINT>"             # e.g. ./DrivAerML_surface.pth  (download with gdown, see README)
SAVE_DIR="./output/transolver_surface_eval"

# -----------------------------------------------------------------------
# --eval 2: Decoupled inference (single GPU)
#   Stage 1 builds the physical state cache layer-by-layer from all chunks.
#   Stage 2 runs full mesh decoding using the cached physical states.
#   --max_ref_chunks limits chunks per run used in Stage 1 (omit = all).
# -----------------------------------------------------------------------
python main_drivaerml_surface.py \
    --cfd_model=Transolver_chunk_opt_matrix_mul \
    --data_dir "$DATA_DIR" \
    --json_file "$JSON_FILE" \
    --model_ckpt "$CKPT" \
    --save_dir "$SAVE_DIR" \
    --gpu 0 \
    --eval 2

# -----------------------------------------------------------------------
# --eval 1: Standard evaluation (single GPU)
# -----------------------------------------------------------------------
# python main_drivaerml_surface.py \
#     --cfd_model=Transolver_chunk_opt_matrix_mul \
#     --data_dir "$DATA_DIR" \
#     --json_file "$JSON_FILE" \
#     --model_ckpt "$CKPT" \
#     --save_dir "$SAVE_DIR" \
#     --gpu 0 \
#     --eval 1
