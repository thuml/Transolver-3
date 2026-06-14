export CUDA_VISIBLE_DEVICES="4"

DATA_DIR="/workspace2/zhouhang/DrivAerML_surface_cell_chunked_20parts_with_normal_with_measure"
JSON_FILE="/workspace2/zhouhang/Transolver-3/drivaerml.json"
CKPT="/workspace/zhouhang/Transolver++_DrivAerML/TrainAndTest/transolver_surface_chunk_ema_0d5_eqd_cell_800_epoch_normal_measure_24layer_input_100000points/model_800_state_dict.pth"
SAVE_DIR="./output/transolver_surface_eval"

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
