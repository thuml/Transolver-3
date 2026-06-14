export CUDA_VISIBLE_DEVICES="0"

DATA_DIR="/workspace2/zhouhang/driverml/DrivaerML_full_data_npy_chunked_interval_400"
JSON_FILE="/workspace2/zhouhang/Transolver-3/drivaerml.json"
CKPT="/workspace/zhouhang/Transolver++_DrivAerML/TrainAndTest/transolver_volume_chunk_ema_0d5_eqd_cell_600_epoch_normal_chunk_checkpoint_matrix_mul_16layer_80000_ab_upt_cosine_annealing_cpt/model_600_state_dict.pth"
SAVE_DIR="./output/transolver_volume_eval"

# -----------------------------------------------------------------------
# --eval 1: Standard evaluation (single GPU)
# -----------------------------------------------------------------------
# python main_drivaerml_volume.py \
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
python main_drivaerml_volume.py \
    --cfd_model=Transolver_chunk_opt_matrix_mul \
    --data_dir "$DATA_DIR" \
    --json_file "$JSON_FILE" \
    --model_ckpt "$CKPT" \
    --save_dir "$SAVE_DIR" \
    --gpu 0 \
    --eval 2 \
    --max_ref_chunks 10
