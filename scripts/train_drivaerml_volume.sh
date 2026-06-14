export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_P2P_LEVEL=NVL

DATA_DIR="/path/to/DrivaerML_volume_chunked"   # <-- set to your chunk directory
SAVE_DIR="./output/transolver_volume"           # <-- set to your output directory
JSON_FILE="./drivaerml.json"                    # <-- path to train/test split JSON

torchrun --nproc_per_node 4 --master_port 10000 main_drivaerml_volume.py \
    --nb_epochs 600 \
    --dataset drivAerML \
    --cfd_model=Transolver_chunk_opt_matrix_mul \
    --data_dir "$DATA_DIR" \
    --json_file "$JSON_FILE" \
    --save_dir "$SAVE_DIR" \
    --eval 0
