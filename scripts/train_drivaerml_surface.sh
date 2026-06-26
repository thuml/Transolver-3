export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Set these two paths before running
DATA_DIR="/path/to/DrivAerML_surface_chunked"   # <-- change to your chunked surface data directory
SAVE_DIR="./output/transolver_surface"           # <-- change to your desired output directory

torchrun --nproc_per_node 1 --master_port 10001 main_drivaerml_surface.py \
    --nb_epochs 800 \
    --dataset drivAerML \
    --cfd_model=Transolver_chunk_opt_matrix_mul \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --eval 0
