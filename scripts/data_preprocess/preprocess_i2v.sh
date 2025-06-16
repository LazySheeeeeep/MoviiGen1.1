PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8
MODEL_PATH="/dataset-v2/yy/Wan2.1/Wan2.1-I2V-14B-720P/"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="/dataset-v2/yy/MoviiGen1.1/dataset/part15_merged.txt"
OUTPUT_DIR="/dataset-v2/yy/MoviiGen1.1/dataset/OpenVidHD_part15_latents/"

torchrun --nproc_per_node=$GPU_NUM --master-port 29512\
    scripts/data_preprocess/preprocess_wan_dataset.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --num_frames 81 \
    --train_batch_size 1 \
    --max_height 1080 \
    --max_width 1920 \
    --dataloader_num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --video_length_tolerance_range 3 \
    --drop_short_ratio 0.0 \
    --random_crop \
    --include_video \
    --include_prompt \
    --dataset "i2v"