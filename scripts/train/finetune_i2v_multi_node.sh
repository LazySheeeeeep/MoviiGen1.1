#!/bin/bash 
 
export PYTHONPATH=./:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
OUTPUT_DIR="/dataset-v2/yy/MoviiGen1.1/i2v_ckpts/"
DATA_JSON_PATH="/dataset-v2/yy/MoviiGen1.1/dataset/OpenVidHD_part15_latents/videos2caption_latest.json"
torchrun \
    --nproc_per_node 8 \
    --master_port 29500 \
    scripts/train/finetune_i2v.py \
    --task "i2v-14B" \
    --dataset_type "i2v" \
    --max_seq_len 170100 \
    --master_weight_type bf16 \
    --ckpt_dir  "/dataset-v2/yy/Wan2.1/Wan2.1-I2V-14B-720P/" \
    --output_dir ${OUTPUT_DIR} \
    --checkpointing_steps 100 \
    --seed 42 \
    --gradient_checkpointing \
    --data_json_path ${DATA_JSON_PATH} \
    --train_batch_size 1 \
    --num_latent_t 21 \
    --sp_size 8 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000 \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --num_height 1080 \
    --num_width 1920 \
    --group_frame \
    --group_resolution \
    --group_ar
