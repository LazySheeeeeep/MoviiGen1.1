#!/bin/bash
export PYTHONPATH=/dataset-v2/yy/MoviiGen1.1:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    
# 创建输出目录
output_dir="/dataset-v2/yy/MoviiGen1.1/i2v_output/wan"
mkdir -p "$output_dir"

sizes=('1280*720' '1920*1080')

for size in "${sizes[@]}"; do
    echo "Processing size: $size"
    # 运行推理
    torchrun --nproc_per_node 8 \
        --master-port 29500 \
        /dataset-v2/yy/MoviiGen1.1/scripts/inference/generate_i2v.py \
        --task i2v-14B \
        --size $size \
        --ckpt_dir /dataset-v2/yy/Wan2.1/Wan2.1-I2V-14B-720P \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 8 \
        --base_seed 42 \
        --frame_num 81 \
        --use_predefine_input \
        --save_dir "${output_dir}/" \
        --predefine_file "/dataset-v2/yy/MoviiGen1.1/predefine_input.json"

    echo "Completed size: $size"
    echo "#############################################"
done

echo "All sizes processed successfully!"
