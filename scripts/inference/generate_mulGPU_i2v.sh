#!/bin/bash
export PYTHONPATH=/dataset-v2/yy/MoviiGen1.1:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 定义步数数组
steps=(100 200 300 400 500 600 700 800 900)

# 循环处理每个检查点
for step in "${steps[@]}"; do
    echo "Processing checkpoint step: $step"
    
    # 创建输出目录
    output_dir="/dataset-v2/yy/MoviiGen1.1/i2v_output/6_16_test/step_${step}"
    mkdir -p "$output_dir"
    
    # 运行推理
    torchrun --nproc_per_node 8 \
        --master-port 29500 \
        /dataset-v2/yy/MoviiGen1.1/scripts/inference/generate_i2v.py \
        --task i2v-14B \
        --size 1280*720 \
        --ckpt_dir "/dataset-v2/yy/MoviiGen1.1/i2v_ckpts/checkpoints/checkpoint-step_${step}" \
        --other_ckpt_dir /dataset-v2/yy/Wan2.1/Wan2.1-I2V-14B-720P \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 8 \
        --base_seed 42 \
        --frame_num 81 \
        --use_predefine_input \
        --save_dir "${output_dir}/" \
        --predefine_file "/dataset-v2/yy/MoviiGen1.1/predefine_input.json"
    
    echo "Completed checkpoint step: $step"
    echo "----------------------------------------"
done

echo "All checkpoints processed successfully!"
