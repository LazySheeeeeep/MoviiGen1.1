#!/bin/bash
export PYTHONPATH=/dataset-v2/yy/MoviiGen1.1:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 定义步数数组
steps=(400 500 600 700 800 900)

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
        --size 1920*1080 \
        --ckpt_dir "/dataset-v2/yy/MoviiGen1.1/i2v_ckpts/checkpoints/checkpoint-step_${step}" \
        --other_ckpt_dir /dataset-v2/yy/Wan2.1/Wan2.1-I2V-14B-720P \
        --image /dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_1.png \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 8 \
        --base_seed 42 \
        --frame_num 81 \
        --prompt "A smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film. A world-weary detective is sitting behind the desk. He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light. The scene is rendered in stark black and white, creating a high-contrast, cinematic mood. The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere." \
        --save_file "${output_dir}/1920*1080_1.mp4"
    
    echo "Completed checkpoint step: $step"
    echo "----------------------------------------"
done

echo "All checkpoints processed successfully!"
