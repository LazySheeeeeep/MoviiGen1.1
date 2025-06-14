export PYTHONPATH=/dataset-v2/yy/MoviiGen1.1:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
 --nproc_per_node 4 \
 --master-port 29500 \
 /dataset-v2/yy/MoviiGen1.1/scripts/inference/generate.py \
 --task t2v-14B \
 --size 1920*832 \
 --sample_steps 50 \
 --ckpt_dir ./MoviiGen1.1 \
 --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 --sample_shift 5\
 --sample_guide_scale 5.0 --prompt "A close-up shot of a regal queen sitting on an ornate golden throne. Her porcelain-like skin glows softly from light filtering through stained glass windows. Her piercing ice-blue eyes, framed by long lashes, exude calm authority. Wavy auburn hair with golden highlights cascades over her shoulders, complementing a delicate, jewel-encrusted crown that sparkles subtly. She wears an emerald green gown adorned with gold embroidery of vines and flowers, its intricate details visible even in the tight frame. In her right hand rests a silver scepter tipped with a glowing sapphire, casting a faint blue light on her fingers. Her expression is serene yet touched with sorrow, reflecting the burden of her rule. The scene is rendered with fine detail, emphasizing the textures of her gown, the brilliance of her crown, and the solemnity of her gaze."\
 --base_seed 42 --frame_num 81\
 --save_file ./output/1920*832_4.mp4
