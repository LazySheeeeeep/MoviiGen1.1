# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "image": None,
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        "image": None,
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
        "image": None,
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}

PRE_DEFINE_ROOT = {
    "1280*720": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720",
    "1920*1080": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080"
}

PRE_DEFINE_INPUT = {
    "1280*720": [
        # {"prompt": "A smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film. A world-weary detective is sitting behind the desk. He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light. The scene is rendered in stark black and white, creating a high-contrast, cinematic mood. The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere.",
        # "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720_1.png",
        # "file_name": "1280*720_1.mp4"},
        {"prompt": "A sunlit outdoor scene with soft, natural light illuminating a patch of lush green grass in the background. A Border Collie with a rich brown and white coat stands in profile as the main subject, its fur detailed and windswept, ears perked and eyes focused on something out of frame, conveying alertness and intelligence. The dog lifts its head slightly, nose twitching as it sniffs the air, while its tail gives a gentle wag, showing anticipation and curiosity. The moment feels lively and attentive, as if the dog is ready to spring into action at any second. The image features a shallow depth of field, creating a creamy bokeh behind the dog and drawing all attention to its expressive profile. The camera uses a static, close-up side shot to capture the texture of the fur and the thoughtful, animated gaze, emphasizing the serene and natural setting.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720_2.png",
        "file_name": "1280*720_2.mp4"},
        {"prompt": "In a serene, misty outdoor setting, soft natural light gently filters through lush green foliage, creating a dreamy and ethereal atmosphere. The main subject, a young woman wearing a traditional, colorful patterned dress, stands waist-deep in calm water. She holds a woven conical hat in one hand, her other hand lightly touching the water’s surface. Her face is turned slightly upward, eyes sparkling and a gentle smile on her lips, as if she is lost in a joyful thought or greeting the morning light. The scene captures a sense of tranquility and cultural beauty, with the water reflecting subtle colors and the background softly blurred. The camera employs a steady, medium close-up shot, focusing on her expression and the texture of the hat, emphasizing the peaceful, poetic mood.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720_3.png",
        "file_name": "1280*7200_3.mp4"},
        {"prompt": "In a sun-dappled meadow surrounded by wildflowers and tall grasses, soft golden sunlight filters through the trees, casting a warm and inviting glow over the scene. The main subject, a young girl with long blonde hair, sits cross-legged on the grass, holding a small guitar in her lap. She strums the strings with one hand, her fingers skillfully moving along the frets, absorbed in the music. Suddenly, she lifts her head with a bright, joyful smile, her eyes sparkling with happiness as she glances upward, sharing her delight with the world around her. The atmosphere is cheerful and carefree, filled with the innocence of childhood and the peacefulness of nature. The camera captures a gentle, medium shot, focusing on her expression and the movement of her hands, emphasizing the warmth and spontaneity of the moment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720_4.png",
        "file_name": "1280*720_4.mp4"},
        {"prompt": "A tranquil forest scene filled with tall trees and dense green foliage, dappled sunlight filtering through the canopy and casting soft, warm light onto the forest floor. A narrow, winding dirt path leads deeper into the woods, surrounded by vibrant undergrowth and the gentle rustle of leaves. The atmosphere is peaceful and inviting, evoking a sense of exploration and quiet wonder. The camera adopts a first-person perspective, moving steadily forward along the path, as if the viewer is walking through the forest. The steady, immersive motion draws attention to the textures of the earth and trees, capturing the serene beauty and natural rhythm of the woodland environment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1280_720_5.png",
        "file_name": "1280*720_5.mp4"},
    ],
    "1920*1080": [
        {"prompt": "A smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film. A world-weary detective is sitting behind the desk. He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light. The scene is rendered in stark black and white, creating a high-contrast, cinematic mood. The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_1.png",
        "file_name": "1920*1080_1.mp4"},
        {"prompt": "A sunlit outdoor scene with soft, natural light illuminating a patch of lush green grass in the background. A Border Collie with a rich brown and white coat stands in profile as the main subject, its fur detailed and windswept, ears perked and eyes focused on something out of frame, conveying alertness and intelligence. The dog lifts its head slightly, nose twitching as it sniffs the air, while its tail gives a gentle wag, showing anticipation and curiosity. The moment feels lively and attentive, as if the dog is ready to spring into action at any second. The image features a shallow depth of field, creating a creamy bokeh behind the dog and drawing all attention to its expressive profile. The camera uses a static, close-up side shot to capture the texture of the fur and the thoughtful, animated gaze, emphasizing the serene and natural setting.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_2.png",
        "file_name": "1920*1080_2.mp4"},
        {"prompt": "In a serene, misty outdoor setting, soft natural light gently filters through lush green foliage, creating a dreamy and ethereal atmosphere. The main subject, a young woman wearing a traditional, colorful patterned dress, stands waist-deep in calm water. She holds a woven conical hat in one hand, her other hand lightly touching the water’s surface. Her face is turned slightly upward, eyes sparkling and a gentle smile on her lips, as if she is lost in a joyful thought or greeting the morning light. The scene captures a sense of tranquility and cultural beauty, with the water reflecting subtle colors and the background softly blurred. The camera employs a steady, medium close-up shot, focusing on her expression and the texture of the hat, emphasizing the peaceful, poetic mood.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_3.png",
        "file_name": "1920*1080_3.mp4"},
        {"prompt": "In a sun-dappled meadow surrounded by wildflowers and tall grasses, soft golden sunlight filters through the trees, casting a warm and inviting glow over the scene. The main subject, a young girl with long blonde hair, sits cross-legged on the grass, holding a small guitar in her lap. She strums the strings with one hand, her fingers skillfully moving along the frets, absorbed in the music. Suddenly, she lifts her head with a bright, joyful smile, her eyes sparkling with happiness as she glances upward, sharing her delight with the world around her. The atmosphere is cheerful and carefree, filled with the innocence of childhood and the peacefulness of nature. The camera captures a gentle, medium shot, focusing on her expression and the movement of her hands, emphasizing the warmth and spontaneity of the moment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_4.png",
        "file_name": "1920*1080_4.mp4"},
        {"prompt": "A tranquil forest scene filled with tall trees and dense green foliage, dappled sunlight filtering through the canopy and casting soft, warm light onto the forest floor. A narrow, winding dirt path leads deeper into the woods, surrounded by vibrant undergrowth and the gentle rustle of leaves. The atmosphere is peaceful and inviting, evoking a sense of exploration and quiet wonder. The camera adopts a first-person perspective, moving steadily forward along the path, as if the viewer is walking through the forest. The steady, immersive motion draws attention to the textures of the earth and trees, capturing the serene beauty and natural rhythm of the woodland environment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/1920_1080_5.png",
        "file_name": "1920*1080_5.mp4"},
    ],
    "2560*1440": [
        {"prompt": "A sunlit outdoor scene with soft, natural light illuminating a patch of lush green grass in the background. A Border Collie with a rich brown and white coat stands in profile as the main subject, its fur detailed and windswept, ears perked and eyes focused on something out of frame, conveying alertness and intelligence. The dog lifts its head slightly, nose twitching as it sniffs the air, while its tail gives a gentle wag, showing anticipation and curiosity. The moment feels lively and attentive, as if the dog is ready to spring into action at any second. The image features a shallow depth of field, creating a creamy bokeh behind the dog and drawing all attention to its expressive profile. The camera uses a static, close-up side shot to capture the texture of the fur and the thoughtful, animated gaze, emphasizing the serene and natural setting.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/2560_1440_2.png",
        "file_name": "2560*1440_2.mp4"},
        {"prompt": "In a serene, misty outdoor setting, soft natural light gently filters through lush green foliage, creating a dreamy and ethereal atmosphere. The main subject, a young woman wearing a traditional, colorful patterned dress, stands waist-deep in calm water. She holds a woven conical hat in one hand, her other hand lightly touching the water’s surface. Her face is turned slightly upward, eyes sparkling and a gentle smile on her lips, as if she is lost in a joyful thought or greeting the morning light. The scene captures a sense of tranquility and cultural beauty, with the water reflecting subtle colors and the background softly blurred. The camera employs a steady, medium close-up shot, focusing on her expression and the texture of the hat, emphasizing the peaceful, poetic mood.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/2560_1440_3.png",
        "file_name": "2560*1440_3.mp4"},
        {"prompt": "In a sun-dappled meadow surrounded by wildflowers and tall grasses, soft golden sunlight filters through the trees, casting a warm and inviting glow over the scene. The main subject, a young girl with long blonde hair, sits cross-legged on the grass, holding a small guitar in her lap. She strums the strings with one hand, her fingers skillfully moving along the frets, absorbed in the music. Suddenly, she lifts her head with a bright, joyful smile, her eyes sparkling with happiness as she glances upward, sharing her delight with the world around her. The atmosphere is cheerful and carefree, filled with the innocence of childhood and the peacefulness of nature. The camera captures a gentle, medium shot, focusing on her expression and the movement of her hands, emphasizing the warmth and spontaneity of the moment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/2560_1440_4.png",
        "file_name": "2560*1440_4.mp4"},
        {"prompt": "A tranquil forest scene filled with tall trees and dense green foliage, dappled sunlight filtering through the canopy and casting soft, warm light onto the forest floor. A narrow, winding dirt path leads deeper into the woods, surrounded by vibrant undergrowth and the gentle rustle of leaves. The atmosphere is peaceful and inviting, evoking a sense of exploration and quiet wonder. The camera adopts a first-person perspective, moving steadily forward along the path, as if the viewer is walking through the forest. The steady, immersive motion draws attention to the textures of the earth and trees, capturing the serene beauty and natural rhythm of the woodland environment.",
        "image": "/dataset-v2/yy/MoviiGen1.1/i2v_img/2560_1440_5.png",
        "file_name": "2560*1440_5.mp4"},
    ],
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--other_ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory for encoder and other models.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--use_predefine_input",
        action="store_true",
        default=False,
        help="Whether to use predefined prompt and image (for I2V).")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The dir to save the generated image or video to.")
    parser.add_argument(
        "--predefine_file",
        type=str,
        default=None,
        help="The file saved the predifined information.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def get_input(args):
    import json

    pre_define_root = PRE_DEFINE_ROOT[args.size]
    with open(args.predefine_file, 'r', encoding='utf-8') as f:
        pre_define_input = json.load(f)

    for item in pre_define_input:
        image_name = item['image']

        # 合成完整路径
        full_path = os.path.join(pre_define_root, image_name)
        item['image'] = full_path

        # 生成 file_name
        file_name = f"{args.size}_{image_name.replace('-', '_').replace('.png', '.mp4')}"
        item['file_name'] = file_name

    return pre_define_input


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    pre_define_input = []
    if args.use_predefine_input:
        assert args.predefine_file, "The predifine file can't be None when using predefine input."
        pre_define_input = get_input(args)
    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        pre_define_input.append({"prompt": args.prompt, "image": args.image, "file_name": None})

    if "t2v" in args.task or "t2i" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            other_ckpt_dir=args.other_ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

    for input in pre_define_input:
        prompt = input["prompt"]
        image = input["image"]
        if "t2v" in args.task or "t2i" in args.task:
            logging.info(f"Input prompt: {prompt}")
            if args.use_prompt_extend:
                logging.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = prompt_expander(
                        prompt,
                        tar_lang=args.prompt_extend_target_lang,
                        seed=args.base_seed)
                    if prompt_output.status == False:
                        logging.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logging.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logging.info(f"Extended prompt: {prompt}")

            logging.info(
                f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)

        else:
            assert image, "I2V task must have an image input."
            logging.info(f"Input prompt: {prompt}")
            logging.info(f"Input image: {image}")

            img = Image.open(image).convert("RGB")
            if args.use_prompt_extend:
                logging.info("Extending prompt ...")
                if rank == 0:
                    prompt_output = prompt_expander(
                        prompt,
                        tar_lang=args.prompt_extend_target_lang,
                        image=img,
                        seed=args.base_seed)
                    if prompt_output.status == False:
                        logging.info(
                            f"Extending prompt failed: {prompt_output.message}")
                        logging.info("Falling back to original prompt.")
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
                logging.info(f"Extended prompt: {prompt}")

            logging.info("Generating video ...")
            video = wan_i2v.generate(
                prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model)

        if rank == 0:
            if args.use_predefine_input:
                if args.save_dir is None:
                    args.save_dir = '.'
                args.save_file = None
            
            file_name = input["file_name"]

            if args.save_file is None:
                if file_name is None:
                    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    formatted_prompt = prompt.replace(" ", "_").replace("/",
                                                                            "_")[:50]
                    suffix = '.png' if "t2i" in args.task else '.mp4'
                    args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix
                    args.save_file = os.path.join(args.save_dir, args.save_file)
                else:
                    args.save_file = os.path.join(args.save_dir, file_name)

            if "t2i" in args.task:
                logging.info(f"Saving generated image to {args.save_file}")
                cache_image(
                    tensor=video.squeeze(1)[None],
                    save_file=args.save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
            else:
                logging.info(f"Saving generated video to {args.save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=args.save_file,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
