import os

import torch
from wan.modules.t5 import T5EncoderModel
from safetensors import safe_open
from torch.nn import functional as F

from scripts.train.model.model_seq import WanModel


def load_wan(config, checkpoint_dir, device_id, rank, weight_path=None, model_type='t2v'):
    transformers = WanModel.from_pretrained(checkpoint_dir, model_type=model_type)
    if weight_path:
        state_dict = load_weights(weight_path)
        result = transformers.load_state_dict(state_dict, strict=True)
        if rank <= 0:
            print("Resume Missing keys:", result.missing_keys)
            print("Resume Unexpected keys:", result.unexpected_keys)
            print(f"load weights from {weight_path} success!")
    return transformers


def save_null_pt(model_path="ZuluVision/MoviiGen1.1"):
    LEN = 512
    text_encoder = T5EncoderModel(
        text_len=LEN,
        dtype=torch.bfloat16,
        device="cuda",
        checkpoint_path=os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(model_path, "google/umt5-xxl"),
        shard_fn= None,
    )
    null_encoded = text_encoder("",device="cuda")[0]
    print(null_encoded.shape)
    pad_len = LEN - null_encoded.shape[0]
    null_encoded = F.pad(
        null_encoded,
        (0, 0, 0, pad_len),  # (左边填充, 右边填充, 上边填充, 下边填充)
        value=0
    )
    print(null_encoded.shape)
    torch.save(null_encoded, "data/null.pt")

def load_weights(weight_path, device="cpu"):
    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


if __name__ == "__main__":
    save_null_pt()
