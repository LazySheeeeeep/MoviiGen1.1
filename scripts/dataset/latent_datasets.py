import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
        txt_max_len=512,  # For WanX
        prompt_type="prompt_embed_path",
        seed=42,
        resolution_mix=None,
        resolution_mix_p=0.2,
        dataset_type='t2v',
    ):
        assert dataset_type in ['t2v', 'i2v'], "Only support T2V or I2V task."
        self.dataset_type = dataset_type
        # data_merge_path: video_dir, latent_dir, prompt_embed_dir, json_path
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)
        self.video_dir = os.path.join(self.datase_dir_path, "video")
        self.latent_dir = os.path.join(self.datase_dir_path, "latent")
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.prompt_attention_mask_dir = os.path.join(self.datase_dir_path, "prompt_attention_mask")

        if self.dataset_type == 'i2v':
            self.y_dir = os.path.join(self.datase_dir_path, "y")
            self.clip_fea_dir = os.path.join(self.datase_dir_path, "clip_feature")

        with open(self.json_path, "r") as f:
            data_annos = json.load(f)

        self.data_anno = []

        if "aspect_ratio_bin" in data_annos[0]:
            keep_ratio = [0, 1]
            aspect_ratios = []
            for anno in data_annos:
                if anno["aspect_ratio_bin"] in keep_ratio:
                    self.data_anno.append(anno)
                    aspect_ratios.append(anno["aspect_ratio_bin"])

            self.aspect_ratios = np.array(aspect_ratios)

        # json.load(f) already keeps the order
        # self.data_anno = sorted(self.data_anno, key=lambda x: x['latent_path'])
        self.num_latent_t = num_latent_t
        self.txt_max_len = txt_max_len
        # just zero embeddings [256, 4096]
        self.uncond_prompt_embed = torch.zeros(512, 4096).to(torch.float32)
        # 512 zeros
        self.uncond_prompt_mask = torch.zeros(512).bool()
        self.lengths = [data_item["length"] if "length" in data_item else 1 for data_item in self.data_anno]
        self.prompt_type = prompt_type

        self.base_seed = seed
        self.resolution_mix = resolution_mix
        self.resolution_mix_p = resolution_mix_p

    # Add this method
    def set_epoch(self, epoch):
        """Sets the current epoch for deterministic random choices."""
        self.epoch = epoch

    def __getitem__(self, idx):
        latent_file = self.data_anno[idx]["latent_path"]
        try:
            prompt_embed_file = self.data_anno[idx][self.prompt_type]
            prompt_embed_dir = self.prompt_embed_dir
        except:
            prompt_embed_candidates = self.prompt_type.split(",")
            prompt_embed_file = latent_file
            this_type = np.random.choice(prompt_embed_candidates, 1, p=[0.2, 0.3, 0.5])[0]
            prompt_embed_dir = os.path.join(self.datase_dir_path, this_type)

        if "prompt_attention_mask" in self.data_anno[idx]:
            prompt_attention_mask_file = self.data_anno[idx]["prompt_attention_mask"]
        else:
            prompt_attention_mask_file = None

        latent_dir = os.path.join(self.datase_dir_path, "latent")

        if self.resolution_mix is not None:
            item_epoch_seed = self.base_seed + self.epoch + idx
            local_random = random.Random(item_epoch_seed)
            if local_random.random() < self.resolution_mix_p:
                latent_dir = os.path.join(self.datase_dir_path, self.resolution_mix)

        # load
        latent = torch.load(
            os.path.join(latent_dir, latent_file),
            map_location="cpu",
            weights_only=True,
        )

        latent = latent.squeeze(0)[:, -self.num_latent_t:]
        #  print(f"original latent shape: {original_shape} ==> {latent.shape}")

        if random.random() < self.cfg_rate:
            assert False, "should not enter here"
            prompt_embed = self.uncond_prompt_embed
            prompt_attention_mask = self.uncond_prompt_mask
        else:
            prompt_embed = torch.load(
                os.path.join(prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            if prompt_attention_mask_file is not None:
                prompt_attention_mask = torch.load(
                    os.path.join(self.prompt_attention_mask_dir, prompt_attention_mask_file),
                    map_location="cpu",
                    weights_only=True,
                )
            else:
                prompt_attention_mask = None

            orig_len = prompt_embed.shape[0]
            if self.txt_max_len > 0:
                embed_dim = prompt_embed.shape[1]

                if orig_len < self.txt_max_len:
                    padding = torch.zeros(self.txt_max_len - orig_len, embed_dim,
                                          device=prompt_embed.device,
                                          dtype=prompt_embed.dtype)
                    prompt_embed = torch.cat([prompt_embed, padding], dim=0)
                elif orig_len > self.txt_max_len:
                    prompt_embed = prompt_embed[:self.txt_max_len]
                    orig_len = self.txt_max_len

                prompt_attention_mask = torch.zeros(self.txt_max_len, dtype=torch.long)
                prompt_attention_mask[:orig_len] = 1
            else:
                prompt_attention_mask = torch.ones(orig_len, dtype=torch.long)
        # print(latent.shape, prompt_embed.shape, prompt_attention_mask.shape)

        if self.dataset_type == 'i2v':
            y_file = self.data_anno[idx]["y_path"]
            clip_fea_file = self.data_anno[idx]["clip_feature_path"]

            y = torch.load(
                os.path.join(self.y_dir, y_file),
                map_location="cpu",
                weights_only=True,
            )

            clip_fea = torch.load(
                os.path.join(self.clip_fea_dir, clip_fea_file),
                map_location="cpu",
                weights_only=True,
            )

        if self.dataset_type == 'i2v':
            return latent, prompt_embed, prompt_attention_mask, y, clip_fea
        else:
            return latent, prompt_embed, prompt_attention_mask

    def __len__(self):
        return len(self.data_anno)


def latent_collate_function(batch):
    # return latent, prompt, latent_attn_mask, text_attn_mask
    # latent_attn_mask: # b t h w
    # text_attn_mask: b 1 l
    # needs to check if the latent/prompt' size and apply padding & attn mask

    latents, prompt_embeds, prompt_attention_masks = zip(*batch)

    # calculate max shape
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])

    # padding
    latents = [
        torch.nn.functional.pad(
            latent,
            (
                0, max_t - latent.shape[1],
                0, max_h - latent.shape[2],
                0, max_w - latent.shape[3],
            ),
        ) for latent in latents
    ]
    # attn mask
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)

    # set to 0 if padding
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0

    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)

    latents = torch.stack(latents, dim=0)

    return latents, prompt_embeds, latent_attn_mask, prompt_attention_masks

def i2v_latent_collate_function(batch):
    """
    Collate function that handles both T2V and I2V training data.
    
    T2V batch format: (latent, prompt_embed, prompt_attention_mask)
    I2V batch format: (latent, prompt_embed, prompt_attention_mask, y, clip_fea)
    """
    
    # 检查batch格式
    if len(batch[0]) == 5:  # I2V格式
        latents, prompt_embeds, prompt_attention_masks, y_tensors, clip_features = zip(*batch)
        has_i2v_data = True
    elif len(batch[0]) == 3:  # T2V格式
        latents, prompt_embeds, prompt_attention_masks = zip(*batch)
        has_i2v_data = False
        y_tensors = None
        clip_features = None
    else:
        raise ValueError(f"Unexpected batch format with {len(batch[0])} elements")
    
    # === 处理video latents (保持与原来一致) ===
    max_t = max([latent.shape[1] for latent in latents])
    max_h = max([latent.shape[2] for latent in latents])
    max_w = max([latent.shape[3] for latent in latents])
    
    # Padding video latents (和原来完全一样)
    padded_latents = [
        torch.nn.functional.pad(
            latent,
            (
                0, max_w - latent.shape[3],  # width padding
                0, max_h - latent.shape[2],  # height padding
                0, max_t - latent.shape[1],  # time padding
            ),
        ) for latent in latents
    ]
    
    # 创建video latents的attention mask (和原来完全一样)
    latent_attn_mask = torch.ones(len(latents), max_t, max_h, max_w)
    for i, latent in enumerate(latents):
        latent_attn_mask[i, latent.shape[1]:, :, :] = 0  # time dimension
        latent_attn_mask[i, :, latent.shape[2]:, :] = 0  # height dimension
        latent_attn_mask[i, :, :, latent.shape[3]:] = 0  # width dimension
    
    # === 处理text embeddings (和原来完全一样) ===
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    prompt_attention_masks = torch.stack(prompt_attention_masks, dim=0)
    
    # === 处理I2V特有数据（如果存在） ===
    if has_i2v_data:
        # 处理y tensors (reference image latents)
        max_y_t = max([y.shape[1] for y in y_tensors])
        max_y_h = max([y.shape[2] for y in y_tensors])
        max_y_w = max([y.shape[3] for y in y_tensors])
        
        padded_y_tensors = [
            torch.nn.functional.pad(
                y,
                (
                    0, max_y_w - y.shape[3],  # width
                    0, max_y_h - y.shape[2],  # height
                    0, max_y_t - y.shape[1],  # time
                ),
            ) for y in y_tensors
        ]
        
        # 创建y tensors的attention mask
        y_attn_mask = torch.ones(len(y_tensors), max_y_t, max_y_h, max_y_w)
        for i, y in enumerate(y_tensors):
            y_attn_mask[i, y.shape[1]:, :, :] = 0  # time
            y_attn_mask[i, :, y.shape[2]:, :] = 0  # height
            y_attn_mask[i, :, :, y.shape[3]:] = 0  # width
        
        # 处理CLIP features (通常是固定尺寸，直接stack)
        clip_features = torch.stack(clip_features, dim=0)
        
        # Stack所有数据
        latents_stacked = torch.stack(padded_latents, dim=0)
        y_tensors_stacked = torch.stack(padded_y_tensors, dim=0)
        
        return (
            latents_stacked,           # [B, C, T, H, W] - video latents
            prompt_embeds,             # [B, L, D] - text embeddings
            latent_attn_mask,          # [B, T, H, W] - video attention mask
            prompt_attention_masks,    # [B, L] - text attention mask
            y_tensors_stacked,         # [B, C, T, H, W] - reference image latents
            y_attn_mask,              # [B, T, H, W] - reference image attention mask
            clip_features,             # [B, D] - CLIP features
        )
    else:
        # T2V格式
        latents_stacked = torch.stack(padded_latents, dim=0)
        
        return (
            latents_stacked,           # [B, C, T, H, W] - video latents
            prompt_embeds,             # [B, L, D] - text embeddings  
            latent_attn_mask,          # [B, T, H, W] - video attention mask
            prompt_attention_masks,    # [B, L] - text attention mask
        )


if __name__ == "__main__":
    dataset = LatentDataset("data/moviidb_v0.1/preprocess/720p/videos2caption.json", num_latent_t=21, cfg_rate=0.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=latent_collate_function)
    for latent, prompt_embed, latent_attn_mask, prompt_attention_mask in dataloader:
        print(
            latent.shape,
            prompt_embed.shape,
            latent_attn_mask.shape,
            prompt_attention_mask.shape,
        )
        import pdb

        pdb.set_trace()
