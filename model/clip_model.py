from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hf_env import resolve_hf_repo_path


def get_clip_base(model):
    return model


def encode_image_features(clip_model, pixel_values):
    out = clip_model.vision_model(pixel_values=pixel_values)
    feat = clip_model.visual_projection(out.pooler_output)
    return F.normalize(feat, dim=-1)


def encode_text_features(clip_model, input_ids, attention_mask):
    out = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    feat = clip_model.text_projection(out.pooler_output)
    return F.normalize(feat, dim=-1)


class FixedTextEncoder(nn.Module):
    def __init__(self, clip_model, input_ids, attention_mask):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('input_ids', input_ids, persistent=False)
        self.register_buffer('attention_mask', attention_mask, persistent=False)

    def forward(self):
        return encode_text_features(self.clip_model, self.input_ids, self.attention_mask)


def build_dummy_text_inputs(num_classes: int, max_text_len: int, bos_token_id: int = 0, eos_token_id: int = 2):
    ids = torch.zeros((num_classes, max_text_len), dtype=torch.long)
    mask = torch.zeros((num_classes, max_text_len), dtype=torch.long)
    for c in range(num_classes):
        ids[c, 0] = bos_token_id
        core_len = min(4, max_text_len - 2)
        for i in range(core_len):
            ids[c, 1 + i] = 10 + ((c + i) % 200)
        end_pos = 1 + core_len
        if end_pos < max_text_len:
            ids[c, end_pos] = eos_token_id
            mask[c, : end_pos + 1] = 1
        else:
            mask[c, :] = 1
    return ids, mask


def build_text_tokens(clip_model_name: str, class_names: List[str], template: str, max_text_len: int, random_init: bool):
    if random_init:
        return build_dummy_text_inputs(num_classes=len(class_names), max_text_len=max_text_len)
    from transformers import AutoTokenizer
    source = resolve_hf_repo_path(clip_model_name)
    tok = AutoTokenizer.from_pretrained(
        source,
        local_files_only=True,
    )
    prompts = [template.format(name) for name in class_names]
    out = tok(prompts, padding='max_length', truncation=True, max_length=max_text_len, return_tensors='pt')
    return out['input_ids'], out['attention_mask']


def build_clip_model(clip_model_name: str, class_names: List[str], text_template: str, max_text_len: int, random_init: bool, device):
    from transformers import CLIPConfig, CLIPModel

    if random_init:
        cfg = CLIPConfig(
            projection_dim=256,
            text_config={
                'hidden_size': 256,
                'intermediate_size': 512,
                'num_hidden_layers': 4,
                'num_attention_heads': 4,
                'max_position_embeddings': max(64, max_text_len),
                'vocab_size': 1000,
            },
            vision_config={
                'hidden_size': 256,
                'intermediate_size': 512,
                'num_hidden_layers': 4,
                'num_attention_heads': 4,
                'image_size': 224,
                'patch_size': 16,
            },
        )
        clip = CLIPModel(cfg)
    else:
        source = resolve_hf_repo_path(clip_model_name)
        clip = CLIPModel.from_pretrained(
            source,
            local_files_only=True,
        )

    model = clip.to(device)
    input_ids, attention_mask = build_text_tokens(
        clip_model_name=clip_model_name,
        class_names=class_names,
        template=text_template,
        max_text_len=max_text_len,
        random_init=random_init,
    )
    text_encoder = FixedTextEncoder(model, input_ids=input_ids, attention_mask=attention_mask).to(device)
    return model, text_encoder
