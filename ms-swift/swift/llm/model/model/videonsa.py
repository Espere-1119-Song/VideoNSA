# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Tuple, Type

import torch
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_device_count, get_dist_setting, get_env_args, get_logger
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..patcher import patch_fixed_device, patch_get_input_embeddings, patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, get_model_tokenizer_reward_model,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl, ModelInfo, use_submodel_func

logger = get_logger()
dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}


def fix_qwen_inplace_bug(model) -> None:
    # qwen-vl, qwen-audio
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix in-place operation bug
        patch_output_clone(first_drop)

def _qwen_vl_visual_block_forward(
    self,
    q_x: torch.Tensor,
    k_x: Optional[torch.Tensor] = None,
    v_x: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

    x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
    z = self.mlp(self.ln_2(x))
    x = x.to(z.device) + z  # FIX
    return x

def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return

    if os.getenv('VIDEO_MAX_PIXELS') and not os.getenv('VIDEO_TOTAL_PIXELS'):
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1120
        os.environ['VIDEO_TOTAL_PIXELS'] = str(int(128000 * 28 * 28 * 0.9))
    for key in [
            'image_factor', 'min_pixels', 'max_pixels', 'max_ratio', 'video_min_pixels', 'video_max_pixels',
            'video_total_pixels', 'frame_factor', 'fps', 'fps_min_frames', 'fps_max_frames'
    ]:
        type_func = float if key == 'fps' else int
        setattr(vision_process, key.upper(), get_env_args(key, type_func, getattr(vision_process, key.upper())))
    _read_video_decord = vision_process._read_video_decord

    def _new_read_video_decord(ele: dict):
        from swift.llm import load_file
        ele['video'] = load_file(ele['video'])
        return _read_video_decord(ele)

    vision_process.VIDEO_READER_BACKENDS['decord'] = _new_read_video_decord
    vision_process._patch = True

def get_model_tokenizer_qwen2_vl(*args, **kwargs):
    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        if hasattr(base_model.model, 'embed_tokens'):
            embed_tokens = base_model.model.embed_tokens
        else:
            embed_tokens = base_model.model.language_model.embed_tokens
        patch_output_clone(embed_tokens)
        patch_output_to_input_device(embed_tokens)
        patch_get_input_embeddings(base_model.visual, 'patch_embed')

    from qwen_vl_utils import vision_process
    patch_qwen_vl_utils(vision_process)
    return model, tokenizer

def get_model_tokenizer_videonsa(*args, **kwargs):
    from .qwen25_vl.modeling_qwen2_5_vl import VideoNSAForConditionalGeneration
    kwargs['automodel_class'] = VideoNSAForConditionalGeneration
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.videonsa, [
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-7B-Instruct'),
            ]),
        ],
        TemplateType.qwen2_5_vl,
        get_model_tokenizer_videonsa,
        model_arch=ModelArch.videonsa,
        architectures=['VideoNSAForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video']))
