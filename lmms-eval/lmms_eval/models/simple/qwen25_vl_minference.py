"""
Qwen2.5-VL with MInference optimization for accelerated long-context inference.

This model integrates MInference dynamic sparse attention optimization with Qwen2.5-VL
to achieve significant speedup for long-context multimodal reasoning tasks.

Example usage:
    python -m lmms_eval --model qwen25_vl_minference \\
                        --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,attn_type=minference,kv_type=snapkv \\
                        --tasks mmmu \\
                        --batch_size 1
"""

import warnings
import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import decord
import numpy as np
from loguru import logger as eval_logger
from PIL import Image

import torch.nn as nn
from transformers import AutoTokenizer,Qwen2_5_VLForConditionalGeneration
from accelerate import Accelerator

from ..qwen25_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from ..qwen25_vl.vision_process import process_vision_info

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

# MInference imports
try:
    from minference import MInference
    MINFERENCE_AVAILABLE = True
except ImportError:
    MINFERENCE_AVAILABLE = False
    warnings.warn("MInference not available. Install with: pip install minference")


@register_model("qwen25_vl_minference") 
class Qwen2_5_VLMInference(lmms):
    """
    Qwen2.5-VL Model with MInference optimization for long-context inference acceleration.
    
    This model leverages MInference dynamic sparse attention to achieve up to 10x speedup
    for long-context multimodal tasks while maintaining accuracy.
    
    Supported MInference configurations:
    - attn_type: "minference", "dense", "a_shape", "tri_shape", etc.
    - kv_type: "dense", "snapkv", "pyramidkv", "quest", "streamingllm", etc.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto", 
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        # MInference specific parameters
        attn_type: str = "minference",
        kv_type: str = "dense",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        # Vision parameters 
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        
        # Store MInference configuration
        self.attn_type = attn_type
        self.kv_type = kv_type 
        self.attn_kwargs = attn_kwargs or {}
        self._minference_applied = False
        
        # Store other parameters
        self.pretrained = pretrained
        # Accelerator-aware device placement
        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_map = device_map if device_map else (device if device else "cuda")

        self.batch_size = int(batch_size)
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation
        
        # Vision parameters
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_num_frames = max_num_frames
        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        eval_logger.info(f"Loading Qwen2.5-VL model with MInference optimization...")
        eval_logger.info(f"  Model: {pretrained}")
        eval_logger.info(f"  MInference config: attn_type={attn_type}, kv_type={kv_type}")
        
        # Load model with optional attention implementation
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": self.device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()

        # Enable prefill runtime tracking if model supports it
        if hasattr(self._model, 'enable_prefill_runtime_tracking'):
            self._model.enable_prefill_runtime_tracking()

        for p in self._model.parameters():
            p.requires_grad = False
        
        # Load tokenizer and processor
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, trust_remote_code=True)
        
        # Apply MInference optimization
        self._apply_minference()
        
        # Set model to evaluation mode
        self._model.eval()
        if accelerator.num_processes > 1:
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            eval_logger.info(f"Accelerate multi-process: local_rank={self._rank} world_size={self._world_size}")
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"Model loaded on device: {self._device}")
        
    def _apply_minference(self):
        """Apply MInference optimization to the language model component."""
        if not MINFERENCE_AVAILABLE:
            eval_logger.warning("MInference not available. Using standard model.")
            return
            
        if self.attn_type == "dense":
            eval_logger.info("Using dense attention (no MInference optimization)")
            return
            
        try:
            # Get base model name for MInference configuration
            base_model_name = self._get_base_model_name(self.pretrained)
            
            eval_logger.info(f"Applying MInference with attn_type='{self.attn_type}', kv_type='{self.kv_type}'")
            eval_logger.info(f"Base model name for MInference: {base_model_name}")
            
            # Create MInference patch (wrapper) and apply only to LLM layers
            minference_patch = MInference(
                attn_type=self.attn_type,
                model_name=base_model_name,
                kv_type=self.kv_type,
                attn_kwargs=self.attn_kwargs,
            )

            class _Inner(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    self.layers = nn.ModuleList(list(layers))

            class _LMContainer(nn.Module):
                def __init__(self, gen_model):
                    super().__init__()
                    self._gen = gen_model
                    self.config = gen_model.config
                    if hasattr(gen_model, "model") and hasattr(gen_model.model, "language_model"):
                        layers = getattr(gen_model.model.language_model, "layers", None)
                    else:
                        layers = None
                    if layers is None:
                        raise AttributeError("language_model.layers not found on generation model")
                    self.model = _Inner(layers)

                def prepare_inputs_for_generation(self, *args, **kwargs):
                    return self._gen.prepare_inputs_for_generation(*args, **kwargs)

                def apply(self, fn):
                    fn(self)
                    for layer in self.model.layers:
                        fn(layer)
                    return self

            container = _LMContainer(self._model)
            patched = minference_patch(container)
            if patched is not None and hasattr(patched, "model") and hasattr(patched.model, "layers"):
                if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
                    self._model.model.language_model.layers = patched.model.layers
                elif hasattr(self._model, "language_model"):
                    self._model.language_model.layers = patched.model.layers
                self._minference_applied = True
                eval_logger.info("✅ MInference applied to language_model via wrapper")
            else:
                eval_logger.warning("MInference wrapper returned None; continuing without optimization")
                self._minference_applied = False
            
        except Exception as e:
            eval_logger.error(f"Failed to apply MInference: {e}")
            eval_logger.warning("Falling back to standard model")
            self._minference_applied = False
    
    def _get_base_model_name(self, model_path: str) -> str:
        """Map Qwen2.5-VL model names to their text-only counterparts for MInference."""
        mappings = {
            "Qwen/Qwen2.5-VL-1.5B": "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-VL-3B": "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-VL-7B": "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-VL-32B": "Qwen/Qwen2.5-32B", 
            "Qwen/Qwen2.5-VL-72B": "Qwen/Qwen2.5-72B",
            "Qwen/Qwen2.5-VL-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct", 
            "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
        }
        
        if model_path in mappings:
            return mappings[model_path]
        elif "Qwen2.5-VL" in model_path:
            return model_path.replace("Qwen2.5-VL", "Qwen2.5")
        else:
            eval_logger.warning(f"Unknown model path: {model_path}, using as-is")
            return model_path

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.config.max_position_embeddings

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if batch_first:
            input_ids = [seq.clone().detach() for seq in input_ids]
            max_length = max([seq.size(0) for seq in input_ids])
            padded_sequences = []
            for seq in input_ids:
                padded_seq = torch.full((max_length,), padding_value, dtype=seq.dtype, device=seq.device)
                padded_seq[: seq.size(0)] = seq
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences, dim=0)
        else:
            input_ids = [seq.clone().detach() for seq in input_ids]
            max_length = max([seq.size(0) for seq in input_ids])
            padded_sequences = []
            for seq in input_ids:
                padded_seq = torch.full((max_length,), padding_value, dtype=seq.dtype, device=seq.device)
                padded_seq[: seq.size(0)] = seq
                padded_sequences.append(padded_seq.unsqueeze(0))
            return torch.cat(padded_sequences, dim=0)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for a list of requests using MInference-optimized model."""
        res = []
        
        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return len(toks), x[0]

        from tqdm import tqdm
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (MInference)")
        
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Build chat messages and process multimodal inputs
            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels, "fps": self.fps, "max_frames": self.max_num_frames})
                        elif isinstance(visual, Image.Image):
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    message.append({
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    })
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for j, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if j + 1 < len(text_parts) and text_parts[j + 1]:
                            content_parts.append({"type": "text", "text": text_parts[j + 1]})

                    message.append({
                        "role": "user",
                        "content": content_parts,
                    })

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)

            processor_kwargs: Dict[str, Any] = {"padding": True, "return_tensors": "pt"}
            videos_kwargs: Dict[str, Any] = {}
            if self.fps is not None:
                videos_kwargs["fps"] = float(self.fps)
            if videos_kwargs:
                processor_kwargs["videos_kwargs"] = videos_kwargs

            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, **processor_kwargs)
            inputs = inputs.to(self._device)

            # Generation
            batch_responses = []
            default_gen_kwargs = {"max_new_tokens": 32768, "temperature": 0.0, "top_p": None, "num_beams": 1}
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            # Reset prefill runtime tracking before generation
            if hasattr(self.model, 'reset_prefill_runtime'):
                self.model.reset_prefill_runtime()

            try:
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                # Report prefill runtime after generation
                if hasattr(self.model, 'get_prefill_runtime'):
                    prefill_runtime = self.model.get_prefill_runtime()
                    if prefill_runtime is not None:
                        eval_logger.info(f"MInference Prefill Runtime: {prefill_runtime:.6f} seconds")
                        print(f"MINFERENCE_PREFILL_RUNTIME: {prefill_runtime:.6f} seconds")
                        import sys
                        sys.stdout.flush()
                        print(f"MINFERENCE_PREFILL_RUNTIME: {prefill_runtime:.6f} seconds", file=sys.stderr)
                        sys.stderr.flush()

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                batch_responses.extend(answers)
            except Exception as e:
                eval_logger.error(f"Generation error: {e}")
                batch_responses.extend([""] * len(texts))
            
            # Apply until conditions and reasoning parsing
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")
            until = [item for item in until if item != "\n\n"]

            for i, ans in enumerate(batch_responses):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                batch_responses[i] = parse_reasoning_model_answer(ans)

            res.extend(batch_responses)
            pbar.update(len(contexts))
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Multi-round chat is currently not implemented for this adapter."""
        raise NotImplementedError("generate_until_multi_round is not implemented for qwen25_vl_minference")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for a list of requests."""
        # This is a simplified implementation - you may need to adapt based on your needs
        eval_logger.warning("loglikelihood not fully implemented for qwen25_vl_minference")
        return [(0.0, False) for _ in requests]

    def get_minference_status(self) -> Dict[str, Any]:
        """Get MInference optimization status."""
        return {
            "minference_available": MINFERENCE_AVAILABLE,
            "minference_applied": self._minference_applied,
            "attn_type": self.attn_type,
            "kv_type": self.kv_type,
            "attn_kwargs": self.attn_kwargs,
        }
