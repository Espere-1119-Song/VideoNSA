"""
Qwen2.5-VL with xAttention optimization for accelerated long-context inference.

xAttention reduces attention computation complexity through efficient attention patterns
while maintaining model performance.

Example usage:
    python -m lmms_eval --model qwen25_vl_xattention \\
                        --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \\
                        --tasks mmmu \\
                        --batch_size 1
"""

import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from loguru import logger as eval_logger
from PIL import Image

from transformers import AutoTokenizer,  Qwen2_5_VLForConditionalGeneration
from accelerate import Accelerator, DistributedType
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # bitsandbytes will only be needed if quantization is requested

from ..qwen25_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from ..qwen25_vl.vision_process import process_vision_info

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# MInference imports for xAttention
try:
    from minference import MInference
    MINFERENCE_AVAILABLE = True
except ImportError:
    MINFERENCE_AVAILABLE = False
    warnings.warn("MInference not available. Install with: pip install minference")


@register_model("qwen25_vl_xattention")
class Qwen2_5_VLxAttention(lmms):
    """
    Qwen2.5-VL Model with xAttention optimization for long-context inference acceleration.
    
    xAttention is an efficient attention mechanism that reduces computational complexity
    while maintaining high accuracy for long-context scenarios.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto", 
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        # Optional quantization to reduce VRAM
        load_in_4bit: Optional[bool] = False,
        load_in_8bit: Optional[bool] = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
        # xAttention specific parameters
        attn_kwargs: Optional[Dict[str, Any]] = None,
        xattn_stride: Optional[int] = None,
        xattn_threshold: Optional[float] = None,
        # Vision parameters 
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        # Optional CPU/NVMe offload to reduce peak VRAM during load
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        # Store xAttention configuration
        # xAttention variant (as requested)
        self.attn_type = "xattention"
        self.kv_type = "dense"  # xAttention typically uses dense KV
        self.attn_kwargs = attn_kwargs or {}
        # Bake-in sensible defaults for xAttention
        self.attn_kwargs.setdefault("stride", 16)
        self.attn_kwargs.setdefault("threshold", 0.9)
        # Optionally allow override from CLI if provided
        if xattn_stride is not None:
            try:
                self.attn_kwargs["stride"] = int(xattn_stride)
            except Exception:
                eval_logger.warning(f"Invalid xattn_stride={xattn_stride}, expected int. Keeping default {self.attn_kwargs['stride']}.")
        if xattn_threshold is not None:
            try:
                self.attn_kwargs["threshold"] = float(xattn_threshold)
            except Exception:
                eval_logger.warning(f"Invalid xattn_threshold={xattn_threshold}, expected float. Keeping default {self.attn_kwargs['threshold']}.")
        self._optimization_applied = False
        
        # Store other parameters
        self.pretrained = pretrained
        # Accelerator-aware device assignment
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
        
        eval_logger.info(f"Loading Qwen2.5-VL model with xAttention optimization...")
        eval_logger.info(f"  Model: {pretrained}")

        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": self.device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        # Do NOT force 'minference' at load time; HF core rejects unknown implementations.
        if offload_folder:
            model_kwargs["offload_folder"] = offload_folder
            model_kwargs["offload_state_dict"] = offload_state_dict
        # Apply optional quantization if requested
        if load_in_4bit:
            if BitsAndBytesConfig is None:
                eval_logger.warning("Requested load_in_4bit=True but BitsAndBytesConfig/bitsandbytes not available. Proceeding without 4-bit quantization.")
            else:
                compute_dtype = torch.bfloat16 if bnb_4bit_compute_dtype.lower() == "bfloat16" else torch.float16
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                model_kwargs["quantization_config"] = bnb_config
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            
        # Qwen2.5-VL is a multimodal model; use its dedicated class
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        for p in self._model.parameters():
            p.requires_grad = False
        
        # Load tokenizer and processor
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, trust_remote_code=True)
        
        # Apply xAttention optimization
        self._apply_xattention()
        
        # Set model to evaluation mode
        self._model.eval()

        if accelerator.num_processes > 1:
            # We don't wrap with DDP for now; each process owns one GPU replica
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            eval_logger.info(f"Accelerate multi-process: local_rank={self._rank} world_size={self._world_size}")
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"Model loaded on device: {self._device}")

    def _apply_xattention(self):
        """Apply xAttention via MInference wrapper on the loaded model."""
        if not MINFERENCE_AVAILABLE:
            eval_logger.warning("MInference not available. Using standard model.")
            return

        try:
            base_model_name = self._get_base_model_name(self.pretrained)
            eval_logger.info("Applying xAttention optimization (MInference wrapper)")
            if self.attn_kwargs:
                eval_logger.info(f"xAttention kwargs: {self.attn_kwargs}")
            eval_logger.info(f"Base model name: {base_model_name}")

            # Prefer patching only the LLM (language_model) part
            minference_patch = MInference(attn_type=self.attn_type, model_name=base_model_name, kv_type=self.kv_type, attn_kwargs=self.attn_kwargs)
            # Patch only the LLM part by providing a tiny container with a `.model` attribute,
            # since MInference expects container.model.layers
            if not hasattr(self._model, "language_model") or self._model.language_model is None:
                eval_logger.warning("language_model not found on Qwen2.5-VL; skipping xAttention patch")
                self._optimization_applied = False
                return

            class _Inner(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    # Register layers so visitors can traverse
                    self.layers = nn.ModuleList(list(layers))

            class _LMContainer(nn.Module):
                def __init__(self, gen_model):
                    super().__init__()
                    # Expose a minimal structure expected by MInference wrapper:
                    # - .config
                    # - .model.layers (list/ModuleList of layers to patch)
                    self._gen = gen_model
                    self.config = gen_model.config
                    # Point to the LLM layers under the multimodal wrapper
                    if hasattr(gen_model, "model") and hasattr(gen_model.model, "language_model"):
                        layers = getattr(gen_model.model.language_model, "layers", None)
                    else:
                        layers = None
                    if layers is None:
                        raise AttributeError("language_model.layers not found on generation model")
                    self.model = _Inner(layers)

                def prepare_inputs_for_generation(self, *args, **kwargs):
                    # Forward to the underlying generation model implementation
                    return self._gen.prepare_inputs_for_generation(*args, **kwargs)

                def apply(self, fn):
                    # Apply fn to self and each layer; mimic nn.Module.apply semantics minimally
                    fn(self)
                    for layer in self.model.layers:
                        fn(layer)
                    return self

            container = _LMContainer(self._model)
            patched_container = minference_patch(container)
            if patched_container is not None and hasattr(patched_container, "model"):
                # Update the original LLM layers in place if replaced
                if hasattr(patched_container.model, "layers"):
                    if hasattr(self._model, "model") and hasattr(self._model.model, "language_model"):
                        self._model.model.language_model.layers = patched_container.model.layers
                    elif hasattr(self._model, "language_model"):
                        self._model.language_model.layers = patched_container.model.layers
                self._optimization_applied = True
                eval_logger.info("✅ xAttention (MInference) applied to language_model via wrapper (container.model)")
            else:
                eval_logger.warning("MInference wrapper returned None or missing .model; continuing without xAttention")
                self._optimization_applied = False

        except Exception as e:
            eval_logger.error(f"Failed to apply xAttention: {e}")
            eval_logger.warning("Falling back to standard model")
            self._optimization_applied = False
    
    def _get_base_model_name(self, model_path: str) -> str:
        """Map Qwen2.5-VL model names to their text-only counterparts."""
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
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
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
        """Generate responses using batching/collation like qwen2_5_vl."""
        with torch.no_grad():
            res: List[str] = []

            def _collate(x):
                toks = self.tokenizer.encode(x[0])
                return -len(toks), x[0]

            from tqdm import tqdm
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
            chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
            for chunk in chunks:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                task = task[0]
                split = split[0]
                visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                gen_kwargs = all_gen_kwargs[0]

                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], got {type(until)}")
                until = [item for item in until if item != "\n\n"]

                if isinstance(contexts, tuple):
                    contexts = list(contexts)
                for i in range(len(contexts)):
                    if "<image>" in contexts[i]:
                        contexts[i] = contexts[i].replace("<image>", "")

                batched_messages = []
                for i, context in enumerate(contexts):
                    message = []
                    if self.system_prompt:
                        message.append({"role": "system", "content": self.system_prompt})

                    processed_visuals = []
                    if visual_list[i] is not None:
                        for visual in visual_list[i]:
                            if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                    }
                                )
                            elif isinstance(visual, Image.Image):
                                processed_visuals.append(
                                    {
                                        "type": "image",
                                        "image": visual.convert("RGB"),
                                        "max_pixels": self.max_pixels,
                                        "min_pixels": self.min_pixels,
                                    }
                                )

                    message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": context}]})
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

                default_gen_kwargs = {
                    "max_new_tokens": 1024,
                    "temperature": 0.0,
                    "top_p": None,
                    "num_beams": 1,
                }
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if current_gen_kwargs["temperature"] and current_gen_kwargs["temperature"] > 0 else False,
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

            res = re_ords.get_original(res)
            pbar.close()
            return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for a list of requests."""
        eval_logger.warning("loglikelihood not fully implemented for qwen25_vl_xattention")
        return [(0.0, False) for _ in requests]

    def generate_until_multi_round(self, requests) -> List[str]:
        """Placeholder for multi-round generation.

        This model currently supports single-round generation via `generate_until`.
        Multi-round conversations are not implemented yet to keep parity with
        other simple adapters. If a task requires multi-round behavior, this
        should be extended to build a chat history and call generation per turn.
        """
        raise NotImplementedError("generate_until_multi_round is not implemented for qwen25_vl_xattention")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get xAttention optimization status."""
        return {
            "optimization_type": "xAttention",
            "minference_available": MINFERENCE_AVAILABLE,
            "optimization_applied": self._optimization_applied,
            "attn_kwargs": self.attn_kwargs,
        }
