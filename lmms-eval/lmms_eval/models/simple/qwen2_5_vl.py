import base64
import re
import time
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
# from transformers import (
#     AutoProcessor,
#     AutoTokenizer,
# )

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

# from ..qwen25_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval()
        self._model = self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

        # Initialize runtime tracking variables - default enabled for testing
        self._track_prefill_runtime = True
        self._prefill_runtime = None

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        with torch.no_grad():
            res = []
            # import pdb; pdb.set_trace()

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch
                #   padded context length. this is useful to simplify the batching logic and more importantly to make
                #   automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end
                toks = self.tokenizer.encode(x[0])
                return -len(toks), x[0]

            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            # we group requests by their generation_kwargs,
            # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
            # in the same batch.
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
            chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
            for chunk in chunks:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                task = task[0]
                split = split[0]
                visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                gen_kwargs = all_gen_kwargs[0]

                # Set default until or update values from gen_kwargs if present
                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

                # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
                until = [item for item in until if item != "\n\n"]

                if isinstance(contexts, tuple):
                    contexts = list(contexts)

                for i in range(len(contexts)):
                    if "<image>" in contexts[i]:
                        contexts[i] = contexts[i].replace("<image>", "")

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
                            if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                                vr = decord.VideoReader(visual)
                                first_frame = vr[0].asnumpy()
                                height, width = first_frame.shape[:2]
                                # max_pixels = height * width
                                video_config = {"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
                                # Add max_frames for direct sampling if fps is not set
                                if self.fps is None:
                                    video_config["max_frames"] = self.max_num_frames
                                processed_visuals.append(video_config)
                            elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                                base64_image = visual.convert("RGB")
                                buffer = BytesIO()
                                base64_image.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                    if self.interleave_visuals is False:
                        message.append(
                            {
                                "role": "user",
                                "content": processed_visuals + [{"type": "text", "text": context}],
                            }
                        )
                    else:  # currently support find <image x> in the context
                        image_placeholders = re.findall(r"<image \d+>", context)
                        content_parts = []
                        text_parts = re.split(r"<image \d+>", context)
                        if text_parts[0]:
                            content_parts.append({"type": "text", "text": text_parts[0]})

                        for i, placeholder in enumerate(image_placeholders):
                            img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                            image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                            if processed_visuals and image_idx < len(processed_visuals):
                                content_parts.append(processed_visuals[image_idx])
                            if i + 1 < len(text_parts) and text_parts[i + 1]:
                                content_parts.append({"type": "text", "text": text_parts[i + 1]})

                        message.append(
                            {
                                "role": "user",
                                "content": content_parts,
                            }
                        )

                    batched_messages.append(message)

                texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
                image_inputs, video_inputs = process_vision_info(batched_messages)
                # If videos exist and fps is specified, do fps-based resampling
                if video_inputs is not None and self.fps is not None:
                    # Extract original video file paths from messages for fps-aware sampling
                    video_paths_per_sample = []
                    for msg in batched_messages:
                        paths_for_this_sample = []
                        for content in msg[1]["content"]:
                            if isinstance(content, dict) and content.get("type") == "video":
                                # The processor expects a local path/URI string here
                                if isinstance(content.get("video"), str):
                                    paths_for_this_sample.append(content["video"])
                        video_paths_per_sample.append(paths_for_this_sample)

                    # Iterate each sample and build frame indices
                    for sample_index in range(len(video_inputs)):
                        if video_inputs[sample_index] is None:
                            continue

                        total_frames = int(video_inputs[sample_index].shape[0])
                        if total_frames <= 0:
                            continue

                        if len(video_paths_per_sample[sample_index]) > 0:
                            try:
                                # Use the first video in this sample (common case is 1 video per sample)
                                vr = decord.VideoReader(video_paths_per_sample[sample_index][0])
                                orig_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else 0.0
                            except Exception:
                                # Fallback if we fail to open with decord
                                orig_fps = 0.0

                            if orig_fps > 0.0 and self.fps > 0.0:
                                # Take roughly 1 frame per second: step ~= orig_fps / desired_fps
                                step = max(1, int(round(orig_fps / float(self.fps))))
                                indices = np.arange(0, total_frames, step, dtype=int)
                            else:
                                # If fps unavailable, fall back to uniform sampling by max_num_frames
                                desired = max(1, int(self.max_num_frames))
                                indices = np.linspace(0, total_frames - 1, desired, dtype=int)

                            # Cap the number of frames to avoid OOM
                            if len(indices) > int(self.max_num_frames):
                                indices = indices[: int(self.max_num_frames)]

                            # Ensure at least one frame and include last frame if missing
                            indices = np.unique(indices)
                            if indices.size == 0:
                                indices = np.array([0], dtype=int)
                            if total_frames - 1 not in indices:
                                indices = np.append(indices, total_frames - 1)
                                indices = np.unique(indices)

                            video_inputs[sample_index] = video_inputs[sample_index][indices]
                # Note: If fps is None, video sampling is already handled by max_frames in vision_process.py
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                print("=== PROCESSOR DEBUG ===")
                print(f"texts content: {texts[:1] if texts else 'None'}")  # 看第一个text
                print(f"input_ids type: {type(inputs.input_ids)}")
                print(f"input_ids dtype: {inputs.input_ids.dtype}")
                print(f"input_ids shape: {inputs.input_ids.shape}")
                print(f"input_ids sample values: {inputs.input_ids.flatten()[:10]}")
                # import pdb; pdb.set_trace()
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # Set default generation kwargs
                default_gen_kwargs = {
                    "max_new_tokens": 32768,
                    "temperature": 0.0,  # Set to 0 for greedy default
                    "top_p": None,
                    "num_beams": 1,
                }
                # Update with provided kwargs
                current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
                pad_token_id = self.tokenizer.pad_token_id

                if current_gen_kwargs["temperature"] > 0:
                    current_gen_kwargs["do_sample"] = True
                else:
                    current_gen_kwargs["do_sample"] = False
                    current_gen_kwargs["temperature"] = None
                    current_gen_kwargs["top_p"] = None

                # Track prefill runtime for standard Qwen2.5-VL
                prefill_start_time = None
                if self._track_prefill_runtime:
                    prefill_start_time = time.time()
                    torch.cuda.synchronize() if torch.cuda.is_available() else None

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

                # Record prefill runtime
                if prefill_start_time is not None:
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    prefill_end_time = time.time()
                    prefill_runtime = prefill_end_time - prefill_start_time
                    self._prefill_runtime = prefill_runtime
                    # Output runtime information with attention implementation info
                    attn_impl = getattr(self.model.config, 'attn_implementation', 'unknown')
                    print(f"QWEN25VL_TOTAL_RUNTIME ({attn_impl}): {prefill_runtime:.6f} seconds")
                    import sys
                    sys.stdout.flush()
                    print(f"QWEN25VL_TOTAL_RUNTIME ({attn_impl}): {prefill_runtime:.6f} seconds", file=sys.stderr)
                    sys.stderr.flush()
                    eval_logger.info(f"Qwen2.5-VL Total Runtime ({attn_impl}): {prefill_runtime:.6f} seconds")

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    clean_ans = parse_reasoning_model_answer(ans)
                    res.append(clean_ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                    pbar.update(1)

                    # eval_logger.debug(f"Question: {context}")
                    # eval_logger.debug(f"Model Raw Response: {ans}")
                    # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
            res = re_ords.get_original(res)

            pbar.close()
            return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def enable_prefill_runtime_tracking(self):
        """Enable tracking of prefill runtime"""
        self._track_prefill_runtime = True

    def disable_prefill_runtime_tracking(self):
        """Disable tracking of prefill runtime"""
        self._track_prefill_runtime = False

    def get_prefill_runtime(self):
        """Get the prefill runtime in seconds"""
        return self._prefill_runtime

    def reset_prefill_runtime(self):
        """Reset prefill runtime tracking"""
        self._prefill_runtime = None
