# from transformers import AutoModel, AutoTokenizer, AutoProcessor
from keye_vl_utils import process_vision_info

from models.keye.modeling_keye import KeyeForConditionalGeneration
from models.keye.processing_keye import KeyeProcessor

# default: Load the model on the available device(s)
model_path = "Kwai-Keye/Keye-VL-8B-Preview"
# model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True) # KeyeConfig
import pdb;pdb.set_trace()

model = KeyeForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

processor = KeyeProcessor.from_pretrained(model_path, trust_remote_code=True)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg",
            },
            {"type": "text", "text": "Describe this image./no_think"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
