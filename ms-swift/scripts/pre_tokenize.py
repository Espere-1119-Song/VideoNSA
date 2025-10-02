from swift.llm import load_dataset, get_template, get_model_tokenizer  
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
  
num_proc = 4 
split_dataset_ratio = 0.001

model, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-VL-7B-Instruct')
template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
template.set_mode('train')


# 2. 加载您的 JSONL 数据集  
train_dataset, val_dataset = load_dataset("datasets/jsonl/filter_llava/2_3_m_nextqa.jsonl", split_dataset_ratio=split_dataset_ratio)  
  
train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

train_dataset.save_to_disk("datasets/tokenized/filter_llava/2_3_m_nextqa")
val_dataset.save_to_disk("datasets/tokenized/filter_llava/2_3_m_nextqa_val")