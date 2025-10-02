# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ms-swift is the official ModelScope framework for fine-tuning and deploying large language models (LLMs) and multi-modal large models. It supports 500+ pure text models and 200+ multi-modal models, with training (pre-training, fine-tuning, human alignment), inference, evaluation, quantization, and deployment capabilities.

## Common Commands

### Build and Development
```bash
# Build documentation
make docs

# Run linter (Note: linter.sh script may not exist - use standard Python linters)
make linter

# Run tests (Note: citest.sh script may not exist - use pytest directly)
make test

# Build wheel package
make whl

# Clean build artifacts
make clean
```

### Direct Development Commands
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install specific requirement sets
pip install -r requirements/eval.txt    # Evaluation dependencies
pip install -r requirements/tests.txt   # Testing dependencies
pip install -r requirements/docs.txt    # Documentation dependencies

# Run tests directly (if make test doesn't work)
pytest tests/
python -m pytest tests/specific_test.py

# Manual linting
flake8 swift/
black swift/
isort swift/
```

### Core CLI Commands
All commands use the `swift` CLI entry point:

```bash
# Training
swift sft --model <model> --dataset <dataset>           # Supervised fine-tuning
swift pt --model <model> --dataset <dataset>            # Pre-training
swift rlhf --model <model> --dataset <dataset>          # RLHF training (DPO/PPO/etc)

# Inference and Deployment
swift infer --model <model>                              # Interactive inference
swift deploy --model <model>                            # Deploy model server
swift eval --model <model> --dataset <dataset>          # Model evaluation

# Export and Utilities
swift export --model <model>                            # Export/quantize models
swift merge-lora --model <model>                        # Merge LoRA weights
swift web-ui                                            # Launch Gradio web interface
```

### Training Examples
```bash
# Multi-GPU training with DeepSpeed
export CUDA_VISIBLE_DEVICES=0,1,2,3
swift sft --model Qwen/Qwen2.5-VL-7B-Instruct --deepspeed zero2 --train_type full

# LoRA fine-tuning
swift sft --model llama --dataset alpaca --train_type lora

# Multi-modal training
swift sft --model qwen2-vl --dataset coco-caption --train_type lora
```

## Architecture Overview

### Main Package Structure
- **`swift/`** - Core package containing all modules
  - **`llm/`** - LLM-specific functionality (training, inference, evaluation)
  - **`tuners/`** - Parameter-efficient fine-tuning methods (LoRA, Adapter, etc.)
  - **`trainers/`** - Training algorithms and utilities  
  - **`ui/`** - Gradio-based web interfaces
  - **`utils/`** - Common utilities and helpers
  - **`cli/`** - Command-line interface implementations

### Key Components

#### LLM Module (`swift/llm/`)
- **`model/`** - Model architectures and patching (500+ models supported)
- **`template/`** - Chat templates and conversation formatting
- **`dataset/`** - Dataset loading, preprocessing, and management
- **`train/`** - Training logic for SFT, PT, RLHF
- **`infer/`** - Inference engines (PyTorch, vLLM, SGLang, LMDeploy)
- **`eval/`** - Model evaluation using EvalScope backend
- **`export/`** - Model quantization and export (GPTQ, AWQ, BNB)

#### Training Framework
- **Distributed Training**: DDP, DeepSpeed ZeRO2/ZeRO3, FSDP, Megatron parallelism
- **Parameter-Efficient Methods**: LoRA, QLoRA, DoRA, LoRA+, ReFT, Adapter, etc.
- **Human Alignment**: DPO, GRPO, PPO, KTO, CPO, SimPO, ORPO
- **Quantization**: BNB, AWQ, GPTQ, AQLM, HQQ training support

#### Model Support
- **Text Models**: Qwen, InternLM, GLM, Mistral, DeepSeek, Yi, Baichuan, Gemma
- **Multi-modal**: Qwen2-VL, Llava, InternVL, MiniCPM-V, GLM4v, Phi3.5-Vision
- **Special Models**: Embedding models, Rerankers, Reward models

### Configuration System
- Training arguments in `swift/llm/argument/`
- Model-specific configurations in `swift/llm/model/`
- Template configurations in `swift/llm/template/`
- DeepSpeed configs in `swift/llm/ds_config/`

### Extension System
- **Custom Models**: Register new models in `swift/llm/model/register.py`
- **Custom Datasets**: Register datasets in `swift/llm/dataset/register.py`  
- **Custom Templates**: Add templates in `swift/llm/template/register.py`
- **Plugins**: Extend functionality via `swift/plugin/`

## Development Notes

### Code Conventions
- Uses lazy imports (`_LazyModule`) for performance
- Follows ModelScope/HuggingFace model/tokenizer patterns
- Extensive use of registration decorators for extensibility
- Template-based approach for model conversation formatting

### Key Files to Understand
- `swift/llm/model/register.py` - Model registration and metadata
- `swift/llm/template/register.py` - Template registration system
- `swift/llm/train/sft.py` - Core supervised fine-tuning logic
- `swift/trainers/trainers.py` - Main trainer implementations
- `swift/cli/main.py` - CLI entry point and command routing

### Common Patterns
- Models are registered with decorators and accessed via string identifiers
- Templates handle conversation formatting and special tokens
- Training uses HuggingFace Transformers with custom extensions
- Inference engines are swappable (PyTorch, vLLM, SGLang, LMDeploy)

### Environment Variables
Key environment variables for training:
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `MASTER_ADDR`/`MASTER_PORT` - Distributed training coordination  
- `SEQUENCE_PARALLEL_IMPL` - Sequence parallelism implementation
- `VIDEO_MAX_PIXELS`/`IMAGE_MAX_PIXELS` - Multi-modal processing limits

### Testing
- Tests located in `tests/` directory structured by functionality:
  - `tests/llm/` - Core LLM functionality tests
  - `tests/train/` - Training-related tests  
  - `tests/infer/` - Inference tests
  - `tests/models/` - Model-specific tests
  - `tests/tuners/` - Parameter-efficient tuning tests
- Use `make test` or `pytest tests/` directly
- Test configuration in `tests/run_config.yaml`
- Covers model loading, training, inference, and export functionality

### Installation and Setup
```bash
# Basic installation
pip install ms-swift

# Development installation
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# With evaluation capabilities
pip install ms-swift[eval]

# Full installation with all dependencies
pip install ms-swift[all]
```