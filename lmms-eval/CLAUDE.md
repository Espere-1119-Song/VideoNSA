# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **lmms-eval**, a unified evaluation framework for Large Multimodal Models (LMMs). It supports evaluation across text, image, video, and audio tasks with 100+ benchmarks and 30+ model integrations. The project is inspired by and builds upon lm-evaluation-harness.

## Development Environment

### Conda Environment
- **ALWAYS activate the msswift conda environment first**: `conda activate msswift`
- All commands should be run within the msswift environment
- This environment contains the required dependencies for this project

### Package Management
- **ONLY use `uv`** for package operations in CI/development
- **NEVER use pip directly** - always use `uv pip` when uv is configured
- Installation: `uv add package` for new dependencies
- Running tools: `uv run command`

### Environment Setup
```bash
# IMPORTANT: Always activate msswift environment first
conda activate msswift

# For development
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
uv venv dev
source dev/bin/activate
uv pip install -e .
```

## Core Commands

### Running Evaluations
Main CLI entry point is through Python module:
```bash
python -m lmms_eval --model MODEL_NAME --model_args MODEL_ARGS --tasks TASKS --batch_size BATCH_SIZE --device DEVICE
```

Alternative CLI command (equivalent):
```bash
lmms-eval --model MODEL_NAME --model_args MODEL_ARGS --tasks TASKS --batch_size BATCH_SIZE --device DEVICE
```

Example evaluation:
```bash
python -m lmms_eval --model qwen2_5_vl --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=12845056,attn_implementation=sdpa --tasks mmmu,mme,mmlu_flan_n_shot_generative --batch_size 128 --limit 8 --device cuda:0
```

### Task Discovery
- List all tasks: `lmms-eval --tasks list`
- List with counts: `lmms-eval --tasks list_with_num` (downloads datasets)
- List groups: `lmms-eval --tasks list_groups`
- List tags: `lmms-eval --tasks list_tags`
- List subtasks: `lmms-eval --tasks list_subtasks`

### Development Tools
- **Format code**: `uv run black --line-length=240 .`
- **Sort imports**: `uv run isort --profile black .`
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`

### Testing & Debugging
- **Debug prompts**: `--write_out` shows prompts for first few documents
- **Integrity checks**: `--check_integrity` runs task test suite
- **Limit evaluation**: `--limit N` processes only N examples per task
- **Verbose output**: `--verbosity DEBUG` for detailed error information

## Code Architecture

### Core Components
- **`__main__.py`**: CLI entry point with comprehensive argument parsing
- **`evaluator.py`**: Main evaluation orchestration and distributed execution
- **`api/`**: Core abstractions (model, task, registry, metrics, samplers, filters)
- **`models/`**: Model implementations split by interface type
- **`tasks/`**: 100+ evaluation benchmarks organized by domain
- **`loggers/`**: Result tracking (WandB, evaluation tracker, output formatting)
- **`filters/`**: Data processing pipeline (decontamination, extraction, transformation)
- **`llm_judge/`**: LLM-as-judge evaluation framework with multiple providers

### Model Architecture
Two main model types with different interfaces:
- **`simple/`**: Direct model integrations (70+ models) - most common
- **`chat/`**: Chat template-based models with conversation formatting
- **`qwen25_vl/`**: Custom implementation with specialized components

Available models include: aero, aria, claude, gemini_api, gpt4v, llava variants, qwen variants, internvl, phi3v, and many more.

### Task System Architecture
- **Task configs**: YAML files define evaluation parameters, metrics, few-shot examples
- **Utils modules**: Python logic for data processing and metric computation
- **Modular design**: Tasks can be grouped, tagged, and composed
- **Template system**: Reusable YAML templates for common patterns
- **Multi-modal support**: Images, videos, audio handled through `Instance.args`

### Data Processing Pipeline
1. **Dataset loading**: Automatic downloads from HuggingFace Hub
2. **Context building**: Separated from media processing for memory efficiency
3. **Request batching**: Auto batch sizing with distributed processing
4. **Response filtering**: Multiple filter types for answer extraction
5. **Metric computation**: Bootstrap confidence intervals and aggregation

## Development Workflow

### Code Quality Requirements
- **Line length**: 240 characters (configured in pyproject.toml and pre-commit)
- **Import organization**: isort with black profile
- **Code formatting**: black formatter with specific configuration
- **No type annotations**: Current codebase does not enforce type hints

### Adding New Models
1. Choose model type (`simple/` vs `chat/`)
2. Create model file in appropriate directory
3. Implement required interface methods (`generate_until`, `loglikelihood`, etc.)
4. Add model to `AVAILABLE_*_MODELS` dict in `__init__.py`
5. Test with `--write_out --limit 5` for debugging
6. Add example script in `examples/models/`

### Adding New Tasks
1. Create task directory in `lmms_eval/tasks/`
2. Create YAML configuration with task metadata
3. Implement `utils.py` with processing logic
4. Define metrics and answer extraction filters
5. Test with `--limit` and `--check_integrity`
6. Document task in appropriate location

### Memory and Performance Considerations
- **Large datasets**: Media files processed during response phase only
- **Distributed execution**: Supports both `accelerate` and `torchrun` backends
- **Caching**: SQLite-based caching for responses and dataset requests
- **Batch optimization**: Auto-sizing with padding for distributed ranks

### Environment Variables
Required for many tasks:
- `OPENAI_API_KEY`: OpenAI API access
- `HF_TOKEN`: HuggingFace Hub access
- `HF_HOME`: HuggingFace cache directory
- `ANTHROPIC_API_KEY`, `DASHSCOPE_API_KEY`, etc.: Various API providers
- `LMMS_EVAL_PLUGINS`: External plugin loading

## Common Patterns and Best Practices

### Evaluation Patterns
- Always test with `--limit` before full evaluation
- Use `--check_integrity` to verify task implementation
- Set appropriate `--verbosity` level for debugging
- Cache responses with `--use_cache` for repeated experiments

### Development Patterns
- Follow existing model implementations as templates
- Use descriptive variable names and clear function signatures
- Separate concerns: data loading, processing, metric computation
- Handle multimodal inputs through standardized `Instance.args`

### Error Handling
- Most errors surface through `--verbosity DEBUG`
- Task integrity issues caught by `--check_integrity`
- Memory issues often resolved by adjusting batch sizes
- API rate limiting handled per provider implementation

## Important Notes

- **Package management**: Use `uv` in development environments
- **Model interfaces**: No unified HuggingFace interface yet - each model needs custom implementation
- **Distributed execution**: Framework handles multi-GPU automatically
- **Result reproducibility**: Seed control through `--seed` parameter
- **Plugin system**: External models/tasks supported via `LMMS_EVAL_PLUGINS`