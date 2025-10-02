# VideoNSA: Native Sparse Attention for Video Understanding

VideoNSA is a novel learnable, hardware-aware sparse-attention framework designed to enhance video understanding by addressing key limitations in current video processing pipelines.

## Key Features

- **Massive Context Scaling**: Processes up to 128K vision-text tokens
- **Learned Sparsity**: Intelligently learns sparsity patterns over video tokens
- **Efficient Performance**: Achieves leading results using only 3.6% of full attention budget
- **Decisive-Moment Fidelity**: Preserves critical moments and long-range coherence in videos
- **Built on Qwen2.5-VL-7B**: Leverages state-of-the-art vision-language model

## Methodology

VideoNSA employs a hybrid attention strategy with three complementary branches:

### 1. Compression Branch
- Averages frame key-value blocks to maintain salient visual cues
- Keeps compute budget linear with context length

### 2. Selection Branch
- Ranks and retains the most informative video segments
- Focuses attention on discriminative events

### 3. Sliding Window Branch
- Ensures local temporal coverage
- Captures fine-grained motion details

## Innovations

- **Dynamic Per-Head Gating**: Two-layer MLP gates for adaptive token allocation
- **End-to-End Training**: Fully trainable vision pathway
- **Task-Adaptive**: Dynamically allocates tokens across different video understanding tasks
- **Attention Sink Mitigation**: Reduces attention sink problems through learned sparsity

## Training

- **Dataset**: Filtered LLaVA-Video-178K
- **Sampling Rate**: 4 fps
- **Context Limit**: 36K tokens
- **Compute**: ~4,600 H100 GPU hours

## Applications

- Long video understanding
- Detailed video captioning
- Temporal event analysis
- Video question answering

## Installation

```bash
# TODO: Add installation instructions
```

## Usage

### Training

```bash
# TODO: Add training command
```

### Evaluation

```bash
# TODO: Add evaluation command
```

## Citation

```bibtex
@misc{chai2025auroracapefficientperformantvideo,
      title={AuroraCap: Efficient, Performant Video Detailed Captioning},
      author={Wenhao Chai et al.},
      year={2025}
}
```

## Resources

- [Project Website](https://enxinsong.com/VideoNSA-web/)
