# MoE Probe: A Deep Dive into Mixture of Experts

`moe_probe` is a diagnostic and educational tool designed to visualize the internal routing decisions of Mixture of Experts (MoE) models in real-time. 

Using IBM's **Granite 3.0 3B-A800M**, this project attaches PyTorch forward hooks to the model's router to capture how tokens are assigned to specific experts. It provides a "mental model" for understanding sparse activation—moving beyond the "what" of MoE to the "how" of its real-time execution.

## Key Features

- **Real-time VU Meters**: Visualizes the 40-dimensional softmax output of the gating network for every generated token.
- **Expert Heatmaps**: Generates a 2D fingerprint of a generation task, showing which experts are "backbone" nodes versus task-specific specialists.
- **Architectural Introspection**: Uses the model to analyze its own routing statistics and infer the "specialization" of its experts based on activation patterns.
- **Comparative Demos**: Scripts to compare dense vs. sparse performance and routing behavior across different domains (Code, Math, Linguistics).

## Repository Structure

- `demo_deepdive.py`: The primary interactive demo with VU meters and heatmap generation.
- `chat.py`: A simple CLI interface for interacting with the MoE model.
- `heatmap.py`: Logic for rendering the expert activation heatmaps.
- `inspect_model.py`: Utility to explore the model's layer structure and identify router components.
- `talktrack.md`: The narrative guide used for live demonstrations.

## Getting Started

### Prerequisites

- Linux/macOS (Linux recommended for GPU support)
- Python 3.10+
- NVIDIA GPU with ~8GB+ VRAM (for 4-bit quantized execution)

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bprendie/moe_probe.git
   cd moe_probe
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Adjust for your CUDA version
   pip install transformers accelerate bitsandbytes sentencepiece matplotlib numpy
   ```

### Running the Demo

To launch the real-time routing visualization:

```bash
python3 demo_deepdive.py
```

For a simpler chat interface:

```bash
python3 chat.py
```

## How it Works

In a standard Transformer, every token passes through the same Feed-Forward Network (FFN). In a MoE model like Granite 3.0, that FFN is replaced by **40 independent experts**. 

For every token, a **gating network (router)** selects the **top-2** experts to handle the computation. `moe_probe` intercepts this selection process, allowing you to watch the "learned specialization" of the model emerge live as it switches between languages, code, and creative writing.

## License

[Specify License, e.g., Apache 2.0 or MIT]
