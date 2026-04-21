<div align="center">

# 🏔️ Alpamayo 1.5

### Supercharging Autonomous Driving with Interactive, Steerable Reasoning

[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Alpamayo--1.5--10B-blue)](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

**📖 Please read the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) first!**
The model card contains comprehensive details on model architecture, inputs/outputs, licensing, and tested hardware configurations. This GitHub README focuses on setup, usage, and frequently asked questions.

**Research fork (USC):** this checkout extends the upstream Alpamayo 1.5 codebase with **inference-time steering in trajectory space** via **denoising-time classifier guidance** on the flow-matching expert. See [Research extension: denoising-time trajectory steering](#research-extension-denoising-time-trajectory-steering-usc) for motivation, design, and how to reproduce experiments.

## Prerequisites

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** with `nvcc` (required to compile `flash-attn` from source). If you don't have it, see [Troubleshooting](#flash-attention-issues) for a fallback using PyTorch's built-in SDPA.
- **Python 3.12**

### Hardware requirements

| Configuration                                           | VRAM   |
| ------------------------------------------------------- | ------ |
| Single-sample inference (`num_traj_samples=1`)          | ~24 GB |
| Multi-sample inference (`num_traj_samples=16`)          | ~40 GB |
| Multi-sample inference with CFG (`num_traj_samples=16`) | ~60 GB |

Measured on an NVIDIA H100 80GB GPU.

## Getting Started

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Set up the environment

```bash
uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```

> **Note:** If `uv sync` fails on `flash-attn`, see [Troubleshooting](#flash-attention-issues) below.

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here:

- 🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- 🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Then authenticate:

```bash
hf auth login
```

Get your token at: https://huggingface.co/settings/tokens

> **Note:** The `physical_ai_av` package (auto-installed via dependencies) streams data from the HuggingFace dataset. You must have accepted the dataset access request above before running inference.

## Running Inference

### Test script

NOTE: This script will download both some example data (relatively small) and the model weights (22 GB).
The latter can be particularly slow depending on network bandwidth.
For reference, it takes around 2.5 minutes on a 100 MB/s wired connection.

```bash
python src/alpamayo1_5/test_inference.py
```

In case you would like to obtain more trajectories and reasoning traces, please feel free to increase
the `num_traj_samples` argument in the script.

### Interactive notebooks

We provide notebooks that demonstrate the different capabilities of Alpamayo 1.5 under `notebooks/`, including standard model inference, incorporating navigation guidance, modifying the number of cameras, and visual question answering.

### Inference methods

Alpamayo 1.5 provides two inference methods:

- **`sample_trajectories_from_data_with_vlm_rollout`** -- Full pipeline: the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory predictions conditioned on the VLM's hidden states. This is the primary inference method used by the test script and most notebooks.

- **`generate_text`** -- Text-only generation for visual question answering (VQA). Returns extracted text fields.

### Denoising-time guidance (research)

The flow-matching sampler (`src/alpamayo1_5/diffusion/flow_matching.py`) accepts an optional **`denoising_guidance_fn(x, t, v, step_index) -> delta_v`**. Each Euler step becomes **`v ← v + delta_v`** then **`x ← x + dt · v`**. Pass it through **`diffusion_kwargs`** when calling **`sample_trajectories_from_data_with_vlm_rollout`**:

```python
from alpamayo1_5.steering import HeuristicBehaviorClassifier, classifier_gradient_guidance_fn

clf = HeuristicBehaviorClassifier().to("cuda")
guidance = classifier_gradient_guidance_fn(clf, target_class=0, scale=0.15)

pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
    data=model_inputs,
    diffusion_kwargs={"denoising_guidance_fn": guidance},
    # ... other kwargs unchanged ...
)
```

**A/B script** (baseline vs guided on one Physical AI clip, same RNG reset per run):

```bash
# Quieter logs
ALPAMAYO_DEBUG=0 python src/alpamayo1_5/compare_denoising_guidance.py

# Stronger intervention (tune carefully)
ALPAMAYO_DEBUG=0 python src/alpamayo1_5/compare_denoising_guidance.py --guidance-scale 0.5 --target-class 0
```

`--target-class` indexes the heuristic prototype: **0** ≈ yield-ish (negative mean accel), **1** neutral, **2** accel-ish. **`--deterministic`** enables stricter cuDNN settings when you need more reproducible A/Bs.

## Research extension: denoising-time trajectory steering (USC)

### Goal

We study **where to intervene** in a Vision-Language-Action (VLA) stack that first writes a **Chain-of-Causation (CoC)** with a VLM, then samples a **continuous trajectory** with a **separate action expert**. Prior work (and our own architectural analysis) suggests that **editing reasoning tokens or the KV cache** can yield **inconsistent** trajectory effects, may push the expert **off-distribution**, and confounds **faithfulness** questions about CoC text.

Our primary direction is **trajectory-level control inside the expert’s denoising loop**: at each flow-matching step, augment the learned vector field with a **classifier score gradient**:

$$v \leftarrow v + \lambda \, \nabla_x \log p_C(y \mid x, t)$$

Here **`x`** is the noisy action tensor (normalized unicycle accel/curvature over 64 waypoints), **`t`** is flow time in **`[0, 1]`**, **`y`** is a target **behavioral** class, and **`C`** is a lightweight classifier. The VLM, CoC text, and **frozen `past_key_values`** are **unchanged**; only the **Euler integration** over actions is steered. This matches the “denoising-time guidance” formulation in our group’s midterm report and keeps the expert conditioned on **in-distribution** visual + reasoning context.

### What the upstream model already does (relevant pieces)

1. **VLM (`Qwen3VLForConditionalGeneration`)** consumes multi-camera images, ego history tokens fused into the prompt, and autoregressively generates **CoC** until a **`<|traj_future_start|>`** stop; discrete future-trajectory logits are masked during that phase.
2. The resulting **KV cache** is reused by a **second transformer (“expert”)** without its own token embedding layer. For each flow step, **`action_in_proj(x, t)`** builds query embeddings; the expert attends to the cache; **`action_out_proj`** maps hidden states back to a **velocity field** in action space.
3. **`FlowMatching`** integrates that field with **fixed-step Euler** from **`t = 0`** to **`t = 1`** (default 10 steps in released configs unless overridden).

Our extension sits strictly in step (3), optionally modifying **`v`** after the expert forward.

### What we implemented in this repo

| Piece | Role |
| ----- | ---- |
| **`FlowMatching.sample` / `_euler`** | Optional **`denoising_guidance_fn`**; gradient-based **`delta_v`** is added to **`v`** each step. |
| **`alpamayo1_5/steering/`** | **`HeuristicBehaviorClassifier`**: differentiable **3-class** logits from **pooled (mean accel, mean curvature)** vs fixed prototypes (stand-in until **\(C\)** is trained on Physical AI). **`classifier_gradient_guidance_fn`** wraps it as a **`denoising_guidance_fn`**. |
| **`compare_denoising_guidance.py`** | Loads the default example clip, runs **baseline** then **guided** with **`torch.cuda.manual_seed_all`** (and friends) reset to the same seed before each full **`sample_trajectories_from_data_with_vlm_rollout`**, prints **minADE**, **FDE**, mean XY step length, and deltas. |
| **`test_inference.py`** | Optional **`ALPAMAYO_DEBUG`** logging for the expert + Euler loop (see script docstring). |
| **`tests/test_denoising_guidance.py`** | Unit tests for the guidance hook and tensor shapes. |

**Important:** the shipped **`HeuristicBehaviorClassifier`** is for **pipeline debugging and coarse steering experiments**, not a substitute for a **dataset-trained, noise-conditioned `C(x, t)`** as in the full research plan.

### Preliminary observations (single-clip sanity checks)

On the default Physical AI example clip, with **identical CoC** between A/B and matched noise at the first Euler step:

- A modest **`scale`** (e.g. **0.15**, **`target_class=0`**) produced small **improvements** in **minADE / FDE** vs ground truth in our script’s metrics.
- A larger **`scale`** (e.g. **0.5**) increased the **magnitude of `v`** corrections and produced **larger** **ΔminADE / ΔFDE** on that clip.

These results **do not** generalize beyond the tested scenario; they only validate that the **hook is live** and **λ scales the intervention** as expected.

### Next steps (research roadmap)

1. **Train `C(x, t)`** on Physical AI AV with noisy actions from the **forward** flow and **behavior labels** aligned to CoC vocabulary.
2. **Sweep `λ`** and optional **time schedules** (early vs late denoising); measure **behavioral hit rate** vs **minADE/minFDE** tradeoffs (evaluation protocol in the midterm).
3. **Scale** to scenario strata (nominal vs long-tail) and compare against **reasoning-level** interventions.

## Project Structure

```
alpamayo1.5/
├── notebooks/
│   ├── inference.ipynb                  # Standard model inference
│   ├── inference_cam_num.ipynb          # Inference with different camera counts
│   ├── inference_nav.ipynb              # Inference with navigation guidance
│   └── inference_vqa.ipynb              # Visual question answering
├── src/
│   └── alpamayo1_5/
│       ├── action_space/
│       │   └── ...                      # Action space definitions (unicycle accel/curvature)
│       ├── diffusion/
│       │   └── flow_matching.py         # Euler flow matching (+ optional denoising guidance)
│       ├── geometry/
│       │   └── ...                      # Geometry utilities and modules
│       ├── models/
│       │   └── ...                      # Alpamayo1_5, VLM + expert, rollout API
│       ├── steering/
│       │   └── denoising_guidance.py    # Heuristic classifier + gradient guidance factory
│       ├── __init__.py                  # Package marker
│       ├── compare_denoising_guidance.py # A/B: baseline vs guided trajectory sampling
│       ├── config.py                    # Model and experiment configuration
│       ├── helper.py                    # Utility functions
│       ├── load_physical_aiavdataset.py # Dataset loader
│       └── test_inference.py            # End-to-end inference smoke test
├── tests/
│   └── test_denoising_guidance.py       # Unit tests for guidance hook
├── pyproject.toml                       # Project dependencies
└── uv.lock                              # Locked dependency versions
```

## Troubleshooting

### Flash Attention issues

The model uses Flash Attention 2 by default. `flash-attn` requires CUDA Toolkit (specifically `nvcc`) at build time. If you see build errors during `uv sync`:

**Option A: Install without flash-attn and use SDPA fallback**

```bash
uv sync --active --no-install-package flash-attn
```

Then load the model with PyTorch's built-in scaled dot-product attention:

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

**Option B: Install CUDA Toolkit, then retry**

Install CUDA Toolkit 12.x (e.g., via your package manager or [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)), ensure `nvcc` is on your PATH, then re-run:

```bash
uv sync --active
```

## Frequently Asked Questions (FAQ)

<details>
<summary><strong>How does Alpamayo 1.5 relate to Alpamayo 1?</strong></summary>

Alpamayo 1.5 expands upon the architecture released in Alpamayo 1 and fully realizes what is described in our paper [*"Alpamayo 1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
"*](https://arxiv.org/abs/2511.00088). Specifically:

| Feature                                 | Description                                                      | Alpamayo 1             | Alpamayo 1.5       |
| --------------------------------------- | ---------------------------------------------------------------- | ---------------------- | ------------------ |
| **Chain-of-Causation (CoC) reasoning**  | Hybrid auto-labeling with human in the loop for reasoning traces | ✅ Included            | ✅ Included        |
| **Vision-Language-Action architecture** | Cosmos-Reason backbone + action expert                           | ✅ Included            | ✅ Included        |
| **Trajectory prediction**               | 6.4s horizon, 64 waypoints at 10 Hz                              | ✅ Supported           | ✅ Supported       |
| **RL post-training**                    | Reinforcement learning for reasoning/action consistency          | ❌ Not RL post-trained | ✅ RL post-trained |
| **Navigation conditioning**             | Explicit navigation inputs                                       | ❌ Not supported       | ✅ Supported       |
| **General VQA**                         | Supports visual question answering                               | ❌ Not supported       | ✅ Supported       |
| **Flexible multi-camera support**       | Supports a variable number of input cameras                      | ❌ Not supported       | ✅ Supported       |

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept navigation inputs?</strong></summary>

Yes! Please see `notebooks/inference_nav.ipynb` for examples.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 support general VQA?</strong></summary>

Yes! Please see `notebooks/inference_vqa.ipynb` for examples.

</details>

<details>
<summary><strong>Was Alpamayo 1.5 post-trained with Reinforcement Learning (RL)?</strong></summary>

Yes! Alpamayo 1.5 has undergone RL post-training, achieving improvements in reasoning quality and reasoning-trajectory alignment as a result.

</details>

<details>
<summary><strong>Does Alpamayo 1.5 accept different numbers of cameras?</strong></summary>

Yes! Please see `notebooks/inference_cam_num.ipynb` for examples. Note that model accuracy may degrade with fewer cameras, the magnitude of which will depend on the specific scenario. For instance, it is expected that Alpamayo 1.5 would struggle to see cross-traffic in a right turn if only provided a front-facing camera.

</details>

<details>
<summary><strong>What are the minimum GPU requirements?</strong></summary>

You need an NVIDIA GPU with at least **24 GB VRAM** for inference. Tested configurations include RTX 3090, A100, H100, and B200. Running on GPUs with less memory (e.g., 16 GB) will likely result in CUDA out-of-memory errors. Please refer to our [hardware requirements](#hardware-requirements) for more information.

</details>

<details>
<summary><strong>Can I use this model in production / commercial applications?</strong></summary>

No. The model weights are released under a **non-commercial license**. This release is intended for research, experimentation, and evaluation purposes only. See the [License](#license) section and the [HuggingFace Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B) for details.

</details>

## License

Apache License 2.0 - see [LICENSE](./LICENSE) for details.

## Disclaimer

Alpamayo 1.5 is a pre-trained reasoning model designed to accelerate research and development in the autonomous vehicle (AV) domain. It is intended to serve as a foundation for a range of AV-related use cases-from instantiating an end-to-end backbone for autonomous driving to enabling reasoning-based auto-labeling tools. In short, it should be viewed as a building block for developing customized AV applications.

Important notes:

- Alpamayo 1.5 is provided solely for research, experimentation, and evaluation purposes.
- Alpamayo 1.5 is not a fully fledged driving stack. Among other limitations, it lacks access to critical real-world sensor inputs, does not incorporate required diverse and redundant safety mechanisms, and has not undergone automotive-grade validation for deployment.

By using this model, you acknowledge that it is a research tool intended to support scientific inquiry, benchmarking, and exploration—not a substitute for a certified AV stack. The developers and contributors disclaim any responsibility or liability for the use of the model or its outputs.

## Citation

If you use Alpamayo 1.5 in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo-R1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
