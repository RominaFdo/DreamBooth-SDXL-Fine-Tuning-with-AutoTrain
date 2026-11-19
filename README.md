# DreamBooth SDXL Fine-Tuning with AutoTrain

A small notebook for fine-tuning **Stable Diffusion XL** on your own images with **DreamBooth**, using Hugging Face [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced).

The sample dataset is five photos of trees in [`images/`](images/). The goal is to teach SDXL a specific subject from a handful of pictures so it can generate new images of it from a text prompt.

> **Status:** an early learning project from 2024. The notebook sets up the config, but the training run was never completed (see [Known issues](#known-issues)).

## How it works

1. Install `autotrain-advanced`.
2. Set the base model, project name and instance prompt.
3. Set the training hyperparameters.
4. Export them as environment variables.
5. Call the `autotrain dreambooth` CLI to train LoRA weights on the images in `images/`.

## Configuration

| Setting | Value |
| --- | --- |
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| Instance prompt | `a photo of a tree` |
| Resolution | 1024 |
| Steps | 500 |
| Learning rate | 1e-4 |
| Batch size / grad. accumulation | 1 / 4 |
| Mixed precision | fp16 |
| 8-bit Adam / xformers / train text encoder | off |

## Requirements

- Python 3.10+
- An NVIDIA GPU with roughly 16 GB+ VRAM for SDXL at 1024px (or use Google Colab / Kaggle)
- A Hugging Face account, logged in with `huggingface-cli login` (needed to download the SDXL weights)

## Usage

```bash
pip install -U autotrain-advanced
```

Open `imgGen.ipynb`, change the variables in the "basic details" and "Hyper Parameters" cells, then run every cell. You can also run the training command straight from a Linux/macOS shell:

```bash
autotrain dreambooth \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --project-name tree-example \
  --image-path images/ \
  --prompt "a photo of sks tree" \
  --resolution 1024 \
  --batch-size 1 \
  --num-steps 500 \
  --gradient-accumulation 4 \
  --lr 1e-4 \
  --mixed-precision fp16
```

When training finishes, the LoRA weights are saved in the project folder. Load them on top of the base model with `diffusers`:

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("tree-example", weight_name="pytorch_lora_weights.safetensors")

image = pipe("a photo of sks tree in the snow", num_inference_steps=30).images[0]
image.save("output.png")
```

**Tip:** DreamBooth works best when the prompt contains a rare identifier token such as `sks` (for example `a photo of sks tree`). The model then ties your subject to that token instead of overwriting what it already knows about "tree".

## Known issues

These are the problems that stopped the original notebook from running:

- **`%autotrain` is not a magic command.** In Jupyter, use `!autotrain ...` to run it as a shell command.
- **Bash-only syntax.** The `$( [[ ... ]] && echo ... )` lines that add optional flags need bash. They fail on Windows; run the notebook on Linux, WSL or Colab.
- **`%rmdir /images/.ipynb_checkpoints`** uses an absolute path and Windows switch parsing. Use `!rm -rf images/.ipynb_checkpoints` instead.
- **AutoTrain versions.** The notebook was written for `autotrain-advanced` 0.8.x. Newer releases may rename or drop CLI flags, so check `autotrain dreambooth --help`.

## Project structure

```
.
├── imgGen.ipynb   # training notebook
└── images/        # training images (5 photos of trees)
```
