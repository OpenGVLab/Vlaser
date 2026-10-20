# Vlaser VLA Quick Start

This codebase is based on [open-pi-zero](https://github.com/allenzren/open-pi-zero). You can refer to [open-pi-zero](https://github.com/allenzren/open-pi-zero) for more details of the codebase.

## 🛠️ Installation

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n vlaser python=3.10
  conda activate vlaser
  ```
- Install dependencies using requirements.txt:

  ```bash
  pip install -r requirements.txt
  ```
  if you have problem while installing dependencies with requirements.txt, you can install the dependencies from [InternVL](https://github.com/OpenGVLab/InternVL) and [open-pi-zero](https://github.com/allenzren/open-pi-zero) step by step.

### How to run the same on Intel's XPU

Skip the CUDA pins of `torch` and `torchvision` in `requirements.txt` and install the XPU build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install transformers==4.48.0 hydra-core==1.3.2 omegaconf==2.3.0 einops timm opencv-python-headless sentencepiece bitsandbytes imageio decord
```

The vendored `src/model/internvl_chat/internvl` package is missing two modules that action inference imports (a `.gitignore` rule skips them). Restore them with `cp -rn ../../Vlaser_VLM/internvl_chat/internvl/* src/model/internvl_chat/internvl/` from this folder.

Run the same evaluation scripts as above, with `ZE_AFFINITY_MASK=0` set (it selects the card the way `CUDA_VISIBLE_DEVICES=0` does) and `device=xpu:0` in place of `device=cuda:0`. The `device` field of the eval configs is empty by default, which picks XPU, then CUDA, then CPU. Attention in the action path is plain PyTorch, so flash-attn is not needed. `use_bf16=True` is the usual choice on XPU.

The SimplerEnv simulator itself (ManiSkill2 and SAPIEN) is not part of this change and stays on its own CUDA or Vulkan setup.

## 📦 Training

- Download [Vlaser Models](https://huggingface.co/collections/OpenGVLab/vlaser) before Training.

- Here is an example training script in [slurm/train_internvl.sh](./slurm/train_internvl.sh), you can use is as follows.
    ```bash
    sh slurm/train_internvl.sh
    ```
    you can custom your training strategy both in the script or the corresponding .yaml file. You can edit your own .yaml file as well. Details can be found in [open-pi-zero](https://github.com/allenzren/open-pi-zero).
  
## 📊 Evaluation

- Here are the examples of evaluating the model
    ```bash
    # for Bridge
    sh slurm/eval_simpler_bridge.sh
    # for Fractal
    sh slurm/eval_simpler_fractal.sh
    ```
    Similar with the training process, you can modify the evaluation both in the script or the corresponding .yaml file. You can edit your own .yaml file as well. Details can be found in [open-pi-zero](https://github.com/allenzren/open-pi-zero).

    You can download our model [Vlaser-2B-VLA](https://huggingface.co/OpenGVLab/Vlaser-2B-VLA) and evaluate with these scripts.

## Notes

This codebase is also support a third-party implement of [$\pi_{0}$](https://www.physicalintelligence.company/blog/pi0), you can find the details [here](https://github.com/allenzren/open-pi-zero).

## Acknowledgement

This codebase is based on [open-pi-zero](https://github.com/allenzren/open-pi-zero). Thank you for the great work!