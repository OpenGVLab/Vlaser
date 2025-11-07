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