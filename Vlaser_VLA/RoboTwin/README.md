# Vlaser VLA Quick Start

This codebase is based on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). You can refer to [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) for more details of the codebase.

## 🛠️ Installation

- Please refer to [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) for environment setup.

## 📦 Training

- Download Training datasets from [Vlaser_vla](https://huggingface.co/datasets/ganlinyang/Vlaser_vla), note that we only use a subset of original RoboTwin 2.0 training sets for vla post-training and evaluation.

- Download pre-trained VLM models [Vlaser-RoboTwin VLM Models](https://huggingface.co/ganlinyang/Vlaser-RoboTwin/tree/main/VLMs) used as training initialization, we provide four models including **Vlaser-QA**, **Vlaser-Spatial**, **Vlaser-Grounding** and **Vlaser-All**.

- For VLA post-training, please run the script:
    ```bash
    cd Vlaser_VLA/RoboTwin/policy/internvla_2B_parallel_decoding
    accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8980 vla-scripts/train_mine_new.py config=configs/training_based_on_internvl3_2B.yaml
    ```
  
## 📊 Evaluation

- Download the VLA models after post-training from [Vlaser-RoboTwin VLA Models](https://huggingface.co/ganlinyang/Vlaser-RoboTwin/tree/main/VLAs) and put under folder `vlaser_ckpt/`.

- Here are the examples of evaluating the model
    ```bash
    cd Vlaser_VLA/RoboTwin/
    bash eval.sh [all|general|spatial|grounding]
    ```

## Acknowledgement

This codebase is based on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Thank you for the great work!