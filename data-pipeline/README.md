# Vlaser Data pipeline Quick Start

This codebase is used to generate the In-domain QA data items used for VLM pretraining, based on [open-pi-zero](https://github.com/allenzren/open-pi-zero). You can refer to [open-pi-zero](https://github.com/allenzren/open-pi-zero) for more details of the codebase.

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

## SimplerEnv data pipeline

  ```bash
  cd data-pipeline 
  bash slurm/data_generator.sh [bridge|fractal] [general|spatial|grounding]
  ```

This data script supports data generation for WidowX(bridge) and Google Robot(fractal), ranging data types including general QA, spatial intelligence QA as well as grounding QA.

For data quality filtering, please refer to:

  ```bash
  cd data-pipeline
  python src/agent/filter.py \
  --input_folder xxx # Path for jsonl containing generated data above
  --image_root xxx # Path for image root path
  --output_root xxx # Path for data items after filtering
  ```

## RoboTwin data pipeline

  ```bash
  cd data-pipeline/RoboTwin-QA
  python [GeneralQA|GroundingQA|SpatialQA].py
  ```

This data script supports data generation for RoboTwin 2.0, ranging data types including general QA, spatial intelligence QA as well as grounding QA.

## Notes

This codebase is also support a third-party implement of [$\pi_{0}$](https://www.physicalintelligence.company/blog/pi0), you can find the details [here](https://github.com/allenzren/open-pi-zero).

## Acknowledgement

This codebase is based on [open-pi-zero](https://github.com/allenzren/open-pi-zero). Thank you for the great work!