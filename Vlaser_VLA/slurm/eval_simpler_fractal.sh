#!/bin/bash

#SBATCH --job-name=eval-fractal
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=15:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"
ROOT_PATH_DIR="${PWD}"
export VLA_DATA_DIR="$DEFAULT_DATA_DIR"
export VLA_LOG_DIR="$DEFAULT_LOG_DIR"
export ROOT_PATH="$ROOT_PATH_DIR"
# export CUDA_VISIBLE_DEVICES=1
# better to run jobs for each task

# this list is testing for original open-pi-zero repo
TASK_CONFIGS=(
    "google_robot_pick_horizontal_coke_can:fractal_coke"
    "google_robot_pick_vertical_coke_can:fractal_coke"
    "google_robot_pick_standing_coke_can:fractal_coke"
    "google_robot_move_near_v0:fractal_move"
    "google_robot_open_drawer:fractal_drawer"
    "google_robot_close_drawer:fractal_drawer"
    "google_robot_place_apple_in_closed_top_drawer:fractal_apple"
)

# this list is testing for widly used SimplerENV
TASK_CONFIGS=(
    "google_robot_pick_coke_can:fractal_coke"
    "google_robot_move_near_v0:fractal_move"
    "google_robot_open_drawer:fractal_drawer"
)

TASK_CONFIG=${TASK_CONFIGS[0]}
TASK="${TASK_CONFIG%%:*}"
CONFIG_NAME="${TASK_CONFIG##*:}_internvl_448"

echo $CONFIG_NAME

# EVAL_MATCH=1 for Visual Matching and EVAL_VAR=1 for Variant Aggregation
EVAL_MATCH=1 INTERNVL=1 DEBUG_448=1 PYTHONPATH=. HYDRA_FULL_ERROR=1 \
    python \
    scripts/run.py \
    --config-name=$CONFIG_NAME \
    --config-path=../config/eval \
    device=cuda:0 \
    seed=42 \
    env.task=$TASK \
    horizon_steps=4 \
    act_steps=2 \
    env.adapter.max_seq_len=384 \
    cond_steps=1 \
    use_bf16=False \
    use_torch_compile=True \
    num_inference_steps=10 \
    name=eval \
    time_max_period=100.0 \
    action_expert_rope_theta=100.0 \
    checkpoint_path=your_ckpt_path 
