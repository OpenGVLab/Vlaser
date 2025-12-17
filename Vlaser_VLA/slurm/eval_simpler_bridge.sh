#!/bin/bash

#SBATCH --job-name=eval-bridge
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"
export VLA_DATA_DIR="$DEFAULT_DATA_DIR"
export VLA_LOG_DIR="$DEFAULT_LOG_DIR"
# better to run jobs for each task
TASKS=(
    "widowx_carrot_on_plate"
    "widowx_put_eggplant_in_basket"
    "widowx_spoon_on_towel"
    "widowx_stack_cube"
)

N_EVAL_EPISODE=240   # octo simpler runs 3 seeds with 24 configs each, here we run 10 seeds

for TASK in ${TASKS[@]}; do

    INTERNVL=1 DEBUG_448=1 PYTHONPATH=. \
        python3 \
        scripts/run.py \
        --config-name=bridge_internvl_448.yaml \ # modify this as you need
        --config-path=../config/eval \
        device=cuda:0 \
        seed=42 \
        n_eval_episode=$N_EVAL_EPISODE \
        n_video=$N_EVAL_EPISODE \
        env.task=$TASK \
        horizon_steps=4 \
        act_steps=4 \
        use_bf16=False \
        use_torch_compile=True \
        name=bridge_beta \
        checkpoint_path=your_ckpt_path
done
