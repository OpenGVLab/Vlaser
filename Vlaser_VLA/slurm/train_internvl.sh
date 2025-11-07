#!/bin/bash

export WANDB__SERVICE_WAIT=300

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)



master_addr=$(scontrol show hostname ${node_list} | head -n1)

echo $master_addr
ps -ux | grep python | awk '{print $2}' | xargs kill
INTERNVL=1 IMAGE_448=1 PYTHONPATH=. HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank=$SLURM_PROCID --master_addr=$master_addr \
  scripts/run.py \
  --config-name=fractal_internvl \
  action_lr=0.00005 \
  vlm_lr=0.00005 \
  n_epochs=10 \
  flow_sampling=beta \
  global_batch_size=1024 \
  n_nodes=4 \
  use_torch_compile=True \
  use_bf16=True \
  use_amp=True \
  time_max_period=100.0 \
  action_expert_rope_theta=100.0 \
  cond_steps=1 \
  horizon_steps=4 \
  name=train \
  max_seq_len=384 \
  data.train.shuffle_buffer_size=200000 \
  pretrained_model_path=your_model_path