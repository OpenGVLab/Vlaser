set -x

# 命令行参数: $1=模型类型(bridge|fractal), $2=任务类型(general|spatial|grounding)
MODEL_TYPE=${1:-bridge}
TASK_TYPE=${2:-general}

GPUS=${GPUS:-32}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=${GPUS} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python scripts/run.py \
  --config-name=${MODEL_TYPE}-${TASK_TYPE} \
  debug=True \
  wandb=null \
  log_dir=${MODEL_TYPE}/qa_${TASK_TYPE}/ \
  global_batch_size=256 \
  per_device_batch_size=8 \
  flow_sampling=beta \
  data.train.shuffle_buffer_size=10000 \
  data.train.num_parallel_calls=10 \
  eval_freq=32 \
  eval_size=64 \
  save_model_freq=16 \
  save_model_start=0 \
  lora=False \
  quantize=False \
  use_torch_compile=False \
  use_bf16=True \
  use_amp=True \
  use_ema=True \
  ema_decay=0.99 \
  ema_device=cuda \
  use_swa=False \
  swa_start=0 \
  swa_freq=2 \
  swa_device=cpu \
  action_lr_scheduler.warmup_steps=0 \
  vlm_lr_scheduler.warmup_steps=0 \
  2>&1 | tee "${MODEL_TYPE}/qa_${TASK_TYPE}/training_log.txt"
