"""
Launcher for all experiments.

"""

import logging
import math
import os
import random
import sys

import hydra
import numpy as np
import pretty_errors
import torch
from omegaconf import OmegaConf, open_dict
import time
import subprocess
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.distributed as dist
from torch.distributed import init_process_group
import tensorflow_io as tfio
# dummy
print(pretty_errors.__version__)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def init_distributed_mode():
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = rank % torch.cuda.device_count()

    world_size = int(os.environ["SLURM_NTASKS"])
    local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    if "MASTER_PORT" not in os.environ:
        port = 22384

        print(f'MASTER_PORT = {port}')
        os.environ["MASTER_PORT"] = str(port)

        time.sleep(3)

    node_list = os.environ['SLURM_STEP_NODELIST']
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["GROUP_RANK"] = str(0)
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")

def _main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # figure out the current gpu
    multi_gpu = torch.cuda.device_count() > 1 or cfg.get("n_nodes", 1) > 1
    if multi_gpu:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            init_process_group(backend="nccl")

        init_distributed_mode()
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    with open_dict(cfg):
        cfg.gpu_id = gpu_id
        cfg.multi_gpu = multi_gpu

    # seeding
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()

    if multi_gpu:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/train"),
    config_name="bridge.yaml",
)  # defaults
def main(cfg: OmegaConf):
    _main(cfg)


if __name__ == "__main__":
    main()
