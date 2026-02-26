"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""
import os
# Sane Defaults




os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['NCCL_ALGO'] = 'Tree' # for A800


import sys
sys.path.append(sys.path[0]+'/..')


import json
import logging
import math

import torch.nn as nn
import shutil
import time
from pathlib import Path
from typing import Type, Union
from tqdm import tqdm
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from utils import get_config, flatten_omega_conf, AverageMeter, get_action_masks
config = get_config()
accelerator = Accelerator(
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    mixed_precision=config.training.mixed_precision,
    log_with="wandb",
    project_dir=str(Path(config.experiment.output_dir) / "logs"),
    split_batches=True,
)

from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from torch.utils.data.distributed import DistributedSampler

from data_utils.materialize import get_vla_dataset_and_collator

from models import Showo, UnitModel

from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
from internvl.model.action_heads import L1RegressionActionHead

from einops import rearrange
try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

from datetime import date
import random

import torch
os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    module_args: dict,
):
    """
    Initializes a module.
    Checkpoint loading is handled separately.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): A descriptive name for the module.
        module_args (dict): Args for initializing the module.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)

    return module


def train():
    #########################
    # SETUP Accelerator     #
    #########################
    # config = get_config()
    # if config.training.parallel_decoding:
    #     from models import Qwen2ForCausalLM
    # else:
    #     from transformers import Qwen2ForCausalLM


    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    #     mixed_precision=config.training.mixed_precision,
    #     log_with="wandb",
    #     project_dir=config.experiment.logging_dir,
    #     split_batches=True,
    # )
    total_batch_size_per_gpu = config.training.per_device_batch_size


    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)
        seed_everything(config.training.seed)



    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = False
        date_str = date.today().strftime("%Y-%m-%d")

        # 只获取当前日期
        run_id = wandb.util.generate_id()+'_'+date_str+'batch_size_32_no_vision'

        config.wandb.run_id = run_id
        os.environ["WANDB_API_KEY"] = config.wandb.api_key
        wandb_init_kwargs = dict(
            name=os.path.abspath(__file__).split('/')[-2]+'_'+date_str,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)


    # assert config.dataset.step_window_size%config.dataset.action_chunk_size==0, logger.info("step_window_size must be a multiple of action_chunks")

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")
    # vlm, tokenizer = load(model_args, data_args, cfg.resume)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path, padding_side="left")

    # tokenizer.add_tokens(token_list, special_tokens=True) v ???
    tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
    tokenizer.add_special_tokens({'additional_special_tokens':
                                      ["<soi>", "<eoi>", "<sot>", "<eot>", # "<soa>", "<eoa>",
                                       "<left_arm_soa>", "<left_arm_eoa>", "<right_arm_soa>", "<right_arm_eoa>",
                                       "<left_arm_sost>", "<left_arm_eost>", "<right_arm_sost>", "<right_arm_eost>",]})  # sost是 start of state



    # ----------------------------------- TODO Delete ----------------------------------
    # 根据codebook_size添加图像相关的tokens, <img_0> ~ <img_codebook_size-1>
    # img_tokens = [f"<img_{i}>" for i in range(1, config.model.vq_model.codebook_size+1)]
    #act_tokens = [f"<act_{i}>" for i in range(1, config.dataset.action_vocab_size+1)]
    state_tokens = [f"<state_{i}>" for i in range(1, config.dataset.state_vocab_size+1)]

    #state_tokens.extend(act_tokens)
    # img_tokens.extend(state_tokens)

    tokenizer.add_tokens(state_tokens)
    
    # ----------------------------------- TODO Delete ----------------------------------
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')
    tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids('<soi>')
    tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids('<eoi>')
    tokenizer.sot_token_id = tokenizer.convert_tokens_to_ids('<sot>')
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids('<eot>')
    tokenizer.soa_token_id = tokenizer.convert_tokens_to_ids('<soa>')
    tokenizer.eoa_token_id = tokenizer.convert_tokens_to_ids('<eoa>')
    tokenizer.left_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_soa>')
    tokenizer.left_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eoa>')
    tokenizer.right_arm_soa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_soa>')
    tokenizer.right_arm_eoa_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eoa>')


    tokenizer.left_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_sost>')
    tokenizer.left_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eost>')
    tokenizer.right_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_sost>')
    tokenizer.right_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eost>')


    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.ignore_id = -100

    # img_token_start_idx = tokenizer(["<img_1>"]).input_ids[0][0]
    act_token_start_idx = tokenizer(["<act_1>"]).input_ids[0][0]
    state_token_start_idx = tokenizer(["<state_1>"]).input_ids[0][0]


    from internvl.model.load import load
    from internvl.training.arguments import ModelArguments, DataTrainingArguments
    
    # Map TrainConfig parameters to ModelArguments
    model_args = ModelArguments(
        model_name_or_path=config.model.model_name_or_path if not config.model.resume else config.model.pretrained_checkpoint,
        mlp_path=config.model.mlp_path,
        vision_select_layer=config.model.vision_select_layer,
        drop_path_rate=config.model.drop_path_rate,
        grad_checkpoint=config.model.grad_checkpoint,
        use_fast_tokenizer=config.model.use_fast_tokenizer,
        ps_version=config.model.ps_version,
    )
    
    # Map TrainConfig parameters to DataTrainingArguments
    data_args = DataTrainingArguments(
        max_seq_length=config.training.max_seq_length,
        force_image_size=config.training.force_image_size,
        down_sample_ratio=config.training.down_sample_ratio,
        conv_style=config.training.conv_style,
        dynamic_image_size=config.training.dynamic_image_size,
        use_thumbnail=config.training.use_thumbnail,
        min_dynamic_patch=config.training.min_dynamic_patch,
        max_dynamic_patch=config.training.max_dynamic_patch,
        normalize_type=config.training.normalize_type,
    )

    accelerator_device = accelerator.device
    vlm = load(model_args, data_args)
    vlm.to(accelerator_device, dtype=torch.bfloat16)
    # ViT do not need to train

    # vlm = Qwen2ForCausalLM.from_pretrained(
    #     config.model.showo.vlm_model_path,
    #     torch_dtype=torch.bfloat16,
    #     # low_cpu_mem_usage=False,  # if set low_cpu_mem_usage to True, there will be something wrong with resize_token_embeddings https://github.com/huggingface/accelerate/issues/1620#issuecomment-2413317047
    #     attn_implementation="flash_attention_2" #"eager", # "flash_attention_2"
    # )
    vlm.language_model.resize_token_embeddings(len(tokenizer))
    hidden_size = vlm.language_model.config.hidden_size
    vla = Showo(vlm)
    
    

    # instantiate continuous action head for L1 regression
    # TODO check config
    action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        {"input_dim": hidden_size, "hidden_dim": hidden_size, "action_dim": 14},
    )
    model = UnitModel(
        vla,
        # ActionProjector(input_dim=config.dataset.codebook_dim//config.dataset.num_codebooks, output_dim=config.model.showo.hidden_size),  # here the input dim is codebook_dim/num_codebooks
        #ImageProjector(input_dim=1152, output_dim=config.model.showo.hidden_size),
        action_head,
    )

    model.to(accelerator.device, dtype=torch.bfloat16)
    model.requires_grad_(True)



    os.makedirs(run_dir := (Path("runs") / config.wandb.run_id), exist_ok=True)
    os.makedirs(Path("runs") / config.wandb.run_id / "checkpoints", exist_ok=True)



    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable")




    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    # Instantiate optimizer
    trainable_params = [param for param in vlm.language_model.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            trainable_params,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    logger.info(f"Creating Dataset with Mixture `{config.dataset.data_mix}`")
    # from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

    # Get VLA Dataset & Collator
    with accelerator.main_process_first():  # cache data is generated by the first process, then the other ones just pick the cache.

        vla_dataset, collator = get_vla_dataset_and_collator(
            Path(config.dataset.data_root_dir),
            config.dataset.data_mix,
            # image_transform=vlm.vision_backbone.get_image_transform(),
            tokenizer=tokenizer,
            # prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
            # default_image_resolution=(3, 224, 224),
            shuffle_buffer_size=config.dataset.shuffle_buffer_size,
            image_aug=config.dataset.image_aug,
            img_history_size=config.dataset.img_history_size,
            action_chunk_size=config.dataset.action_chunk_size,
            act_token_start_idx=act_token_start_idx,
            state_min=config.dataset.state_min,
            state_max=config.dataset.state_max,
            state_vocab_size=config.dataset.state_vocab_size,
            state_token_start_idx=state_token_start_idx,
            image_size=config.dataset.image_size,
            target_length=config.training.target_length,
            instruction_path=config.dataset.instruction_path
        )

    # # Save dataset statistics for de-normalization at inference time
    # if overwatch.is_rank_zero():
    #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    datasampler = DistributedSampler(vla_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True,)
    dataloader = DataLoader(
        vla_dataset,
        batch_size=config.training.per_device_batch_size,
        sampler=datasampler,
        collate_fn=collator,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        # worker_init_fn=worker_init_fn,
    )

    # total_batch_size = (
    #         config.training.per_device_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    # )
    num_update_steps_per_epoch = math.ceil(len(dataloader) / (config.training.gradient_accumulation_steps))
    # num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch*num_train_epochs
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps*accelerator.num_processes,
        num_warmup_steps=int(config.lr_scheduler.params.warmup_ratio*max_train_steps)*accelerator.num_processes,
    )

    # Combine these dataloaders into a single iterable model
    iterables = {
        # "t2i_flow": train_dataloader_t2i,
        # "lm_flow": train_dataloader_lm,
        "vla_flow": dataloader,
    }
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)


    # vq_model.to(images_feat_dtype)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    accelerator_device = accelerator.device

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in tqdm(combined_dataloader, total=num_update_steps_per_epoch):
            # 在这里处理数据
            # pixel_values_steps: [batch, step_window_size, 224,224,3]
            batch = batch["vla_flow"]

            (
             pixel_values_steps,  # shape: [batch, img_history_size, 224,224,3]
             input_text_ids,  # shape: [batch, len]
             state_ids,  # shape: [batch, 14]
             action_steps_values,  # shape: [batch, img_history_size, len]
             # action_end_idx  # shape: [batch, img_history_size+1], +1是因为有起始index:0
             ) =  \
                (batch["pixel_values_steps"],
                 batch["input_text_ids"],
                 batch["state_ids"],
                 batch["action_steps_ids"],
                )
                
            action_chunk_size = config.dataset.action_chunk_size
            action_dim = action_steps_values.shape[-1]
            # 我们要从pixel_values_steps, input_text_ids选出index为0, 1*action_chunk_size, (step_num-1)*action_chunk_size,

            batch_size = pixel_values_steps.size(0)
            img_input_len = pixel_values_steps.size(1)

            step_num = max(1, img_input_len-1)  # 如果不预测img (此时img_input_len=1), step_num=1； 如果预测img (此时img_input_len=2), step_num=1
            pixel_values_steps = pixel_values_steps.reshape(batch_size * img_input_len, *(pixel_values_steps.size()[2:])).to(
                accelerator_device, non_blocking=True).to(dtype=torch.bfloat16)  # [batch, img_history_size, 3, 224,224] -> [batch*img_history_size,3, 224,224]

            # 我们要得到每个input text的长度
            input_text_lengths = input_text_ids.ne(tokenizer.pad_token_id)
            input_text_lengths = torch.sum(input_text_lengths, dim=1)

            input_text_ids = input_text_ids.to(accelerator_device, non_blocking=True)
            action_steps_values = action_steps_values.to(accelerator_device, non_blocking=True)  # [batch, img_history_size, 8, 14]

            tokenizer.left_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_sost>')
            tokenizer.left_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<left_arm_eost>')
            tokenizer.right_arm_sost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_sost>')
            tokenizer.right_arm_eost_token_id = tokenizer.convert_tokens_to_ids('<right_arm_eost>')

            state_ids = state_ids.squeeze()
            left_arm_state_start = torch.ones((batch_size, 1)).long() * tokenizer.left_arm_sost_token_id
            left_arm_state_end = torch.ones((batch_size, 1)).long() * tokenizer.left_arm_eost_token_id
            right_arm_state_start = torch.ones((batch_size, 1)).long() * tokenizer.right_arm_sost_token_id
            right_arm_state_end = torch.ones((batch_size, 1)).long() * tokenizer.right_arm_eost_token_id
            # 增加state分隔符
            state_ids = torch.cat(
                [
                    left_arm_state_start,
                    state_ids[:, :7],
                    left_arm_state_end,
                    right_arm_state_start,
                    state_ids[:, 7:],
                    right_arm_state_end
                ], dim=1
            )
            state_ids = state_ids.to(accelerator_device, non_blocking=True)
            with torch.no_grad():
                
                # img_embeddings = img_encoder(pixel_values_steps, output_hidden_states=True).hidden_states[-2]
                img_embeddings = model.vla.model.extract_feature(pixel_values_steps)


            # internvlm has a projector  
            # img_embeddings = model.image_projector(img_embeddings)


            text_embeddings = model.vla.model.language_model.model.embed_tokens(input_text_ids)  # [batch, len, 896]
            img_embeddings = img_embeddings.contiguous().view(
                batch_size, img_input_len, -1,
                text_embeddings.shape[-1])  # [batch*img_input_len, 256, 896]->[batch, img_input_len, 256, 896]

            # left_act_embeddings = model.action_projector(left_action_features.to(dtype=torch.bfloat16))  # [batch, step_num, 4, 8, 896]
            # right_act_embeddings = model.action_projector(right_action_features.to(dtype=torch.bfloat16))  # [batch, step_num, 4, 8, 896]

            shape_features = (batch_size, action_chunk_size * action_dim // 2, hidden_size)
            
            # print(action_chunk_size, action_dim // 2, hidden_size) 
            
            left_act_embeddings = torch.zeros(shape_features, dtype=torch.bfloat16).to(accelerator_device)
            right_act_embeddings = torch.zeros(shape_features, dtype=torch.bfloat16).to(accelerator_device)
            
            

            state_embeddings = model.vla.model.language_model.model.embed_tokens(state_ids) # [batch, 14, 896]

            # state_embeddings = model.state_projector(state_ids.to(dtype=torch.bfloat16)) # [batch, 896], 这里的state_ids实际是一个14维向量: [1, 14]
            observation_img_embeddings = img_embeddings[:, 0]  # [batch, 256, 896]


            text_start_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.sot_token_id)
            text_end_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.eot_token_id)

            img_start_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.soi_token_id)
            img_end_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.eoi_token_id)


            instance_start_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.bos_token_id)
            instance_end_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones(batch_size, 1).long().to(accelerator_device) * tokenizer.eos_token_id)

            left_act_start_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones((left_act_embeddings.size()[0], 1)).long().to(accelerator_device) * tokenizer.left_arm_soa_token_id)
            left_act_end_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones((left_act_embeddings.size()[0], 1)).long().to(accelerator_device) * tokenizer.left_arm_eoa_token_id)
            right_act_start_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones((left_act_embeddings.size()[0], 1)).long().to(accelerator_device) * tokenizer.right_arm_soa_token_id)
            right_act_end_embeddings = model.vla.model.language_model.model.embed_tokens(
                torch.ones((left_act_embeddings.size()[0], 1)).long().to(accelerator_device) * tokenizer.right_arm_eoa_token_id)

            # print(f"!!! DEBUG: left_act_start_embeddings: {left_act_start_embeddings.shape}")
            # print(f"!!! DEBUG: left_act_end_embeddings: {left_act_end_embeddings.shape}")
            # print(f"!!! DEBUG: right_act_start_embeddings: {right_act_start_embeddings.shape}")
            # print(f"!!! DEBUG: right_act_end_embeddings: {right_act_end_embeddings.shape}")

            # 对batch里的每条数据单独出来
            # 现在我们有 "text_tokens padded_token", "image_tokens", "action_tokens"
            # 要变成"text_tokens image_tokens action_tokens padded_token", 这里将text_tokens后的padded_tokens调整到了最后面


            if config.training.parallel_decoding:
                left_act_embeddings = left_act_embeddings*0
                right_act_embeddings = right_act_embeddings*0

            # 不含future img, 即step_num=1
            left_image_action_interleaved_embeddings = torch.cat([
                left_act_start_embeddings,  # (batch_size, 1, hidden_size)
                left_act_embeddings,  # (batch_size, action_chunk_size * action_dim / 2, hidden_size)
                left_act_end_embeddings,  # (batch_size, 1, hidden_size)
            ], dim=1).contiguous().view(batch_size, -1, text_embeddings.shape[-1])   #[batch, l, 896],
            
            # print('left_image_action_interleaved_embeddings', left_image_action_interleaved_embeddings.shape)

            right_image_action_interleaved_embeddings = torch.cat([
                right_act_start_embeddings,   # (batch_size, 1, hidden_size)
                right_act_embeddings,  # (batch_size, action_chunk_size * action_dim / 2, hidden_size)
                right_act_end_embeddings,   # (batch_size, 1, hidden_size)
            ], dim=1).contiguous().view(batch_size, -1, text_embeddings.shape[-1])  # [batch, l, 896],
            # print('right_image_action_interleaved_embeddings', right_image_action_interleaved_embeddings.shape)

            image_action_interleaved_embeddings = torch.cat((
                left_image_action_interleaved_embeddings,
                right_image_action_interleaved_embeddings
            ), dim=1)


            input_embeddings = torch.cat([
                instance_start_embeddings,

                img_start_embeddings,
                observation_img_embeddings,
                img_end_embeddings,
                
                text_start_embeddings,
                text_embeddings,
                text_end_embeddings,
                
                state_embeddings,

                image_action_interleaved_embeddings,
                instance_end_embeddings,  # eos
            ], dim=1)  # [batch, len, dim]

            # attention_mask = torch.ones(*(input_embeddings.size()[:2])).to(accelerator_device)
            # for b in range(batch_size):
            #     pad_start = input_text_lengths[b]+2 # 2是指前缀的instance_start和text_start
            #     pad_end = pad_start + input_text_ids.size(-1) - input_text_lengths[b]
            #     attention_mask[b, pad_start:pad_end] = 0
            
            # TODO check attention_mask
            text_section_start_index = 4 + observation_img_embeddings.shape[1]

            text_attention_mask = input_text_ids.ne(tokenizer.pad_token_id).long()

            attention_mask = torch.ones_like(input_embeddings[..., 0], dtype=torch.long)
            attention_mask[:, text_section_start_index : text_section_start_index + input_text_ids.shape[1]] = text_attention_mask

            # attention_mask = labels.ne(tokenizer.pad_token_id)  # 这里我们不像show-o那样直接搞一个下三角矩阵，我还想用flash-attn加速训练
            with accelerator.accumulate(model):
                output: CausalLMOutputWithPast = model.vla(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                )
                
                # ground_truth_token_ids = batch["labels"][:, 1:].to(accelerator_device)
                # action_steps_values# [batch, 1, 8, 14]
                current_action_mask, next_actions_mask = get_action_masks(input_embeddings.shape, observation_img_embeddings.shape,
                                                                            text_embeddings.shape, state_embeddings.shape,
                                                                            left_image_action_interleaved_embeddings.shape, 
                                                                            right_image_action_interleaved_embeddings.shape,
                                                                            action_dim, accelerator_device
                                                                            )

                last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
                # Get hidden states for text portion of prompt+response (after the vision patches)
                # <|im_end|>\n
                # text_hidden_states = last_hidden_states[:, :-1]
                # Get hidden states for action portion of response
                # batch_size = batch["input_ids"].shape[0]
                # print(f"!!! DEBUG: last_hidden_states.shape: {last_hidden_states.shape}")
                actions_hidden_states = (
                    last_hidden_states[current_action_mask | next_actions_mask]
                    .reshape(batch_size, action_chunk_size * action_dim, -1)
                    .to(torch.bfloat16)
                )  # (B, act_chunk_len, D)

                # Predict action
                predicted_actions = action_head.predict_action(actions_hidden_states) # 
                # Get full L1 loss
                loss = torch.nn.L1Loss()(predicted_actions, action_steps_values.squeeze(dim=1))


                # Get detailed L1 losses for logging
                # should_log_l1_loss = use_l1_regression
                # if should_log_l1_loss:
                #     ground_truth_curr_action = ground_truth_actions[:, 0]
                #     predicted_curr_action = predicted_actions[:, 0]
                #     ground_truth_next_actions = ground_truth_actions[:, 1:]
                #     predicted_next_actions = predicted_actions[:, 1:]
                #     curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
                #     next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
                #     metrics.update(
                #         {
                #             "curr_action_l1_loss": curr_action_l1_loss.item(),
                #             "next_actions_l1_loss": next_actions_l1_loss.item(),
                #         }
                #     )


                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                # print([name for name in model.named_parameters()])

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    total_norm = log_grad_norm(model, accelerator, global_step + 1)
                    logger.info(f"total_norm is {total_norm:0.2f}")

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss": loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "epoch: cur/total": f"{epoch}/{num_train_epochs}",

                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step/Total step: {global_step + 1}/{max_train_steps} "
                        f"Loss_mmu: {loss.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"epoch: cur/total {epoch}/{num_train_epochs}",

                    )

                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model,action_head, tokenizer, config, accelerator, global_step + 1)
            global_step += 1

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, action_head, tokenizer, config, accelerator, global_step)



    accelerator.end_training()

def save_checkpoint(model, action_head, tokenizer, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"



    if accelerator.is_main_process:
        vla_model = accelerator.unwrap_model(model.vla.model)
        vla_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model.vla.model),
            safe_serialization=False
        )

        # action_projector_model = accelerator.unwrap_model(model.action_projector)
        # state_projector_model = accelerator.unwrap_model(model.state_projector)
        checkpoint = {
            'action_head': accelerator.get_state_dict(action_head),
            # 'state_projector': accelerator.get_state_dict(model.image_projector),  # TODO 后面记得改成"image_projector"
        }
        torch.save(checkpoint, save_path / "unwrapped_model/projector.pth")


        tokenizer.save_pretrained(save_path / "unwrapped_model",)
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")



def log_grad_norm(model, accelerator, global_step):
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_norm = total_grad_norm ** 0.5
    accelerator.log({"total_norm": f"{total_norm:0.2f}"}, step=global_step)
    return total_norm

if __name__ == "__main__":
    train()
