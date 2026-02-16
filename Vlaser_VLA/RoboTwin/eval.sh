# 命令行参数: $1=模型类型(all|general|spatial|grounding)
MODEL_TYPE=${1:-all}

python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name beat_block_hammer --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name click_bell --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name handover_mic --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name move_can_pot --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name move_pillbottle_pad --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name move_playingcard_away --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750

python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name pick_diverse_bottles --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name place_mouse_pad --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name place_container_plate --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name place_phone_stand --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name place_burger_fries --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750
python script/eval_policy_qwen_vae_layer2_head_8_parallel_decoding.py --task_name shake_bottle --gpu 0 --ckpt_config vlaser_ckpt/${MODEL_TYPE} --ckpt_config_id 34750