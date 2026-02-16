
accelerate launch \
--config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml \
--main_process_port=8980 
vla-scripts/train_mine_new.py config=configs/training_based_on_internvl3_2B.yaml