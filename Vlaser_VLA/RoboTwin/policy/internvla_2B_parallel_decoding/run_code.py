import os
import time
import subprocess

# 要执行的命令
CMD_TO_EXECUTE = "accelerate launch --config_file accelerate_configs/8_gpus_deepspeed_zero2.yaml --main_process_port=8880 vla-scripts/train_mine_new.py config=configs/training_based_on_gpt2_0.1B.yaml"

# 显存占用阈值（1GB）
MEMORY_THRESHOLD_MB = 1024

def get_gpu_memory_usage():
    """获取每个 GPU 的显存使用情况（单位：MB）。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        memory_usages = result.stdout.strip().split('\n')
        return [int(usage) for usage in memory_usages]
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return []

def check_and_execute():
    while True:
        memory_usages = get_gpu_memory_usage()

        if all(memory_usage < MEMORY_THRESHOLD_MB for memory_usage in memory_usages):
            # 所有显卡显存占用都小于1GB，执行命令
            # print("Executing command...")
            os.system(CMD_TO_EXECUTE)
            break
        else:
            # 否则等待一分钟后再检测
            # print("Not all GPU memory usage is below 1GB. Waiting for 1 minute...")
            time.sleep(60)

if __name__ == "__main__":
    check_and_execute()