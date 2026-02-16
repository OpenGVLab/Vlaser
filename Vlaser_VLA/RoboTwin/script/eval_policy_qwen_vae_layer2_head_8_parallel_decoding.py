import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import subprocess
sys.path.append(sys.path[0]+"/..")
sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
import json

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args.task_name
    task_config = usr_args.task_config
    ckpt_setting = usr_args.ckpt_setting
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args.policy_name
    instruction_type = usr_args.instruction_type
    save_dir = None
    video_save_dir = None
    video_size = None

    # get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["checkpoint_id"] = usr_args.checkpoint_id
    args["result_save_folder"] = usr_args.result_save_folder
    args['ckpt_config'] = usr_args.ckpt_config
    args['ckpt_config_id'] = usr_args.ckpt_config_id


    args["pred_img"] = False


    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"{args['result_save_folder']}/{task_config}/{args['ckpt_config']}/checkpoint-{args['ckpt_config_id']}/{args['task_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)

    vla_model_path = os.path.join(os.getcwd(), f"{args['ckpt_config']}/checkpoint-{args['ckpt_config_id']}/unwrapped_model")

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    # usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    # usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args.seed

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 50
    topk = 1

    # model = get_model(usr_args)
    st_seed, suc_num,suc_num_50, vla_model_path = eval_policy(task_name,
                                                   TASK_ENV,
                                                   args,
                                                   # model,
                                                   st_seed,
                                                   test_num=test_num,
                                                   video_size=video_size,
                                                   instruction_type=instruction_type)
    # suc_nums.append(suc_num)

    # topk_success_rate = sorted(suc_nums, reverse=True)[:topk]





    # file_path = os.path.join(save_dir, f"{vla_model_path.split('/')[-4]}_result.json")
    file_path = os.path.join(save_dir, f"{usr_args.seed}_result.json")
    cur_data = {
        "model_path": vla_model_path,
        "Timestamp": current_time,
        "Instruction Type": instruction_type,
        "task_name": task_name,
        "score_50": suc_num_50 / 50,
        "score": suc_num / test_num
    }
    # print(cur_data)
    write_data(file_path, cur_data)
    print(f"Data has been saved to {file_path}")
    # return task_reward
    exit()


def write_data(file_path, new_data):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 解析已有的 JSON 数据
            file_data = json.load(file)

            # 检查是否存在一个列表，如果存在，则追加数据
            if isinstance(file_data, list):
                file_data.append(new_data)
            else:
                # 如果不是列表，可以创建一个新的列表或处理其他数据结构
                raise ValueError("JSON root should be a list.")
    except FileNotFoundError:
        # 如果文件不存在，则创建一个新的文件并初始化为列表
        file_data = [new_data]

    # 将更新后的数据写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        # 确保 JSON 格式化输出，使文件更易读
        json.dump(file_data, file, ensure_ascii=False, indent=4)







def eval_policy(task_name,
                TASK_ENV,
                args,
                # model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    cur_test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # head_num = 8
    vla_model_path = os.path.join(os.getcwd(), f"{args['ckpt_config']}/checkpoint-{args['ckpt_config_id']}/unwrapped_model")
    args["hidden_size"] = 1536
    args["action_dim"] = 14
    args["num_action_chunks"] = 8

    suc_num_50 = 999

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [("None", episode_info["info"])]  # 这里的None主要是为了匹配后面的我自己修改的generate_episode_descriptions
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])

        print(f"the instruction for {args['task_name']} is {instruction}")

        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction


        succ = False


        TASK_ENV.init_policy_qwen_fix_text_length_vae_layer2_head_parallel_decoding(args,
                                                vla_model_path)
        TASK_ENV.apply_qwen_fix_text_length(args, cur_test_num, mode="vae", parallel_decoding=True)

        if TASK_ENV.eval_success:
            succ = True


        # # reset_func(model)
        # while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        #     observation = TASK_ENV.get_obs()
        #     eval_func(TASK_ENV, model, observation)
        #     if TASK_ENV.eval_success:
        #         succ = True
        #         break
        # # task_total_reward += TASK_ENV.episode_score
        # if TASK_ENV.eval_video_path is not None:
        #     TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\n\033[92mSuccess!\033[0m")
        else:
            print("\n\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        cur_test_num += 1
        if cur_test_num == 50:
            suc_num_50 = TASK_ENV.suc
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{cur_test_num}\033[0m => \033[95m{round(TASK_ENV.suc/cur_test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc , suc_num_50 , vla_model_path


def parse_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default="click_bell")
    parser.add_argument("--task_config", type=str, default="demo_randomized_clean_table")
    parser.add_argument("--result_save_folder", type=str, default="eval_result_pipper_randomized_clean_table_A800")

    parser.add_argument("--ckpt_setting", type=str, default="qwen_demo_clean")
    parser.add_argument("--policy_name", type=str, default="vae_mine")
    parser.add_argument("--ckpt_config", type=str, default="internvl3_2B")
    parser.add_argument("--ckpt_config_id", type=str, default="10000")
    parser.add_argument("--instruction_type", type=str, default="unseen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")

    parser.add_argument("--checkpoint_id", type=str, default="75000")


    # checkpoint_num = usr_args['checkpoint_num']

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = usr_args.gpu
    main(usr_args)
