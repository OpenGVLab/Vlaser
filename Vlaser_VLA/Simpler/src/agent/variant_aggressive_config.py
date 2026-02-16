import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
import gymnasium as gym
import os

def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))

def find_suitable_args(task):


    BASIC_ARGS = {
        "policy-model" : "rt1",
        "policy-setup" : "google_robot",
        "ckpt-path" : None,
        "additional-env-save-tags" : None,
        "env-name": None,
        "scene-name" : None,
        "enable-raytracing" : False,
        "robot": "google_robot_static",
        "obs-camera-name" : None,
        "action-scale" : 1.0,
        "control-freq" : 3,
        "sim-freq": 513,
        "max-episode-steps" : 80,
        "rgb-overlay-path" : None,
        "robot-init-x-range" : [0.35, 0.35, 1],
        "robot-init-y-range" : [0.20, 0.20, 1],
        "robot-init-rot-quat-center" : [1, 0, 0, 0],
        "robot-init-rot-rpy-range" : [0, 0, 1, 0, 0, 1, 0, 0, 1],
        "obj-variation-mode" : "xy",
        "obj-episode-range" : [0,60],
        "obj-init-x-range" : [-0.35, -0.12, 5],
        "obj-init-y-range" : [-0.02, 0.42, 5],
        "additional-env-build-kwargs-variants" : None,
        "logging-dir" : "./results",
        "tf-memory-limit" : 3072,
        "octo-init-rng" : 0,   
    }


    COKE_CAN = [
        {
            # 基础配置（单一场景）
            "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
            "scene-name": ["google_pick_coke_can_1_v4"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],  # 修正了符号
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            # 等价于脚本里三种朝向循环
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True},
                {"upright": True},
                {"laid_vertically": True},
            ],
        },
        {
            # 桌面纹理（两个不同 scene）
            "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
            "scene-name": [
                "Baked_sc1_staging_objaverse_cabinet1_h870",
                "Baked_sc1_staging_objaverse_cabinet2_h870",
            ],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True},
                {"upright": True},
                {"laid_vertically": True},
            ],
        },
        {
            # 含干扰物（普通 + more 两种干扰强度）
            "env-name": ["GraspSingleOpenedCokeCanDistractorInScene-v0"],
            "scene-name": ["google_pick_coke_can_1_v4"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            # 等价于两次调用：一次仅朝向；一次朝向+more
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True},
                {"upright": True},
                {"laid_vertically": True},
                {"lr_switch": True, "distractor_config": "more"},
                {"upright": True, "distractor_config": "more"},
                {"laid_vertically": True, "distractor_config": "more"},
            ],
        },
        {
            # 背景变化（两个 alt 背景）
            "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
            "scene-name": [
                "google_pick_coke_can_1_v4_alt_background",
                "google_pick_coke_can_1_v4_alt_background_2",
            ],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True},
                {"upright": True},
                {"laid_vertically": True},
            ],
        },
        {
            # 光照变暗
            "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
            "scene-name": ["google_pick_coke_can_1_v4"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True, "slightly_darker_lighting": True},
                {"upright": True, "slightly_darker_lighting": True},
                {"laid_vertically": True, "slightly_darker_lighting": True},
            ],
        },
        {
            # 光照变亮
            "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
            "scene-name": ["google_pick_coke_can_1_v4"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True, "slightly_brighter_lighting": True},
                {"upright": True, "slightly_brighter_lighting": True},
                {"laid_vertically": True, "slightly_brighter_lighting": True},
            ],
        },
        {
            # 摄像头角度（两个不同的 env_name）
            "env-name": [
                "GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0",
                "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0",
            ],
            "scene-name": ["google_pick_coke_can_1_v4"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.20, 0.20, 1],
            "obj-init-x-range": [-0.35, -0.12, 5],
            "obj-init-y-range": [-0.02, 0.42, 5],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "additional-env-build-kwargs-variants": [
                {"lr_switch": True},
                {"upright": True},
                {"laid_vertically": True},
            ],
        },
    ]


    DRAWER = [

        {
            # 基础配置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": [
                "OpenTopDrawerCustomInScene-v0",
                "OpenMiddleDrawerCustomInScene-v0",
                "OpenBottomDrawerCustomInScene-v0",
                "CloseTopDrawerCustomInScene-v0",
                "CloseMiddleDrawerCustomInScene-v0",
                "CloseBottomDrawerCustomInScene-v0",
            ],
            "enable-raytracing": True,  # 启用光线追踪
            "max-episode-steps" : 113,
            # 机器人的初始化配置
            "robot-init-x-range": [0.65, 0.85, 3],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [0, 0, 1],
            "obj-init-y-range": [0, 0, 1],
        },
        {
            # 背景场景
            "scene-name": ["modern_bedroom_no_roof", "modern_office_no_roof"],  # 例如，添加其他背景场景
            "env-name": [
                "OpenTopDrawerCustomInScene-v0",
                "OpenMiddleDrawerCustomInScene-v0",
                "OpenBottomDrawerCustomInScene-v0",
                "CloseTopDrawerCustomInScene-v0",
                "CloseMiddleDrawerCustomInScene-v0",
                "CloseBottomDrawerCustomInScene-v0",
            ],
            "max-episode-steps" : 113,
            "additional-env-build-kwargs-variants": [{"shader_dir": "rt"}],  # 额外的环境构建参数

            # 机器人的初始化配置
            "robot-init-x-range": [0.65, 0.85, 3],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [0, 0, 1],
            "obj-init-y-range": [0, 0, 1],
        },
        {
            # 光照设置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": [
                "OpenTopDrawerCustomInScene-v0",
                "OpenMiddleDrawerCustomInScene-v0",
                "OpenBottomDrawerCustomInScene-v0",
                "CloseTopDrawerCustomInScene-v0",
                "CloseMiddleDrawerCustomInScene-v0",
                "CloseBottomDrawerCustomInScene-v0",
            ],
            "max-episode-steps" : 113,
            "additional-env-build-kwargs-variants": [
                {"shader_dir": "rt", "light_mode": "brighter"},
                {"shader_dir": "rt", "light_mode": "darker"},
            ],  # 含光照模式的配置

            # 机器人的初始化配置
            "robot-init-x-range": [0.65, 0.85, 3],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [0, 0, 1],
            "obj-init-y-range": [0, 0, 1],
        },
        {
            # 新站点配置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": [
                "OpenTopDrawerCustomInScene-v0",
                "OpenMiddleDrawerCustomInScene-v0",
                "OpenBottomDrawerCustomInScene-v0",
                "CloseTopDrawerCustomInScene-v0",
                "CloseMiddleDrawerCustomInScene-v0",
                "CloseBottomDrawerCustomInScene-v0",
            ],
            "max-episode-steps" : 113,
            "additional-env-build-kwargs-variants": [
                {"shader_dir": "rt", "station_name": "mk_station2"},
                {"shader_dir": "rt", "station_name": "mk_station3"},
            ],  # 含站点名称的配置

            # 机器人的初始化配置
            "robot-init-x-range": [0.65, 0.85, 3],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [0, 0, 1],
            "obj-init-y-range": [0, 0, 1],
        },
    ]

    MOVE_NEAR = [
        {
            # 基础配置
            "scene-name": ["google_pick_coke_can_1_v4"],
            "env-name": ["MoveNearGoogleInScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        },
        {
            # 含干扰物配置
            "scene-name": ["google_pick_coke_can_1_v4"],
            "env-name": ["MoveNearGoogleInScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
            "additional-env-build-kwargs-variants": [{"no_distractor": True}],
        },
        {
            # 背景场景配置
            "scene-name": ["google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2"],
            "env-name": ["MoveNearGoogleInScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        },
        {
            # 光照配置
            "scene-name": ["google_pick_coke_can_1_v4"],
            "env-name": ["MoveNearGoogleInScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
            "additional-env-build-kwargs-variants": [
                {"slightly_darker_lighting": True},
                {"slightly_brighter_lighting": True},
            ],
        },
        {
            # 桌面纹理配置
            "scene-name": ["Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"],
            "env-name": ["MoveNearGoogleInScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        },
        {
            # 摄像头角度配置
            "scene-name": ["google_pick_coke_can_1_v4"],
            "env-name": ["MoveNearAltGoogleCameraInScene-v0", "MoveNearAltGoogleCamera2InScene-v0"],
            "robot-init-x-range": [0.35, 0.35, 1],
            "robot-init-y-range": [0.21, 0.21, 1],
            "obj-variation-mode": "episode",
            "obj-episode-range": [0, 60],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        },
    ]


    PLACE_INTO_CLOSED_DRAWER = [
        {
            # 基础配置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
            "max-episode-steps" : 200,
            "robot-init-x-range": [0.65, 0.65, 1],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [-0.08, -0.02, 3],
            "obj-init-y-range": [-0.02, 0.08, 3],
            "enable-raytracing": True,  # 启用光线追踪
            "additional-env-build-kwargs-variants": [{"model_ids": "apple"}],  # 默认的额外参数
        },
        {
            # 背景场景配置
            "scene-name": ["modern_bedroom_no_roof", "modern_office_no_roof"],
            "env-name": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
            "max-episode-steps" : 200,
            "robot-init-x-range": [0.65, 0.65, 1],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [-0.08, -0.02, 3],
            "obj-init-y-range": [-0.02, 0.08, 3],
            "additional-env-build-kwargs-variants": [
                {"shader_dir": "rt", "model_ids": "apple"}
            ],  # 启用光线追踪和指定模型ID
        },
        {
            # 光照配置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
            "max-episode-steps" : 200,
            "robot-init-x-range": [0.65, 0.65, 1],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [-0.08, -0.02, 3],
            "obj-init-y-range": [-0.02, 0.08, 3],
            "additional-env-build-kwargs-variants": [
                {"shader_dir": "rt", "light_mode": "brighter", "model_ids": "apple"},
                {"shader_dir": "rt", "light_mode": "darker", "model_ids": "apple"},
            ],  # 含光照模式设置
        },
        {
            # 新站点配置
            "scene-name": ["frl_apartment_stage_simple"],
            "env-name": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
            "max-episode-steps" : 200,
            "robot-init-x-range": [0.65, 0.65, 1],
            "robot-init-y-range": [-0.2, 0.2, 3],
            "robot-init-rot-quat-center": [0, 0, 0, 1],
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
            "obj-init-x-range": [-0.08, -0.02, 3],
            "obj-init-y-range": [-0.02, 0.08, 3],
            "additional-env-build-kwargs-variants": [
                {"shader_dir": "rt", "station_name": "mk_station2", "model_ids": "apple"},
                {"shader_dir": "rt", "station_name": "mk_station3", "model_ids": "apple"},
            ],  # 配置不同的站点
        },
    ]




    merge_dict = []

    if task == "google_robot_pick_coke_can":

        for i in range(len(COKE_CAN)):

            merge_dict.append( {**BASIC_ARGS, **COKE_CAN[i]})
    elif task == "google_robot_open_drawer" or task == "google_robot_close_drawer":
        for i in range(len(DRAWER)):

            merge_dict.append( {**BASIC_ARGS, **DRAWER[i]})
    elif task == "google_robot_move_near_v0":
        for i in range(len(MOVE_NEAR)):

            merge_dict.append( {**BASIC_ARGS, **MOVE_NEAR[i]})
    elif task == "google_robot_place_apple_in_closed_top_drawer":
        for i in range(len(PLACE_INTO_CLOSED_DRAWER)):

            merge_dict.append( {**BASIC_ARGS, **PLACE_INTO_CLOSED_DRAWER[i]})


    for i in range(len(merge_dict)):
        # import ipdb;ipdb.set_trace()
        merge_dict[i]["robot-init-xs"] = parse_range_tuple(merge_dict[i]["robot-init-x-range"])
        merge_dict[i]["robot-init-ys"] = parse_range_tuple(merge_dict[i]["robot-init-y-range"])
        merge_dict[i]["robot-init-quats"] = []
        for r in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][:3]):
            for p in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][3:6]):
                for y in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][6:]):
                    merge_dict[i]["robot-init-quats"].append((Pose(q=euler2quat(r, p, y)) * Pose(q=merge_dict[i]["robot-init-rot-quat-center"])).q)
        if merge_dict[i]["obj-variation-mode"] == "xy":
            merge_dict[i]["obj-init-xs"] = parse_range_tuple(merge_dict[i]["obj-init-x-range"])
            merge_dict[i]["obj-init-ys"] = parse_range_tuple(merge_dict[i]["obj-init-y-range"])
        if merge_dict[i]["obs-camera-name"] is not None:
            # import ipdb;ipdb.set_trace()
            if merge_dict[i]["additional-env-save-tags"] is None:
                merge_dict[i]["additional-env-save-tags"] = f"obs_camera_{merge_dict[i]['obs-camera-name']}"
            else:
                merge_dict[i]["additional-env-save-tags"] = merge_dict[i]["additional-env-save-tags"] + f"_obs_camera_{merge_dict[i]['obs-camera-name']}"

    return merge_dict


def find_suitable_args_match(task):


    BASIC_ARGS = {
        "policy-model" : "rt1",
        "policy-setup" : "google_robot",
        "ckpt-path" : None,
        "additional-env-save-tags" : None,
        "env-name": None,
        "scene-name" : None,
        "enable-raytracing" : False,
        "robot": "google_robot_static",
        "obs-camera-name" : None,
        "action-scale" : 1.0,
        "control-freq" : 3,
        "sim-freq": 513,
        "max-episode-steps" : 80,
        "rgb-overlay-path" : None,
        "robot-init-x-range" : [0.35, 0.35, 1],
        "robot-init-y-range" : [0.20, 0.20, 1],
        "robot-init-rot-quat-center" : [1, 0, 0, 0],
        "robot-init-rot-rpy-range" : [0, 0, 1, 0, 0, 1, 0, 0, 1],
        "obj-variation-mode" : "xy",
        "obj-episode-range" : [0,60],
        "obj-init-x-range" : [-0.35, -0.12, 5],
        "obj-init-y-range" : [-0.02, 0.42, 5],
        "additional-env-build-kwargs-variants" : None,
        "logging-dir" : "./results",
        "tf-memory-limit" : 3072,
        "octo-init-rng" : 0,   
    }


    COKE_CAN = [
        {
        "scene-name": ["google_pick_coke_can_1_v4"],
        "env-name": ["GraspSingleOpenedCokeCanInScene-v0"],
        "robot-init-x-range": [0.35, 0.35, 1],
        "robot-init-y-range": [0.20, 0.20, 1],
        "obj-init-x-range": [-0.35, -0.12, 5],
        "obj-init-y-range": [-0.02, 0.42, 5],
        "robot-init-rot-quat-center": [0, 0, 0, 1],
        "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
        "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png"),
        "additional-env-build-kwargs-variants": [
            {"lr_switch": True, "urdf_version": None},
            {"upright": True, "urdf_version": None},
            {"laid_vertically": True, "urdf_version": None},
            {"lr_switch": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
            {"upright": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
            {"laid_vertically": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
            {"lr_switch": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
            {"upright": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
            {"laid_vertically": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
            {"lr_switch": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
            {"upright": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
            {"laid_vertically": True, "urdf_version": "recolor_cabinet_visual_matching_1"}
        ]
        }
    ]


    DRAWER = [
            {
                # A0
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.644, 0.644, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [-0.179, -0.179, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.03, -0.03 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # A1
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.765, 0.765, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [-0.182, -0.182, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.02, -0.02 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # A2
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.889, 0.889, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [-0.203, -0.203, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.06, -0.06 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # B0
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.652, 0.652, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.009, 0.009, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # B1
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.752, 0.752, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.009, 0.009, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # B2
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.851, 0.851, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.035, 0.035, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # C0
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.665, 0.665, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.224, 0.224, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # C1
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.765, 0.765, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.222, 0.222, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.025, -0.025 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
            {
                # c2
                "scene-name": ["dummy_drawer"],
                "env-name": [
                    "OpenTopDrawerCustomInScene-v0", 
                    "OpenMiddleDrawerCustomInScene-v0", 
                    "OpenBottomDrawerCustomInScene-v0", 
                    "CloseTopDrawerCustomInScene-v0", 
                    "CloseMiddleDrawerCustomInScene-v0", 
                    "CloseBottomDrawerCustomInScene-v0"
                ],
                "max-episode-steps" : 113,
                "robot-init-x-range": [0.865, 0.865, 1],  # 每个列表为三维坐标（x坐标范围）
                "robot-init-y-range": [0.222, 0.222, 1],  # 每个列表为三维坐标（y坐标范围）
                "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
                "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.025, -0.025 ,1],  # 旋转角度
                "obj-init-x-range": [0, 0, 1],  # 物体初始化x坐标范围
                "obj-init-y-range": [0, 0, 1],  # 物体初始化y坐标范围
                "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png"),
                "enable-raytracing": True,
                "additional-env-build-kwargs-variants": [
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                    {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
                ]
            },
    ]

    MOVE_NEAR = [
       {
        # 基础配置
        "scene-name": ["google_pick_coke_can_1_v4"],
        "env-name": ["MoveNearGoogleBakedTexInScene-v0"],  # 使用env_name
        "robot-init-x-range": [0.35, 0.35, 1],  # 机器人初始化x坐标
        "robot-init-y-range": [0.21, 0.21, 1],  # 机器人初始化y坐标
        "obj-variation-mode": "episode",  # 对象变化模式
        "obj-episode-range": [0, 60],  # 对象变化的范围
        "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
        "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],  # 初始旋转角度范围
        "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png"),
        "additional-env-build-kwargs-variants": [
            {"urdf_version": "None"},
            {"urdf_version": "recolor_tabletop_visual_matching_1"},
            {"urdf_version": "recolor_tabletop_visual_matching_2"},
            {"urdf_version": "recolor_cabinet_visual_matching_1"}
        ],
        "additional-env-save-tags": "baked_except_bpb_orange"
        }
    ]


    PLACE_INTO_CLOSED_DRAWER = [
        {
        # 基础配置
            "scene-name": ["dummy_drawer"],
            "env-name": [
                "PlaceIntoClosedTopDrawerCustomInScene-v0",
                "PlaceIntoClosedMiddleDrawerCustomInScene-v0",
                "PlaceIntoClosedBottomDrawerCustomInScene-v0"
            ],
            "robot-init-x-range": [0.644, 0.644, 1],  # 每个三维坐标
            "robot-init-y-range": [-0.179, -0.179, 1],  # 每个三维坐标
            "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],  # 旋转角度范围
            "obj-init-x-range": [-0.08, -0.02, 3],  # 物体初始化x坐标范围
            "obj-init-y-range": [-0.02, 0.08, 3],  # 物体初始化y坐标范围
            "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png"),
            "additional-env-build-kwargs-variants": [
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
            ],
            "additional-env-save-tags": "baked_apple_v2"
        },
        {
            "scene-name": ["dummy_drawer"],
            "env-name": [
                "PlaceIntoClosedTopDrawerCustomInScene-v0",
                "PlaceIntoClosedMiddleDrawerCustomInScene-v0",
                "PlaceIntoClosedBottomDrawerCustomInScene-v0"
            ],
            "robot-init-x-range": [0.652, 0.652, 1],  # 每个三维坐标
            "robot-init-y-range": [0.009, 0.009, 1],  # 每个三维坐标
            "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],  # 旋转角度范围
            "obj-init-x-range": [-0.08, -0.02, 3],  # 物体初始化x坐标范围
            "obj-init-y-range": [-0.02, 0.08, 3],  # 物体初始化y坐标范围
            "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png"),
            "additional-env-build-kwargs-variants": [
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
            ],
            "additional-env-save-tags": "baked_apple_v2"
        },
        {
            "scene-name": ["dummy_drawer"],
        "env-name": [
                "PlaceIntoClosedTopDrawerCustomInScene-v0",
                "PlaceIntoClosedMiddleDrawerCustomInScene-v0",
                "PlaceIntoClosedBottomDrawerCustomInScene-v0"
            ],
            "robot-init-x-range": [0.665, 0.665, 1],  # 每个三维坐标
            "robot-init-y-range": [0.224, 0.224, 1],  # 每个三维坐标
            "robot-init-rot-quat-center": [0, 0, 0, 1],  # 初始旋转四元数
            "robot-init-rot-rpy-range": [0, 0, 1, 0, 0, 1, 0, 0, 1],  # 旋转角度范围
            "obj-init-x-range": [-0.08, -0.02, 3],  # 物体初始化x坐标范围
            "obj-init-y-range": [-0.02, 0.08, 3],  # 物体初始化y坐标范围
            "rgb-overlay-path": os.path.join(os.getcwd(),"SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png"),
            "additional-env-build-kwargs-variants": [
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_cabinet_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_1"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": "recolor_tabletop_visual_matching_2"},
                {"station_name": "mk_station_recolor", "light_mode": "simple", "disable_bad_material": True, "urdf_version": None}
            ],
            "additional-env-save-tags": "baked_apple_v2"
        }
    ]




    merge_dict = []

    if task == "google_robot_pick_coke_can":

        for i in range(len(COKE_CAN)):

            merge_dict.append( {**BASIC_ARGS, **COKE_CAN[i]})
    elif task == "google_robot_open_drawer" or task == "google_robot_close_drawer":
        for i in range(len(DRAWER)):

            merge_dict.append( {**BASIC_ARGS, **DRAWER[i]})
    elif task == "google_robot_move_near_v0":
        for i in range(len(MOVE_NEAR)):

            merge_dict.append( {**BASIC_ARGS, **MOVE_NEAR[i]})
    elif task == "google_robot_place_apple_in_closed_top_drawer":
        for i in range(len(PLACE_INTO_CLOSED_DRAWER)):

            merge_dict.append( {**BASIC_ARGS, **PLACE_INTO_CLOSED_DRAWER[i]})


    for i in range(len(merge_dict)):
        # import ipdb;ipdb.set_trace()
        merge_dict[i]["robot-init-xs"] = parse_range_tuple(merge_dict[i]["robot-init-x-range"])
        merge_dict[i]["robot-init-ys"] = parse_range_tuple(merge_dict[i]["robot-init-y-range"])
        merge_dict[i]["robot-init-quats"] = []
        for r in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][:3]):
            for p in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][3:6]):
                for y in parse_range_tuple(merge_dict[i]["robot-init-rot-rpy-range"][6:]):
                    merge_dict[i]["robot-init-quats"].append((Pose(q=euler2quat(r, p, y)) * Pose(q=merge_dict[i]["robot-init-rot-quat-center"])).q)
        if merge_dict[i]["obj-variation-mode"] == "xy":
            merge_dict[i]["obj-init-xs"] = parse_range_tuple(merge_dict[i]["obj-init-x-range"])
            merge_dict[i]["obj-init-ys"] = parse_range_tuple(merge_dict[i]["obj-init-y-range"])
        if merge_dict[i]["obs-camera-name"] is not None:
            # import ipdb;ipdb.set_trace()
            if merge_dict[i]["additional-env-save-tags"] is None:
                merge_dict[i]["additional-env-save-tags"] = f"obs_camera_{merge_dict[i]['obs-camera-name']}"
            else:
                merge_dict[i]["additional-env-save-tags"] = merge_dict[i]["additional-env-save-tags"] + f"_obs_camera_{merge_dict[i]['obs-camera-name']}"

    return merge_dict




def get_env(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    # import ipdb;ipdb.set_trace()
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }

    return env, env_reset_options
    # obs, _ = env.reset(options=env_reset_options)



def build_maniskill2_env(env_name, **kwargs):
    # Create environment
    if kwargs.get("rgb_overlay_path", None) is not None:
        if kwargs.get("rgb_overlay_cameras", None) is None:
            # Set the default camera to overlay real images for the visual-matching evaluation setting
            if "google_robot_static" in kwargs["robot"]:
                kwargs["rgb_overlay_cameras"] = ["overhead_camera"]
            elif "widowx" in kwargs["robot"]:
                kwargs["rgb_overlay_cameras"] = ["3rd_view_camera"]
            else:
                raise NotImplementedError()
    env = gym.make(env_name, **kwargs)

    return env















