"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    # 如果traj_len是100, window_size是50, 则chunk_indices是
    # [[-49, -48, -47, ...,  -2,  -1,   0],
    #  [-48, -47, -46, ...,  -1,   0,   1],
    #  [-47, -46, -45, ...,   0,   1,   2],
    #        ...,
    #  [ 48,  49,  50, ...,  95,  96,  97],
    #  [ 49,  50,  51, ...,  96,  97,  98],
    #  [ 50,  51,  52, ...,  97,  98,  99]]

    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    # 获取第一列
    first_column = chunk_indices[:, 0]

    # 生成布尔掩码（mask），仅保留第一列 >= 0 的行
    mask = first_column >= 0

    # 过滤表格
    # [[ 0,  1,  2, ..., 47, 48, 49],
    #  [ 1,  2,  3, ..., 48, 49, 50],
    #  [ 2,  3,  4, ..., 49, 50, 51],
    #   ...,
    #  [48, 49, 50, ..., 95, 96, 97],
    #  [49, 50, 51, ..., 96, 97, 98],
    #  [50, 51, 52, ..., 97, 98, 99]]
    filtered_chunk_indices = tf.boolean_mask(chunk_indices, mask)


    # 如果future_action_window_size是0, 则action_chunk_indices是 #理论上我们不需要如果future_action_window_size
    # [[-49, -48, -47, ...,  -1,   0],
    #  [-48, -47, -46, ...,   0,   1],
    #  [-47, -46, -45, ...,   1,   2],
    #    ...,
    #  [ 48,  49,  50, ...,  96,  97],
    #  [ 49,  50,  51, ...,  97,  98],
    #  [ 50,  51,  52, ...,  98,  99]]

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )

    # 获取第一列
    first_column = action_chunk_indices[:, 0]

    # 生成布尔掩码（mask），仅保留第一列 >= 0 的行
    mask = first_column >= 0

    # 过滤表格
    # [[ 0,  1,  2, ..., 47, 48, 49],
    #  [ 1,  2,  3, ..., 48, 49, 50],
    #  [ 2,  3,  4, ..., 49, 50, 51],
    #   ...,
    #  [48, 49, 50, ..., 95, 96, 97],
    #  [49, 50, 51, ..., 96, 97, 98],
    #  [50, 51, 52, ..., 97, 98, 99]]
    filtered_action_chunk_indices = tf.boolean_mask(action_chunk_indices, mask)




    # # [[0, 0, 0, ..., 0, 0, 0],
    # #  [0, 0, 0, ..., 0, 0, 1],
    # #  [0, 0, 0, ..., 0, 1, 2],
    # #  ...,
    # #  [48, 49, 50, ..., 95, 96, 97],
    # #  [49, 50, 51, ..., 96, 97, 98],
    # #  [50, 51, 52, ..., 97, 98, 99]]
    #
    # floored_chunk_indices = tf.maximum(chunk_indices, 0)
    floored_chunk_indices = filtered_chunk_indices
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        #[99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        # 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        # 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        # 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        # 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        # 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
        goal_timestep = tf.fill([traj_len], traj_len - 1)




    # floored_action_chunk_indices = tf.minimum(tf.maximum(filtered_action_chunk_indices, 0), goal_timestep[:, None])
    floored_action_chunk_indices = filtered_action_chunk_indices  # 因为我们的future_action_window_size是0，所以肯定在范围内
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # # if no absolute_action_mask was provided, assume all actions are relative
    # if "absolute_action_mask" not in traj and future_action_window_size > 0:
    #     logging.warning(
    #         "future_action_window_size > 0 but no absolute_action_mask was provided. "
    #         "Assuming all actions are relative for the purpose of making neutral actions."
    #     )
    # absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    # neutral_actions = tf.where(
    #     absolute_action_mask[:, None, :],
    #     traj["action"],  # absolute actions are repeated (already done during chunking)
    #     tf.zeros_like(traj["action"]),  # relative actions are zeroed
    # )
    #
    # # actions past the goal timestep become neutral  # 不会超出goal timestep
    # action_past_goal = action_chunk_indices > goal_timestep[:, None]
    # traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
