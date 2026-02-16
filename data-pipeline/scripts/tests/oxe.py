""" "
Visualize and download OXE dataset

Reference: https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb

"""

import os

import tensorflow_datasets as tfds
from PIL import Image

DATASETS = [
    "fractal20220817_data",
    # "kuka",
    "bridge",
    # "taco_play",
    # "jaco_play",
    # "berkeley_cable_routing",
    # "roboturk",
    # "nyu_door_opening_surprising_effectiveness",
    # "viola",
    # "berkeley_autolab_ur5",
    # "toto",
    # "language_table",
    # "columbia_cairlab_pusht_real",
    # "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    # "nyu_rot_dataset_converted_externally_to_rlds",
    # "stanford_hydra_dataset_converted_externally_to_rlds",
    # "austin_buds_dataset_converted_externally_to_rlds",
    # "nyu_franka_play_dataset_converted_externally_to_rlds",
    # "maniskill_dataset_converted_externally_to_rlds",
    # "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    # "ucsd_kitchen_dataset_converted_externally_to_rlds",
    # "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    # "austin_sailor_dataset_converted_externally_to_rlds",
    # "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    # "usc_cloth_sim_converted_externally_to_rlds",
    # "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    # "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    # "utokyo_saytap_converted_externally_to_rlds",
    # "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    # "utokyo_xarm_bimanual_converted_externally_to_rlds",
    # "robo_net",
    # "berkeley_mvp_converted_externally_to_rlds",
    # "berkeley_rpt_converted_externally_to_rlds",
    # "kaist_nonprehensile_converted_externally_to_rlds",
    # "stanford_mask_vit_converted_externally_to_rlds",
    # "tokyo_u_lsmo_converted_externally_to_rlds",
    # "dlr_sara_pour_converted_externally_to_rlds",
    # "dlr_sara_grid_clamp_converted_externally_to_rlds",
    # "dlr_edan_shared_control_converted_externally_to_rlds",
    # "asu_table_top_converted_externally_to_rlds",
    # "stanford_robocook_converted_externally_to_rlds",
    # "eth_agent_affordances",
    # "imperialcollege_sawyer_wrist_cam",
    # "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    # "uiuc_d3field",
    # "utaustin_mutex",
    # "berkeley_fanuc_manipulation",
    # "cmu_play_fusion",
    # "cmu_stretch",
    # "berkeley_gnm_recon",
    # "berkeley_gnm_cory_hall",
    # "berkeley_gnm_sac_son",
]


def dataset2path(dataset_name):
    if dataset_name == "robo_net":
        version = "1.0.0"
    elif dataset_name == "language_table":
        version = "0.0.1"
    else:
        version = "0.1.0"
    return f"{os.environ['VLA_DATA_DIR']}/{dataset_name}/{version}"


def as_gif(images, path="temp.gif"):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open(path, "rb").read()
    return gif_bytes


def visualize_image(
    dataset="bridge_dataset",
    display_key="image_0",
    data_dir=f"{os.environ['VLA_DATA_DIR']}/resize_224",
):
    ds, ds_info = tfds.load(
        name=dataset,
        data_dir=data_dir,
        download=False,
        split="train[1000:2000]",
        shuffle_files=True,
        with_info=True,
    )
    if display_key not in ds_info.features["steps"]["observation"]:
        raise ValueError(
            f"The key {display_key} was not found in this dataset.\n"
            + "Please choose a different image key to display for this dataset.\n"
            + "Here is the observation spec:\n"
            + str(ds_info.features["steps"]["observation"])
        )

    # save ds_info in text
    with open(f"temp/{dataset}_info.txt", "w") as f:
        f.write(str(ds_info))

    # inspect data
    iterator = iter(ds)
    while 1:
        episode = next(iterator)

        instructions = [
            step["language_instruction"].numpy().decode("utf-8")
            for step in episode["steps"]
        ]
        images = [step["observation"][display_key] for step in episode["steps"]]
        images = [Image.fromarray(image.numpy()) for image in images]
        proprios = [step["observation"]["state"] for step in episode["steps"]]
        if "carrot" in instructions[0].lower():
            print(instructions)
            print(proprios)

            # print image info and save one
            print(f"Image shape: {images[0].size}")
            images[0].save(f"temp/{dataset}_sample_img.png")
            breakpoint()

        # check instructions
        print(instructions[0])


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bridge_dataset")
    parser.add_argument("--display_key", type=str, default="image_0")
    parser.add_argument(
        "--data_dir", type=str, default=f"{os.environ['VLA_DATA_DIR']}/resize_224"
    )
    args = parser.parse_args()

    visualize_image(**vars(args))
