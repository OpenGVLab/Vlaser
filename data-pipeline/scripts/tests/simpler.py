import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def main(
    task: str = "google_robot_pick_coke_can",
):
    env = simpler_env.make("google_robot_pick_coke_can")
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    print("Reset info", reset_info)
    print("Instruction", instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    print("Image shape and dtype", image.shape, image.dtype)

    done, truncated = False, False
    step = 0
    while not (done or truncated):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        action = env.action_space.sample()  # replace this with your policy inference
        if step == 0:
            print("Action shape and dtype", action.shape, action.dtype)
            print("Action", action)
        obs, reward, done, truncated, info = env.step(
            action
        )  # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
            instruction = new_instruction
            print("New Instruction", instruction)

        print("Step", step, "Reward", reward, "Done", done, "Truncated", truncated)
        step += 1

    episode_stats = info.get("episode_stats", {})
    print("Episode stats", episode_stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="google_robot_pick_coke_can")
    args = parser.parse_args()

    print("=====================================")
    print("Available environments:")
    print(simpler_env.ENVIRONMENTS)
    print("=====================================")
    print("Running task", args.task)
    # [
    #     "google_robot_pick_coke_can",
    #     "google_robot_pick_horizontal_coke_can",
    #     "google_robot_pick_vertical_coke_can",
    #     "google_robot_pick_standing_coke_can",
    #     "google_robot_pick_object",
    #     "google_robot_move_near_v0",
    #     "google_robot_move_near_v1",
    #     "google_robot_move_near",
    #     "google_robot_open_drawer",
    #     "google_robot_open_top_drawer",
    #     "google_robot_open_middle_drawer",
    #     "google_robot_open_bottom_drawer",
    #     "google_robot_close_drawer",
    #     "google_robot_close_top_drawer",
    #     "google_robot_close_middle_drawer",
    #     "google_robot_close_bottom_drawer",
    #     "google_robot_place_in_closed_drawer",
    #     "google_robot_place_in_closed_top_drawer",
    #     "google_robot_place_in_closed_middle_drawer",
    #     "google_robot_place_in_closed_bottom_drawer",
    #     "google_robot_place_apple_in_closed_top_drawer",
    #     "widowx_spoon_on_towel",
    #     "widowx_carrot_on_plate",
    #     "widowx_stack_cube",
    #     "widowx_put_eggplant_in_basket",
    # ]
    main(args.task)
