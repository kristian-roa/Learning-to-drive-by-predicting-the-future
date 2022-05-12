from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from networks import Networks
from os import makedirs, path

from gym.envs.box2d import CarRacing
from DFP_car_racing import DFPAgent, measurementify
from utils import plot, preprocessImg


if __name__ == "__main__":
    # seeds
    seed = 73
    tf.random.set_seed(seed)

    # settings
    record_video = True
    scenario = int(sys.argv[1])
    obstacles = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    plot_enabled = False
    pretrained = False

    # scenario 4 with obstacles, 2.5m steps
    load_scenario, load_date, load_time = 4, 'Mar-26', '22.17'

    # scenario 4a, 1m steps
    # load_scenario, load_date, load_time = 4, 'Mar-26', '02.05'

    # scenario 3
    # load_scenario, load_date, load_time = 3, 'Mar-27', '13.36'

    # scenario 1 -> 4
    # load_scenario, load_date, load_time, pretrained = 4, 'Mar-28', '02.54', True

    # scenario 1
    # load_scenario, load_date, load_time = 1, 'Mar-26', '22.45'

    if plot_enabled:
        if not path.exists(f'DFP/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}'): makedirs(f'DFP/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}')
        if not path.exists(f'DFP/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}/individuals'): makedirs(f'DFP/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}/individuals')

    env = CarRacing(verbose=0, scenario=scenario, obstacles=obstacles, seed=seed, train=False)

    observation = env.reset()
    track_length = env.track_length

    action_names = env.actions_discrete_names
    actions = env.actions_discrete
    action_size = len(actions)

    measurement_size = env.measurement_size
    timesteps = [1, 2, 4, 8, 16, 32]
    goal_size = measurement_size * len(timesteps)

    # test for 100 episodes
    max_game = 100

    img_rows, img_cols = 84, 84
    img_channels = 4  # We stack 4 frames

    skip_frames = 2
    actions_per_action = skip_frames + 1

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps, horizon=1500)

    # agents lr_step now matches the training timesteps
    lr_step = agent.lr_step / (actions_per_action * agent.timestep_per_train)

    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps),
                                       agent.learning_rate, lr_step, agent.lr_decay)

    x_t = preprocessImg(observation)
    s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 84x84x4
    s_t = np.expand_dims(s_t, axis=0)  # 1x84x84x4

    # MEASUREMENTS
    measurements = (env.distance_to_obstacle(), 0, env.tile_visited_count, 0)

    # Initial normalized measurements
    m_t = measurementify(*measurements)

    # Goal
    # (Distance, Speed, Tiles, Crashes)
    relative_goal = [1.0, 0.0, 1.0, -1.0]

    def make_goal(relative_goal, temporal_coefficients):
        goal = np.array(relative_goal * len(timesteps))
        goal = goal.reshape(len(timesteps), measurement_size) * np.asarray(temporal_coefficients).reshape(6, 1)
        goal = goal.flatten()
        return goal

    # goal = make_goal(relative_goal, [1] * len(timesteps))
    goal = make_goal(relative_goal, agent.temporal_coefficients)

    # Goal for Inference (Can change during test-time)
    inference_goal = goal
    # inference_goal = make_goal([1.0, 0.0, 1.0, -1.0], agent.temporal_coefficients)

    finished = False

    # Start training
    sum_reward = GAME = steps = 0

    # GRAPH STUFF
    data = pd.DataFrame(columns=('Episodes', 'Rewards', 'Tiles'))

    goal_names = ['D', 'S', 'T', 'C']
    name = f'_{"_".join(f"{n}{g}" for n, g in zip(goal_names, relative_goal))}'
    stats_goal = f'{" ".join(f"{n}:{g}" for n, g in zip(goal_names, relative_goal))}'

    agent.load_model(f'DFP/scenario_{load_scenario}/models/{load_date}/{load_time}{name}{"_pretrained" if pretrained else ""}_.h5')
    agent.epsilon = agent.final_epsilon  # 0.005

    if record_video:
        from gym.wrappers.record_video import RecordVideo
        env = RecordVideo(env, f'Recordings/DFP/Agent_{load_scenario}_{load_date}_{load_time}',
                          name_prefix=f'scenario{scenario}{"_" + str(obstacles) if scenario in (0, 4) else ""}')

    while not finished:
        r_t = 0

        # Epsilon Greedy
        action_idx = agent.get_action(s_t, m_t, goal, inference_goal)

        for _ in range(skip_frames + 1):
            observation, r, measurements, finished, _ = env.step(actions[action_idx])
            env.render()
            r_t += r; steps += 1
            if finished: break

        sum_reward += r_t

        if steps >= agent.horizon: finished = True
        if finished: observation = env.reset()

        x_t1 = preprocessImg(observation)
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Measurement after transition
        m_t = measurementify(*measurements)

        s_t = s_t1

        # print info
        state = "TEST"

        if finished:
            print("Episode Finish ")

            dist_to_obst, speed, tile_visited_count, _ = measurements
            GAME += 1

            # reset measurements for next episode
            measurements = (env.distance_to_obstacle(), 0, env.tile_visited_count, 0)
            m_t = measurementify(*measurements)

            print("STEPS", steps,
                  "/ GAME", GAME,
                  "/ STATE", state,
                  "/ TILES", f"{tile_visited_count}/{track_length}",
                  "/ Sum reward", round(sum_reward),
                  "/ REWARD", round(r_t, 3),
                  "/ ACTION", action_names[action_idx],
                  "/ DIST to obst", round(dist_to_obst, 3),
                  "/ Speed", round(speed, 3))

            data = data.append({'Episodes': GAME, 'Rewards': sum_reward, 'Tiles': tile_visited_count}, ignore_index=True)

            sum_reward = steps = 0
            track_length = env.track_length

            # This sets finished to false to continue testing
            if max_game is None or GAME < max_game: finished = False

            # Save Agent's Performance Statistics
            if (GAME % agent.stats_window_size == 0 or finished) and plot_enabled:
                print("Update Rolling Statistics")
                plot(data, 'dfp', scenario, load_date, load_time, obstacles, stats_goal, test=True, load_scenario=load_scenario)
