from __future__ import print_function

import random
import sys
from collections import deque
from os import makedirs, path

import numpy as np
import numpy.random
import pandas as pd
import tensorflow as tf

from networks import Networks
from datetime import datetime
from gym.envs.box2d import CarRacing
from utils import plot, preprocessImg


def measurementify(dist_to_obst, speed, tile_visited_count, crash_count):
    return np.array([dist_to_obst / 50,
                     speed / 300,
                     tile_visited_count / 33,
                     crash_count / 2])


class DFPAgent:
    def __init__(self, state_size, measurement_size, action_size, timesteps, horizon=1000, scenario=0):

        # get size of state, measurement, action, and timestep
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.action_size = action_size
        self.timesteps = timesteps
        self.temporal_coefficients = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]

        # these are hyper parameters for the DFP
        self.gamma = 0.99  # Unused ?
        self.learning_rate = 1e-5
        self.lr_step = 150_000  # lr *= lr_decay every lr_step timesteps
        self.lr_decay = 0.8
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.005
        self.batch_size = 32
        self.observe = 2000
        self.explore = 75_000
        self.frame_per_action = 4
        self.timestep_per_train = 5  # Number of timesteps between training interval
        self.horizon = horizon

        if scenario == 0:
            self.learning_rate = 5e-6  # for full size
            self.explore = 100_000
        elif scenario == 4:
            self.learning_rate = 7e-6

        # experience replay buffer
        self.memory = deque()
        self.max_memory = 20000

        # create model
        self.model = None

        # Performance Statistics
        self.stats_window_size = 50  # window size for computing rolling statistics

    def get_action(self, state, measurement, goal, inference_goal):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            # print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            measurement = np.expand_dims(measurement, axis=0)
            goal = np.expand_dims(goal, axis=0)
            f = self.model.predict([state, measurement, goal])  # [1 x measurements * timesteps] * num_action
            f_pred = np.vstack(f)  # num action x measurements * timesteps
            obj = np.sum(np.multiply(f_pred, inference_goal), axis=1)  # num_action

            # for printing the DFP predictions
            '''
            actions = ['NO-OP', 'LEFT', 'RIGHT', 'GAS', 'BRAKE']
            for action, fp in zip(actions, f_pred): print(f'{action}: {np.around(fp.reshape(6, 4), 4)}')
            print('CHOSEN:', actions[np.argmax(obj)])
            print(np.around(obj, 4))
            print(actions)
            print()
            '''

            action_idx = np.argmax(obj)
        return action_idx

    def replay_memory(self, s_t, action_idx, m_t, m_t1, is_terminated):
        self.memory.append((s_t, action_idx, m_t, m_t1, is_terminated))

        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self, goal):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        rand_indices = np.random.choice(len(self.memory) - (self.timesteps[-1] + 1), self.batch_size)

        state_input = np.zeros(((batch_size,) + self.state_size))  # Shape batch_size, img_rows, img_cols, 4
        measurement_input = np.zeros((batch_size, self.measurement_size))
        goal_input = np.tile(goal, (batch_size, 1))
        f_action_target = np.zeros((batch_size, (self.measurement_size * len(self.timesteps))))
        action = []

        for i, idx in enumerate(rand_indices):
            future_measurements = []
            last_offset = 0
            done = False
            for j in range(self.timesteps[-1] + 1):
                if not self.memory[idx + j][4]:  # if episode is not finished
                    if j in self.timesteps:  # 1,2,4,8,16,32
                        if not done:
                            future_measurements += list((self.memory[idx + j][2] - self.memory[idx][2]))
                            last_offset = j
                        else:
                            future_measurements += list((self.memory[idx + last_offset][3] - self.memory[idx][2]))
                else:
                    done = True
                    last_offset = j
                    if j in self.timesteps:  # 1,2,4,8,16,32
                        future_measurements += list((self.memory[idx + last_offset][3] - self.memory[idx][2]))

            f_action_target[i, :] = np.array(future_measurements)
            state_input[i, :, :, :] = self.memory[idx][0]
            measurement_input[i, :] = self.memory[idx][2]
            action.append(self.memory[idx][1])

        f_target = self.model.predict([state_input, measurement_input, goal_input])  # Shape [32x18,32x18,32x18]

        for i in range(self.batch_size): f_target[action[i]][i, :] = f_action_target[i]
        loss = self.model.train_on_batch([state_input, measurement_input, goal_input], f_target)

        return loss

    # load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # seeds
    seed = 73
    tf.random.set_seed(seed)

    # settings
    render = False
    load = True
    load_file = 'scenario_4/models/Mar-30/01.03'
    scenario = int(sys.argv[1])
    obstacles = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    date = datetime.now()
    day = date.strftime("%b-%d")
    clock = date.strftime("%H.%M")

    # folder to save results
    if not path.exists(f'DFP/scenario_{scenario}/models/{day}'): makedirs(f"DFP/scenario_{scenario}/models/{day}")
    if not path.exists(f'DFP/scenario_{scenario}/statistics/{day}'): makedirs(f"DFP/scenario_{scenario}/statistics/{day}")
    if not path.exists(f'DFP/scenario_{scenario}/statistics/{day}/individuals'): makedirs(f"DFP/scenario_{scenario}/statistics/{day}/individuals")

    env = CarRacing(verbose=0, scenario=scenario, obstacles=obstacles, seed=seed)

    observation = env.reset()
    track_length = env.track_length

    action_names = env.actions_discrete_names
    actions = env.actions_discrete
    action_size = len(actions)

    measurement_size = env.measurement_size
    timesteps = [1, 2, 4, 8, 16, 32]
    goal_size = measurement_size * len(timesteps)

    img_rows, img_cols = 84, 84
    img_channels = 4  # We stack 4 frames

    max_time = 1e6  # set to None to train forever

    skip_frames = 2
    actions_per_action = skip_frames + 1

    state_size = (img_rows, img_cols, img_channels)
    agent = DFPAgent(state_size, measurement_size, action_size, timesteps, horizon=1500, scenario=scenario)

    # agents lr_step now matches the training timesteps
    lr_step = agent.lr_step / (actions_per_action * agent.timestep_per_train)

    agent.model = Networks.dfp_network(state_size, measurement_size, goal_size, action_size, len(timesteps),
                                       agent.learning_rate, lr_step, agent.lr_decay)

    x_t = preprocessImg(observation)
    s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 84x84x4
    s_t = np.expand_dims(s_t, axis=0)  # 1x84x84x4

    # MEASUREMENTS
    measurements = (env.distance_to_obstacle(), 0, env.tile_visited_count, 0)
    # (DISTANCE TO OBSTACLE, SPEED, TILES VISITED, CRASHES)

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

    goal = make_goal(relative_goal, agent.temporal_coefficients)

    # Goal for Inference (Can change during test-time)
    inference_goal = goal

    finished = False

    # Start training
    sum_reward = t = time = loss = GAME = steps = 0

    # GRAPH STUFF
    data = pd.DataFrame(columns=('Time steps', 'Rewards', 'Tiles'))

    # adds goal to filename for saving
    goal_names = ['D', 'S', 'T', 'C']
    goal_values = "_".join(f"{n}{g}" for n, g in zip(goal_names, relative_goal))
    name = f'{day}/{clock}_{goal_values}_'
    stats_goal = f'{" ".join(f"{n}:{g}" for n, g in zip(goal_names, relative_goal))}'

    if load:
        load_name = f'_{"_".join([f"{n}{g}" for n, g in zip(goal_names, relative_goal[:measurement_size])])}'
        agent.load_model(f'DFP/{load_file}{load_name}_.h5')
        name += 'pretrained_'
        # agent.epsilon = agent.final_epsilon

    while not finished:
        r_t = 0

        # Epsilon Greedy
        action_idx = agent.get_action(s_t, m_t, goal, inference_goal)

        for _ in range(skip_frames + 1):
            observation, r, measurements, finished, _ = env.step(actions[action_idx])
            if render: env.render()
            r_t += r; steps += 1
            if finished: break

        sum_reward += r_t
        if steps >= agent.horizon: finished = True

        if finished: observation = env.reset()

        x_t1 = preprocessImg(observation)
        x_t1 = np.reshape(x_t1, (1, img_rows, img_cols, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Measurement after transition
        m_t1 = measurementify(*measurements)

        # save the sample to the replay memory and decrease epsilon
        agent.replay_memory(s_t, action_idx, m_t, m_t1, finished)

        # Measurement after transition update
        m_t = m_t1

        # Do the training
        if t > agent.observe and t % agent.timestep_per_train == 0:
            loss = agent.train_minibatch_replay(goal)

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % (10000 // actions_per_action) == 0:
            print("Now we save model")
            agent.model.save_weights(f"DFP/scenario_{scenario}/models/{name}.h5", overwrite=True)

        # print info
        if t <= agent.observe: state = "observe"
        elif agent.observe < t <= agent.observe + agent.explore: state = "explore"
        else: state = "train"

        if finished:
            print("Episode Finish ")

            dist_to_obst, speed, tile_visited_count, _ = measurements
            GAME += 1; time += steps

            # reset measurements for next episode
            measurements = (env.distance_to_obstacle(), 0, env.tile_visited_count, 0)
            m_t = measurementify(*measurements)

            print("TIME", time,
                  "/ STEPS", steps,
                  "/ GAME", GAME,
                  "/ STATE", state,
                  "/ EPSILON", round(agent.epsilon, 4),
                  "/ TILES", f"{tile_visited_count}/{track_length}",
                  "/ Sum reward", round(sum_reward),
                  "/ REWARD", round(r_t, 3),
                  "/ DIST to obst", round(dist_to_obst, 3),
                  "/ Speed", round(speed, 3))

            data = data.append({'Time steps': time, 'Rewards': sum_reward, 'Tiles': tile_visited_count}, ignore_index=True)

            loss = sum_reward = steps = 0
            track_length = env.track_length

            # This sets finished to false to continue training
            if max_time is None or time < max_time: finished = False

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe or finished:
                print("Update Rolling Statistics")
                plot(data, 'dfp', scenario, day, clock, obstacles, stats_goal, pretrained=load)

            if finished:
                print("Now we save model")
                agent.model.save_weights(f"DFP/scenario_{scenario}/models/{name}.h5", overwrite=True)
                env.close()
