import argparse
from datetime import datetime
from os import makedirs, path

import numpy as np
import numpy.random
import pandas as pd
from collections import deque
import random

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from utils import plot, preprocessImg
from gym.envs.box2d import CarRacing

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('scenario', type=int)
parser.add_argument('-o', '--obstacles', type=int, default=3)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9999)
parser.add_argument('--eps_min', type=float, default=0.05)
parser.add_argument('--lr_step', type=float, default=100_000)
parser.add_argument('--lr_decay', type=float, default=0.8)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            Input(shape=self.state_dim),
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.action_dim, activation=None)
        ])

        lr_step = args.lr_step / 50
        lr = ExponentialDecay(initial_learning_rate=args.lr, decay_steps=lr_step, decay_rate=args.lr_decay, staircase=True)
        adam = Adam(learning_rate=lr, beta_1=0.95, beta_2=0.999, epsilon=0.0001)
        model.compile(loss='mse', optimizer=adam)
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, train=True):
        if train:
            self.epsilon *= args.eps_decay
            self.epsilon = max(self.epsilon, args.eps_min)

        q_value = self.predict(state)[0]

        actions = ['NO-OP', 'LEFT', 'RIGHT', 'GAS', 'BRAKE']
        print('Q-values')
        for a, q in zip(actions, q_value): print(f'{a}: \t {q:.4f}')
        print('Chosen action:', actions[np.argmax(q_value)], end='\n\n')

        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)


class Agent:
    def __init__(self, env, info, render=True):
        self.env = env
        self.frame_stack = 4
        self.state_dim = (84, 84, self.frame_stack)
        self.action_dim = len(self.env.actions_discrete)

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

        self.skip_frames = 2
        self.stats_window_size = 50
        self.save_steps = 50
        self.horizon = 1500
        self.render = render

        self.scenario, self.obstacles, self.day, self.clock = info

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            self.model.train(states, targets)

    def train(self, max_time=1e6, pretrained=False):
        time = ep = 0
        data = pd.DataFrame(columns=('Time steps', 'Rewards', 'Tiles'))
        finished = False

        while not finished:
            observation = self.env.reset()
            s_t = preprocessImg(observation)
            state = np.stack(([s_t] * self.frame_stack), axis=-1)  # It becomes 84x84x3

            track_length = self.env.track_length

            steps = total_reward = 0
            done = False

            ep += 1

            while not done:
                action = self.model.get_action(np.expand_dims(state, axis=0))

                reward = 0
                for _ in range(self.skip_frames + 1):
                    next_state, r, sensors, done, _ = self.env.step(self.env.actions_discrete[action])
                    if self.render: self.env.render()
                    reward += r; steps += 1
                    if done: break

                if steps >= self.horizon: done = True

                s_t = np.reshape(preprocessImg(next_state), (84, 84, 1))
                next_state = np.append(s_t, state[:, :, :self.frame_stack-1], axis=-1)

                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            # EPISODE FINISHED
            print("Episode Finish")

            if self.buffer.size() >= args.batch_size: self.replay()
            self.target_update()

            dist_to_obst, speed, tile_visited_count, _ = sensors
            time += steps

            if max_time is not None and time >= max_time: finished = True

            print("TIME", time,
                  "/ STEPS", steps,
                  "/ GAME", ep,
                  "/ TILES", f"{tile_visited_count}/{track_length}",
                  "/ EP reward", round(total_reward),
                  "/ DIST to obst", round(dist_to_obst, 3),
                  "/ Speed", round(speed, 3))

            data = data.append({'Time steps': time, 'Rewards': total_reward, 'Tiles': tile_visited_count}, ignore_index=True)

            if ep % self.stats_window_size == 0 or finished:
                print("Update Rolling Statistics")
                plot(data, 'dqn', self.scenario, self.day, self.clock, self.obstacles, pretrained=pretrained)

            if ep % self.save_steps == 0 or finished:
                print("Now we save model")
                self.save(f"DQN/scenario_{self.scenario}/models/{self.day}/{self.clock}.h5")

    def test(self, max_game=100, plot_enabled=True):
        ep = 0
        data = pd.DataFrame(columns=('Episodes', 'Rewards', 'Tiles'))
        finished = False

        while not finished:
            observation = self.env.reset()
            s_t = preprocessImg(observation)
            state = np.stack(([s_t] * self.frame_stack), axis=-1)  # It becomes 84x84x3

            track_length = self.env.track_length

            steps = total_reward = 0
            done = False

            ep += 1

            while not done:
                action = self.model.get_action(np.expand_dims(state, axis=0), train=False)

                reward = 0
                for _ in range(self.skip_frames + 1):
                    next_state, r, sensors, done, _ = self.env.step(self.env.actions_discrete[action])
                    if self.render: self.env.render()
                    reward += r; steps += 1
                    if done: break

                if steps >= self.horizon: done = True

                if max_game is not None and ep >= max_game: finished = True

                s_t = np.reshape(preprocessImg(next_state), (84, 84, 1))
                next_state = np.append(s_t, state[:, :, :self.frame_stack-1], axis=-1)

                total_reward += reward
                state = next_state

            # EPISODE FINISHED
            print("Episode Finish")

            dist_to_obst, speed, tile_visited_count, _ = sensors

            print("STEPS", steps,
                  "/ GAME", ep,
                  "/ TILES", f"{tile_visited_count}/{track_length}",
                  "/ EP reward", round(total_reward),
                  "/ DIST to obst", round(dist_to_obst, 3),
                  "/ Speed", round(speed, 3))

            data = data.append({'Episodes': ep, 'Rewards': total_reward, 'Tiles': tile_visited_count}, ignore_index=True)

            if (ep % self.stats_window_size == 0 or finished) and plot_enabled:
                print("Update Rolling Statistics")
                plot(data, 'dqn', self.scenario, self.day, self.clock, self.obstacles, test=True, load_scenario=self.load_scenario)

    def load(self, name):
        self.model.model.load_weights(name)
        self.target_update()

    def save(self, name):
        self.target_model.model.save_weights(name)

    def set_info(self, load_day, load_time, load_scenario, epsilon):
        self.day = load_day
        self.clock = load_time
        self.load_scenario = load_scenario
        self.model.epsilon = epsilon


def main():
    scenario = args.scenario
    date = datetime.now()
    day = date.strftime("%b-%d")
    clock = date.strftime("%H.%M")

    info = (scenario, args.obstacles, day, clock)

    train = False  # False for test mode
    load = False  # to continue training
    plot_enabled = False  # only for testing

    seed = 73
    tf.random.set_seed(seed)

    env = CarRacing(verbose=0, scenario=scenario, obstacles=args.obstacles, seed=seed, train=train)
    agent = Agent(env, info)

    if train:
        # folder to save results
        if not path.exists(f'DQN/scenario_{scenario}/models/{day}'): makedirs(f"DQN/scenario_{scenario}/models/{day}")
        if not path.exists(f'DQN/scenario_{scenario}/statistics/{day}'): makedirs(f"DQN/scenario_{scenario}/statistics/{day}")
        if not path.exists(f'DQN/scenario_{scenario}/statistics/{day}/individuals'): makedirs(f"DQN/scenario_{scenario}/statistics/{day}/individuals")

        if load:
            # load_scenario, load_date, load_time = 4, 'Feb-26', '11.40'  # no obstacle agent
            load_scenario, load_date, load_time = 1, 'Feb-16', '17.29'
            agent.load(f'DQN/scenario_{load_scenario}/models/{load_date}/{load_time}.h5')
            # agent.model.epsilon = args.eps_min

        agent.train(pretrained=load)
    else:
        # test
        # scenario 4 with obstacles, 2.5m steps
        # load_scenario, load_date, load_time = 4, 'Feb-24', '22.28'

        # scenario 4 without obstacles, 1m steps
        load_scenario, load_date, load_time = 4, 'Feb-26', '11.40'

        # scenario 3
        # load_scenario, load_date, load_time = 3, 'Mar-01', '11.39'

        # scenario 1
        # load_scenario, load_date, load_time = 1, 'Feb-16', '17.29'

        record_video = False

        if plot_enabled:
            if not path.exists(f'DQN/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}'): makedirs(f'DQN/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}')
            if not path.exists(f'DQN/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}/individuals'): makedirs(f'DQN/Test/agent_{load_scenario}/scenario_{scenario}/{load_date}/individuals')

        if record_video:
            from gym.wrappers.record_video import RecordVideo
            env = RecordVideo(env, f'Recordings/DQN/Agent_{load_scenario}_{load_date}_{load_time}',
                              name_prefix=f'scenario{info[0]}{"_" + str(info[1]) if info[0] in (0, 4) else ""}')
            agent.env = env

        agent.load(f'DQN/scenario_{load_scenario}/models/{load_date}/{load_time}.h5')
        agent.set_info(load_date, load_time, load_scenario, epsilon=0.005)  # same final epsilon as for DFP

        agent.test(plot_enabled=plot_enabled)

    env.close()


if __name__ == "__main__":
    main()
