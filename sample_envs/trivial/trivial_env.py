import gym
import random
import numpy as np
import sys

import yaml

from demac.src.demac.marl_env_interface import MARLEnvInterface

sys.path.append('../../demac/src')

class TrivialEnv(MARLEnvInterface):
    def __init__(self):
        super(TrivialEnv, self).__init__()

        configs = yaml.load(open('./sample_envs/trivial/config/config.yaml', 'r'), Loader=yaml.SafeLoader)

        self.names = [f'{i}' for i in range(configs['num_agents'])]

        # Save number of agents
        self.num_agents = configs['num_agents']

        self.observation_space = {
            name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) for name in self.names
        }

        self.action_space = {
            name: gym.spaces.Discrete(2) for name in self.names
        }

    def reset(self):
        return {name: (random.random()) for name in self.names}

    def step(self, a):
        done = random.random() < 0.005
        return {name: (0.5, 1, done, {}) for name in self.names}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def evaluate(self, num_eps, agents, envs):
        for ep in range(num_eps):
            print()
            print(f'----- Episode {ep} -----')
            print()
            obs = self.reset()
            done = False
            while not done:
                a = {}
                for agent, env_wrapper in zip(agents, envs):
                    act, _ = agent.predict([obs[env_wrapper.name]])
                    a[env_wrapper.name] = act
                step_ret = self.step(a)
                rews = {}
                for agent, env_wrapper in zip(agents, envs):
                    n = env_wrapper.name
                    obs[n] = step_ret[n][0]
                    rews[n] = step_ret[n][1]
                    done = done or step_ret[env_wrapper.name][2]
