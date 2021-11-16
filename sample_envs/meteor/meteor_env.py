import gym
import numpy as np
import yaml

from demac.src.demac.marl_env_interface import MARLEnvInterface

# Possible agent actions
LEFT, STAND, RIGHT = -1, 0, 1


class MeteorGame:
    def __init__(self):
        configs = yaml.load(open('./sample_envs/meteor/config/config.yaml', 'r'), Loader=yaml.SafeLoader)

        self.num_agents = configs['num_agents']
        self.meteor_interval = configs['meteor_interval']
        self.grid_size = configs['grid_size']
        self.max_timesteps = configs['max_timesteps']

        self.curr_meteor_interval = 0
        self.timesteps = 0

        self.grid = None

        self.reset_board()

    def reset_board(self):
        self.grid = [['-' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.timesteps = 0
        # Randomly init agent positions
        agent = 0
        while agent < self.num_agents:
            agent_i = np.random.choice(range(self.grid_size))
            if self.grid[-1][agent_i] != '-':
                continue
            self.grid[-1][agent_i] = str(agent)
            agent += 1

    def move_one_step(self, agent_actions):
        # Move agents
        self.move_agents_one_step(agent_actions)

        # Drop meteor one spot down
        collision = self.drop_meteors_one_step()
        if collision == 0:
            return 0

        # Spawn new meteor when time
        self.curr_meteor_interval += 1
        if self.curr_meteor_interval == self.meteor_interval:
            self.curr_meteor_interval = 0
            self.spawn_meteor()

        # if max timesteps hit, end
        if self.timesteps == self.max_timesteps:
            return 0

        self.timesteps += 1

        return 1

    def move_agents_one_step(self, agent_actions):
        last_row = self.grid[-1].copy()
        for agent in range(self.num_agents):
            agent_i = ''.join(last_row).find(str(agent))

            new_agent_i = min(self.grid_size - 1, max(0, agent_i + agent_actions[agent]))

            if last_row[new_agent_i] != '-':
                continue

            last_row[agent_i] = '-'
            last_row[new_agent_i] = str(agent)

        self.grid[-1] = last_row

    def drop_meteors_one_step(self):
        last_row = self.grid[-1].copy()

        # Cleanse last row of meteor
        for i in range(len(last_row)):
            if last_row[i] == '*':
                last_row[i] = '-'

        self.grid.pop()
        self.grid.insert(0, ['-' for _ in range(self.grid_size)])

        meteor_i = ''.join(self.grid[-1]).find('*')

        if meteor_i != -1:
            # Check meteor collision for last row with agents
            if last_row[meteor_i] != '-':
                last_row[meteor_i] = 'c'
                self.grid[-1] = last_row
                return 0

            last_row[meteor_i] = '*'

        self.grid[-1] = last_row

        return 1

    def spawn_meteor(self):
        # Randomly init meteor position
        self.grid[0][np.random.choice(range(self.grid_size))] = '*'

    def get_meteor_dists(self):
        meteor_dists = [self.grid_size for _ in range(self.grid_size)]
        i = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == '*':
                    meteor_dists[row] = self.grid_size - col - 1
        return meteor_dists

    def get_agent_xs(self):
        agent_xs = [None for _ in range(self.num_agents)]
        i = 0
        for cell_i in range(self.grid_size):
            cell = self.grid[-1][cell_i]
            if cell.isdigit():
                agent_xs[int(cell)] = cell_i
                i += 1

        return agent_xs

    def print_board(self):
        for row in self.grid:
            print(''.join(row))
        print()


class MeteorEnv(MARLEnvInterface):
    def __init__(self, render=False):
        super(MeteorEnv, self).__init__()

        configs = yaml.load(open('./sample_envs/meteor/config/config.yaml', 'r'), Loader=yaml.SafeLoader)

        self.num_agents = configs['num_agents']
        self.meteor_interval = configs['meteor_interval']
        self.grid_size = configs['grid_size']
        self.max_timesteps = configs['max_timesteps']
        self.render = render

        self.env = MeteorGame()

        # Must init a list of names in the order of the agents
        self.names = [str(i) for i in range(self.num_agents)]

        obs_shape = self.num_agents * self.grid_size + self.grid_size
        self.observation_space = {
            name: gym.spaces.Box(low=0, high=self.grid_size, shape=(obs_shape,), dtype=np.int32) for name in self.names
        }

        self.action_space = {
            name: gym.spaces.Discrete(3) for name in self.names
        }

        self.ep_ret = 0
        self.ep_num = 0

    def reset(self):
        if self.render:
            if self.env.timesteps == self.env.max_timesteps:
                print('\nFull score episode!\n')
            print('\nResetting...\n')
        self.ep_ret = 0
        self.ep_num += 1

        self.env.reset_board()
        obs = self.get_obs()

        if self.render:
            self.env.print_board()

        return {name: obs for name in self.names}

    def step(self, action):
        action = self.convert_action(action)

        agent_actions = [action[name] for name in self.names]
        done = self.env.move_one_step(agent_actions)
        done = False if done == 1 else True
        obs = self.get_obs()
        rew = self.get_rews(done)

        if self.render:
            self.env.print_board()

        # Update episodic return and episode number logging variables
        self.ep_ret += rew

        return {name: (obs, rew, done, {}) for name in self.names}

    # Since actor returns actions in {0,1,2}, convert to {-1,0,1}
    @staticmethod
    def convert_action(action):
        for name, act in action.items():
            act = int(act)
            if act == 0:
                action[name] = -1
            elif act == 1:
                action[name] = 0
            elif act == 2:
                action[name] = 1
        return action

    def get_obs(self):
        agent_xs = self.env.get_agent_xs()
        meteor_dists = self.env.get_meteor_dists()
        # Form agent positions as one hot encoding
        agent_ohe = [[0 for _ in range(self.grid_size)] for _ in range(self.num_agents)]
        for agent_i, agent_x in enumerate(agent_xs):
            if agent_x == None:
                continue
            agent_ohe[agent_i][agent_x] = 1
        agent_ohe = np.array(agent_ohe).flatten().tolist()
        # Form meteor distances from bottom 
        return tuple(agent_ohe + meteor_dists)

    def get_rews(self, done):
        rew = 1
        # If collision occurred, give big penalty
        if done and self.env.timesteps != self.env.max_timesteps:
            rew -= self.max_timesteps / 10
        return rew

    def render(self):
        if self.render:
            self.env.print_board()

    def close(self):
        pass

    def evaluate(self, num_eps, agents, envs):
        self.render = True
        for ep in range(num_eps):
            print()
            print(f'----- Episode {ep} -----')
            print()
            obs = self.reset()
            done = False
            while not done:
                a = {}
                for agent, env_wrapper in zip(agents, envs):
                    act, _ = agent.predict(obs[env_wrapper.name])
                    a[env_wrapper.name] = act
                step_ret = self.step(a)
                rews = {}
                for agent, env_wrapper in zip(agents, envs):
                    n = env_wrapper.name
                    obs[n] = step_ret[n][0]
                    rews[n] = step_ret[n][1]
                    done = done or step_ret[env_wrapper.name][2]
