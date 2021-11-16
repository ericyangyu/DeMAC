import random

import gym
import yaml
import numpy as np

from demac.src.demac.marl_env_interface import MARLEnvInterface
from sample_envs.gridnav.agent import Agent
from sample_envs.gridnav.node import Node
from sample_envs.gridnav.constants.actions import *
from sample_envs.gridnav.constants.colors import *
from sample_envs.gridnav.constants.node_names import *


class Grid(MARLEnvInterface):
    def __init__(self):
        """
        Initializes the grid environment for training.
        """
        super(Grid, self).__init__()

        configs = yaml.load(open('./sample_envs/gridnav/config/config.yaml', 'r'), Loader=yaml.SafeLoader)

        self.width = configs['width']
        self.height = configs['height']
        self.sensor_range = configs['sensor_range']
        self.max_timesteps = configs['max_timesteps']
        self.num_agents = configs['num_agents']
        self.puddle_prob = configs['puddle_prob']

        self.grid = None
        self.agents = None  # {name: Agent}
        self.puddle_poses = None  # [(row, col)]
        self.goal_pos = None  # (row, col)
        self.timestep = None
        self.done = None
        self.won = None
        self.action_map = None

        self.color_map = {str(name): YELLOW for name in range(self.num_agents)}
        self.color_map[EMPTY] = BLACK
        self.color_map[GOAL] = ORANGE
        self.color_map[PUDDLE] = BLUE
        self.color_map[COLLISION] = RED

        # Populate DeMAC variables
        self.names = [str(i) for i in range(self.num_agents)]
        # sensor_range x sensor_range - 1 grid around agent to observe, where each box has OHE of empty, agent,
        # puddle, or goal. Add 2 for agent goal distance along each axis
        obs_dim = 4 * ((self.sensor_range * 2 + 1) ** 2 - 1) + 2
        self.observation_space = {
            name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for name in self.names
        }
        self.action_space = {
            name: gym.spaces.Discrete(len(ACTIONS)) for name in self.names
        }
        self.agent_envs = {}

    def reset(self, rand_init=True):
        """
        Initializes the grid.
        :param rand_init: Whether or not we randomly initialize the puddles
        :return:
        """
        self.grid = {(r, c): Node((r, c), self.color_map) for r in range(self.height) for c in range(self.width)}
        self.agents = {}
        self.puddle_poses = []
        self.timestep = 0
        self.done = False
        self.won = False
        self.action_map = None

        if rand_init:
            for node in self.grid.values():
                if random.random() < self.puddle_prob:
                    node.set_occupant(PUDDLE)
                    self.puddle_poses.append(node.get_pos())
                else:
                    node.set_occupant(EMPTY)

        # Initialize goal in valid positions
        while True:
            self.goal_pos = (random.randrange(self.height), random.randrange(self.width))
            if self.grid[self.goal_pos].get_occupant() == EMPTY:
                self.grid[self.goal_pos].set_occupant(GOAL)
                break

        # Initialize agents in valid positions
        for i in range(self.num_agents):
            while True:
                agent_pos = (random.randrange(self.height), random.randrange(self.width))
                if self.grid[agent_pos].get_occupant() == EMPTY:
                    self.agents[str(i)] = Agent(str(i), agent_pos)
                    self.grid[agent_pos].set_occupant(str(i))
                    break

        obs_map = {}
        for agent_name in self.names:
            obs_map[agent_name] = self.__get_obs(agent_name)
        return obs_map

    def step(self, action_map):
        """
        Steps the environment by a timestep. Assumes actions are scalars that map to the ACTIONS constant array
        :param action_map:
        :return:
        """
        obs_map, r_map = {}, {}

        self.action_map = self.__extract_action_map(action_map)
        for agent_name, action in self.action_map.items():
            self.done |= self.move_agent(agent_name, action)

        # Check if we have won yet
        self.won = False not in [agent.get_pos() == self.goal_pos for agent in self.agents.values()]
        self.done |= self.won

        # Check if we are at max_timesteps
        if self.timestep >= self.max_timesteps:
            self.done = True

        self.timestep += 1

        for agent_name in self.names:
            obs_map[agent_name] = self.__get_obs(agent_name)
            r_map[agent_name] = self.__get_r(agent_name)

        # Form observations and rewards for agents
        return {name: (obs_map[name], r_map[name], self.done, {}) for name in self.names}

    def move_agent(self, agent_name, action):
        """
        Moves the agent "agent_name" with given action.
        :param agent_name: The name of the agent
        :param action: The action to take
        :return: True if there is a collision with the current agent, else False
        """
        has_collided = False
        agent_pos = self.agents[agent_name].get_pos()
        move_pos = (agent_pos[0] + action[0], agent_pos[1] + action[1])

        # Only move if within bounds or not collided already
        if 0 <= move_pos[0] < self.height and 0 <= move_pos[1] < self.width and \
                self.grid[agent_pos].get_occupant() != COLLISION and \
                self.grid[agent_pos].get_occupant() != GOAL:

            move_pos_occupant = self.grid[move_pos].get_occupant()

            if move_pos_occupant != EMPTY and move_pos_occupant != GOAL and move_pos_occupant != agent_name:
                self.grid[move_pos].set_occupant(COLLISION)
                self.grid[agent_pos].set_occupant(COLLISION)
                has_collided = True
            elif move_pos_occupant != GOAL and agent_name != move_pos_occupant:
                self.grid[move_pos].set_occupant(self.agents[agent_name].get_name())

            if self.grid[agent_pos].get_occupant() != COLLISION and agent_name != move_pos_occupant:
                self.grid[agent_pos].set_occupant(EMPTY)
                self.agents[agent_name].set_pos(move_pos)

        self.done = has_collided

        return has_collided

    def print_board(self):
        """
        A makeshift print for the board.
        :return:
        """
        print()
        for i in range(self.height):
            row = []
            for j in range(self.width):
                pos = (i, j)
                node = self.grid[pos]
                row.append(node.get_occupant())
            print(row)
        print()

    def get_done(self):
        return self.done

    @staticmethod
    def __extract_action_map(action_map):
        return {name: ACTIONS[ind] for name, ind in action_map.items()}

    def __get_obs(self, agent_name):
        """
        Forms the observation for some agent.
        :param agent_name:
        :return:
        """
        obs = []

        agent_pos = self.agents[agent_name].get_pos()

        # First, form OHE of each valid point within range of current agent pos
        # OHE will be of form (empty, agent, puddle, goal) and a zero 4-vector if it's out of range
        coords = []
        for r in range(-self.sensor_range, self.sensor_range + 1):
            for c in range(-self.sensor_range, self.sensor_range + 1):
                if (r, c) == (0, 0):
                    continue
                elif 0 <= agent_pos[0] + r < self.height and 0 <= agent_pos[1] + c < self.width:
                    coords.append((agent_pos[0] + r, agent_pos[1] + c))
                else:
                    coords.append(None)

        for coord in coords:
            if not coord:
                obs += [0, 0, 0, 0]
                continue

            occupant = self.grid[coord].get_occupant()
            if occupant == EMPTY:
                obs += [1, 0, 0, 0]
            elif occupant == PUDDLE:
                obs += [0, 0, 1, 0]
            elif occupant == GOAL:
                obs += [0, 0, 0, 1]
            else:
                obs += [0, 1, 0, 0]

        # Second, get goal distance from agent as (r, c)
        goal_dist = (self.goal_pos[0] - agent_pos[0], self.goal_pos[1] - agent_pos[1])
        obs += goal_dist

        return obs

    def __get_r(self, agent_name):
        """
        Forms the reward for the given agent.
        :param agent_name: The name of the agent
        :return: a scalar reward
        """
        agent_pos = self.agents[agent_name].get_pos()

        # Edge case: if all agents have won, reward them with 2 * number of timesteps left
        if self.won:
            return 2 * (self.max_timesteps - self.timestep)

        # Edge case: if agent has won already, set reward to 1
        if agent_pos == self.goal_pos:
            return 1

        # Edge case: if this agent has collided, set reward to penalty
        if self.done and self.grid[agent_pos].get_occupant == COLLISION:
            return -30

        # Edge case: do not reward agent for sitting still
        if self.action_map[agent_name] == (0, 0):
            return 0

        # Calculate default reward as a function of the euclidean distance between agent and goal
        goal_dist = np.sqrt((self.goal_pos[0] - agent_pos[0]) ** 2 + (self.goal_pos[1] - agent_pos[1]) ** 2)
        return np.e ** -goal_dist
