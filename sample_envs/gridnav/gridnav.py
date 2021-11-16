import time

import pygame
import yaml

from demac.src.demac.marl_env_interface import MARLEnvInterface
from sample_envs.gridnav.constants.colors import *
from sample_envs.gridnav.constants.node_names import *
from sample_envs.gridnav.grid import Grid


class GridNav(MARLEnvInterface):
    def __init__(self):
        super(GridNav, self).__init__()

        configs = yaml.load(open('./sample_envs/gridnav/config/config.yaml', 'r'), Loader=yaml.SafeLoader)

        self.pixel_size = configs['pixel_size']
        self.width = configs['width']
        self.height = configs['height']
        self.sensor_range = configs['sensor_range']
        self.max_timesteps = configs['max_timesteps']
        self.num_agents = configs['num_agents']
        self.puddle_prob = configs['puddle_prob']

        # Initialize pygame stuff
        pygame.init()
        pygame.display.set_caption("Multi Agent Grid World")
        self.clock = pygame.time.Clock()
        self.screen_res = [self.width * self.pixel_size, self.height * self.pixel_size]
        self.font = pygame.font.SysFont("Calibri", self.pixel_size)
        self.screen = pygame.display.set_mode(self.screen_res, pygame.HWSURFACE, 32)

        self.grid = Grid()
        self.grid.reset()
        self.color_map = {}
        self.surfaces = {}  # {name: (surface, text_surface)}

        # Initialize DeMAC specific variables
        self.names = self.grid.names
        self.observation_space = self.grid.observation_space
        self.action_space = self.grid.action_space

    def reset(self):
        self.grid = Grid()
        ret = self.grid.reset()
        self.color_map = self.grid.color_map

        # Initialize surfaces for agents, goal, puddles, and collision
        self.surfaces = {agent_name: self.__get_surface(agent_name) for agent_name in self.grid.agents.keys()}
        self.surfaces[GOAL] = self.__get_surface(GOAL)
        self.surfaces[PUDDLE] = self.__get_surface(PUDDLE)
        self.surfaces[COLLISION] = self.__get_surface(COLLISION)

        return ret

    def step(self, action_map):
        return self.grid.step(action_map)

    def evaluate(self, num_eps, agents, envs):
        for ep in range(num_eps):
            print()
            print(f'----- Episode {ep} -----')
            print()
            obs = self.reset()
            done = False
            while not done:
                self.draw()
                self.clock.tick(60)

                time.sleep(0.1)

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

            self.draw()
            self.clock.tick(60)
            time.sleep(1)

    def draw(self):
        self.screen.fill(BLACK)

        # Draw agents
        for agent_name, agent in self.grid.agents.items():
            background_surface, text_surface = self.surfaces[agent_name]
            pos = agent.get_pos()

            has_collided = self.grid.grid[pos].get_occupant() == COLLISION
            if has_collided:
                background_surface, _ = self.surfaces[COLLISION]

            background_blit_pos = [pos[1] * self.pixel_size, pos[0] * self.pixel_size]
            background_rect = background_surface.get_rect(topleft=background_blit_pos)
            self.screen.blit(background_surface, background_rect)

            text_blit_pos = [pos[1] * self.pixel_size + self.pixel_size // 3,
                             pos[0] * self.pixel_size + self.pixel_size // 4]
            text_rect = text_surface.get_rect(topleft=text_blit_pos)
            self.screen.blit(text_surface, text_rect)

        # Draw puddles
        for pos in self.grid.puddle_poses:
            background_surface, text_surface = self.surfaces[PUDDLE]

            has_collided = self.grid.grid[pos].get_occupant() == COLLISION
            if has_collided:
                background_surface, text_surface = self.surfaces[COLLISION]

            background_blit_pos = [pos[1] * self.pixel_size, pos[0] * self.pixel_size]
            background_rect = background_surface.get_rect(topleft=background_blit_pos)
            self.screen.blit(background_surface, background_rect)

        # Draw goal
        background_surface, text_surface = self.surfaces[GOAL]
        background_blit_pos = [self.grid.goal_pos[1] * self.pixel_size, self.grid.goal_pos[0] * self.pixel_size]
        background_rect = background_surface.get_rect(topleft=background_blit_pos)
        self.screen.blit(background_surface, background_rect)

        # Draw column and row lines
        for i in range(self.height):
            pygame.draw.line(self.screen, LINE_COLOR, (i * self.pixel_size, 0),
                             (i * self.pixel_size, i * self.pixel_size * self.width))
        for i in range(self.width):
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * self.pixel_size),
                             (i * self.pixel_size * self.height, i * self.pixel_size))

        pygame.display.update()

    def __get_surface(self, name):
        background_surface, text_surface = None, None

        color = self.color_map[name]

        background_surface = pygame.Surface((self.pixel_size, self.pixel_size))
        background_surface.fill(color)

        # Add text to agent
        if name != GOAL and name != PUDDLE:
            text_color = (255 - color[0], 255 - color[1], 255 - color[2])
            text_surface = self.font.render(name, False, text_color)

        return background_surface, text_surface
