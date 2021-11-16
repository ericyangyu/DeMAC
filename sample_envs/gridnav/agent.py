from __future__ import print_function

import random

from sample_envs.gridnav.constants.actions import *


class Agent:
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

    def get_action(self):
        return random.choice(ACTIONS)

    def get_name(self):
        return self.name

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos
