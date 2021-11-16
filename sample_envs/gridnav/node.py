import pygame

from sample_envs.gridnav.constants.node_names import *

class Node:
    def __init__(self, pos, color_map):
        # Save pygame specific parameters here
        self.pos = pos
        self.blit_pos = [self.pos[1] * 15, self.pos[0] * 15]
        self.image = pygame.Surface((15, 15))
        self.rect = self.image.get_rect(topleft=self.blit_pos)

        # Initialize node
        self.color_map = color_map
        self.occupant = EMPTY
        self.color = self.color_map[EMPTY]

    def get_pos(self):
        return self.pos

    def get_occupant(self):
        return self.occupant

    def set_occupant(self, occupant):
        """
        Sets the current node's occupant and color. Also takes into account user clicks for dynamic puddle creation.
        :param occupant: the name of the occupant
        :return: 1 if successful, 0 if failed
        """
        self.occupant = occupant
        self.color = self.color_map[occupant]

    def draw(self, screen):
        """
        Draws the node to the screen
        :param screen: The screen of the game
        :return:
        """
        self.image.fill(self.color)
        screen.blit(self.image, self.rect)
