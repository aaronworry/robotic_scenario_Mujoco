import os
import numpy as np
import math

COLORS = {'red': [0.4, 0, 0], 'green': [0, 0.4, 0], 'blue': [0, 0, 0.4], 'black': [0, 0, 0], 'pink': [0.4, 0, 0.4],
          'yellow': [0.4, 0.4, 0], 'cyan': [0, 0.4, 0.4], 'white': [1., 1., 1.]}


class BaseObject():
    def __init__(self, env, position, thing_type, color='black'):
        self.env = env
        self.type = thing_type
        self.position = position
        self.init_position = position
        self.finished = False
        self.id = self._createBody()
        self.color_name = color
        self.color = COLORS[color]
        self.set_color(self.color_name)

    def _createBody(self):
        if self.type == "cube":
            cube_path = os.path.join(os.path.dirname(__file__),"../../assets/conveyor/blue_cube.urdf")
            return p.loadURDF(cube_path, self.position, useFixedBase = 0)
        elif self.type == 'cylinder':
            cylinder_path = os.path.join(os.path.dirname(__file__),"../../assets/conveyor/yellow_cylinder.urdf")
            return p.loadURDF(cylinder_path, self.position, useFixedBase = 0)
        elif self.type == 'sphere':
            ball_path = os.path.join(os.path.dirname(__file__),"../../assets/conveyor/sphere.urdf")
            return p.loadURDF(ball_path, self.position, useFixedBase = 0)


    def get_position(self):
        position, _ = p.getBasePositionAndOrientation(self.id, physicsClientId=self.env.client)
        self.position = position
        return self.position

    def set_color(self, color):
        self.color_name = color
        self.color = COLORS[color]
        p.changeVisualShape(self.id, 0, textureUniqueId=-1, rgbaColor=(self.color[0], self.color[1], self.color[2], 0.5), physicsClientId=self.env.client)


    def reset(self, pose):  #0.3
        position = pose[0]
        orientation = pose[1]
        p.resetBasePositionAndOrientation(self.id, position, orientation, physicsClientId=self.env.client)


