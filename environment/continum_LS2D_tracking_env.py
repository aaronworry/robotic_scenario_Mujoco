import time
import numpy as np

from gym.spaces import Box

from environment.baseEnv import BaseEnv
from simulator.matplot_2d_wrapper import MatplotConnect
from robots.logarithmic_spiral_2d_robot import TwoStringContinuumRobot


class Target():
    def __init__(self):
        pass

class LS2D_tracking_Env(BaseEnv, MatplotConnect):