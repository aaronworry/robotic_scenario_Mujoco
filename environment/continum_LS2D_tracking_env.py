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
    def __init__(self):
        """
        observe_space
        action_space
        state_ub
        state_lb
        control_ub
        control_lb
        """
        pass
        
    def initialization(self):
        """
        initial the environment
        add robot with initial_state
        add object with initial_state
        """
        raise NotImplementedError
        
        
    def reset(self):
        """
        set the environment to initial state
        set robot to initial_state
        set object to initial_state
        """
        raise NotImplementedError
        
    def step(self):
        """
        update state of environment, including objects and robots
        """
        raise NotImplementedError
        
    def reward(self):
        """
        calculate the reward based on states and actions
        """
        raise NotImplementedError
        
    def get_obs(self):
        """
        get observation for other algorithms
        """
        raise NotImplementedError