from functools import partial
from os import path
from typing import Optional

import gym
import numpy as np
from gym import error, spaces
from gym.spaces import Space

# Only provide abstract function

class BaseEnv(gym.Env):
    """Superclass for all Env environments."""
    """ Manage the robots and objects in Enviroment."""

    def __init__(self, robot):
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


    
        



