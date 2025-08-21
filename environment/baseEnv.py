from functools import partial
from os import path
from typing import Optional

import gym
import numpy as np

# Only provide abstract function

"""
环境需要完成的功能

初始化
实例化机器人 self.robots
实例化目标物 self.objects
实例化与仿真器交互的 self.simulator

外界的读取：observation, reward  比如 
            机器人的状态 ，目标        从实例化的 robot 对象获取
            机器人的传感器参数                  robot 对象 的传感器对象 获取
            物体的状态                        object 对象获取
            
外界的输入： action
            机器人的控制指令
                数值仿真中：发送给 robot 类，让其执行并更新   matplot可视化
                仿真器中：  由env分配给simulator的对象 , 当然可以将速度指令输入给机器人，机器人算出力之后，发送给simulator
                
其他核心功能：
        读取仿真器的数据，分配给 robot object 对象
        数值仿真中 直接从 robot object 对象中获取            matplot
        
        重置：
            仿真器中 : robot.reset()   object.reset()   将他们的信息发送给simulator, 然后simulator reset
            数值仿真中： robot.reset()   object.reset()  可视化
          
"""

"""
物体需要完成的功能
    记录初始的状态，各种自身的信息
    提供读函数给env 用于读取状态
    reset 功能
    提供写函数给env, env从simulator 获取的信息，写入object
    
    如果在数值仿真中，需要 step 功能，根据输入量和时间 更新自身的信息
"""

"""
传感器需要完成的功能
    此部分只适用于 仿真器
    提供读，写函数给env  env从仿真获取的数据写入到传感器对象，env从传感器对象获取的信息发送到observation
"""

"""
机器人需要完成的功能
    记录初始的状态，各种自身的信息
    提供读函数给env 用于读取状态
    reset 功能
    提供写函数给env, env从simulator 获取的信息，写入robot
    
    如果在数值仿真中，需要 step 功能，根据输入量和时间 更新自身的信息
    
    机器人需要集成各种模组
"""

"""
仿真器需要完成的功能
    与各种仿真器接入
    环境的实例化
    写，读操作
"""

class BaseEnv(gym.Env):
    """Superclass for all Env environments."""
    """ Manage the robots and objects in Enviroment."""

    def __init__(self, numerical_simulation = False, simulator_type = "Mujoco"):
        """
        observe_space
        action_space
        state_ub
        state_lb
        control_ub
        control_lb
        """
        if numerical_simulation == True:
            simulator_type = "Numerical"
            
        self.simulator = None
        
        if simulator_type == "Mujoco":
            self.simulator = MujocoWrapper(**kwarg)
        elif simulator_type == "Issac":
            self.simulator = IssacWrapper(**kwarg)
            
        self.initialization()
        
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


    
        



